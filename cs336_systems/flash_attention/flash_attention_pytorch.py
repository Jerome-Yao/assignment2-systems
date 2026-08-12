import torch
from einops import rearrange, einsum
import math


class FlashAttentionImpl(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal):
        assert (
            q.shape[2] == k.shape[2] == v.shape[2]
        ), "q, k, v must have the same d_model"
        batch, q_len, d_model = q.shape
        _, k_len, _ = k.shape
        scale = d_model**-0.5
        device = q.device

        q_tile = 64
        k_tile = 64

        num_q_tiles = math.ceil(q_len / q_tile)
        num_k_tiles = math.ceil(k_len / k_tile)

        Output = torch.empty(
            (batch, q_len, d_model), device=device, dtype=torch.float32
        )
        L = torch.empty((batch, q_len), device=device, dtype=torch.float32)

        for i in range(num_q_tiles):
            q_tile_start, q_tile_end = i * q_tile, min((i + 1) * q_tile, q_len)
            cur_q_tile_size = q_tile_end - q_tile_start

            Q_i = q[:, q_tile_start:q_tile_end, :].float()
            O_i = torch.zeros(
                (batch, cur_q_tile_size, d_model), device=device, dtype=torch.float32
            )
            L_i = torch.zeros(
                (batch, cur_q_tile_size), device=device, dtype=torch.float32
            )
            M_i = torch.full(
                (batch, cur_q_tile_size),
                -float("inf"),
                device=device,
                dtype=torch.float32,
            )

            for j in range(num_k_tiles):
                k_tile_start, k_tile_end = j * k_tile, min((j + 1) * k_tile, k_len)
                K_j = k[:, k_tile_start:k_tile_end, :]
                V_j = v[:, k_tile_start:k_tile_end, :]

                S_ij = (
                    einsum(
                        Q_i,
                        K_j,
                        "batch q_tile_len d_model, batch k_tile_len d_model -> batch q_tile_len k_tile_len",
                    )
                    * scale
                )

                if causal:
                    q_indices = torch.arange(q_tile_start, q_tile_end, device=device)
                    k_indices = torch.arange(k_tile_start, k_tile_end, device=device)
                    causal_mask = q_indices[:, None] >= k_indices[None, :]
                    S_ij = torch.where(causal_mask, S_ij, -float("inf"))

                m_i_new = torch.maximum(M_i, S_ij.max(dim=-1).values)
                P_ij = torch.exp(S_ij - m_i_new.unsqueeze(-1))

                delta_exp_m = torch.exp(M_i - m_i_new)

                L_i = delta_exp_m * L_i + P_ij.sum(dim=-1)

                O_i = delta_exp_m.unsqueeze(-1) * O_i + P_ij @ V_j

                M_i = m_i_new

            O_i = O_i / L_i[..., None]
            Output[:, q_tile_start:q_tile_end, :] = O_i
            L[:, q_tile_start:q_tile_end] = M_i + torch.log(L_i)

        ctx.save_for_backward(q, k, v, Output, L)
        ctx.causal = causal
        return Output

    @staticmethod
    def backward(ctx, grad_outputs):
        q, k, v, O, L = ctx.saved_tensors
        is_causal = ctx.causal

        dQ, dK, dV = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)

        batch_size, N_q, d_head = q.shape
        _, N_k, _ = k.shape
        scale = d_head**-0.5

        D = torch.sum(grad_outputs * O, dim=-1)
        B_q, B_k = 64, 64
        T_q, T_k = math.ceil(N_q / B_q), math.ceil(N_k / B_k)

        for j in range(T_k):
            k_start, k_end = j * B_k, min((j + 1) * B_k, N_k)
            k_j, v_j = k[:, k_start:k_end, :], v[:, k_start:k_end, :]
            for i in range(T_q):
                q_start, q_end = i * B_q, min((i + 1) * B_q, N_q)
                q_i = q[:, q_start:q_end, :]
                dO_i = grad_outputs[:, q_start:q_end, :]
                L_i = L[:, q_start:q_end]
                D_i = D[:, q_start:q_end]

                S_ij = q_i @ k_j.transpose(-1, -2) * scale
                if is_causal:
                    q_indices = torch.arange(q_start, q_end, device=q.device)
                    k_indices = torch.arange(k_start, k_end, device=k.device)
                    causal_mask = q_indices[:, None] >= k_indices[None, :]
                    S_ij = torch.where(causal_mask, S_ij, -float("inf"))

                P_ij = torch.exp(S_ij - L_i[..., None])

                dV[:, k_start:k_end, :] += P_ij.transpose(-1, -2) @ dO_i
                dP_ij = dO_i @ v_j.transpose(-1, -2)
                dS_ij = P_ij * (dP_ij - D_i[..., None])
                dQ[:, q_start:q_end, :] += (dS_ij * scale) @ k_j
                dK[:, k_start:k_end, :] += (dS_ij.transpose(-1, -2) * scale) @ q_i

        return dQ, dK, dV, None
