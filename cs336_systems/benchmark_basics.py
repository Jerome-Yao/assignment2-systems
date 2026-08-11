from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW, get_cosine_lr
import torch
import torch.amp
import timeit
from contextlib import nullcontext
import pandas as pd
import logging
from statistics import mean, stdev
import nvtx

logging.basicConfig(
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

model_configs = {
    "small": {
        "size": "small",
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "size": "medium",
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "size": "large",
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "size": "xl",
        "d_model": 1600,
        "d_ff": 6400,
        "num_layers": 48,
        "num_heads": 25,
    },
    "2.7B": {
        "size": "2.7B",
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
}

vocab_size = 10_000
batch_size = 4
context_lengths = [128, 256, 512, 1024]
context_lengths = [512]
rope_theta = 10000.0
warmup_steps = 5
benchmark_steps = 10
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)


def benchmark(
    model: torch.nn.Module,
    x,
    y,
    mode,
    model_type,
    context_length,
    mixed_precision=True,
):
    logging.info(
        f"start run model: {model_type}, mode: {mode}, context_length: {context_length}, mixed_precision: {mixed_precision}"
    )
    model.train()
    optimizer = AdamW(model.parameters())
    max_lr = 1e-3
    min_lr = 1e-5
    warmup_iters = warmup_steps
    total_iters = warmup_iters + benchmark_steps
    global_step = 0
    lossfn = torch.nn.CrossEntropyLoss()

    def precision_context():
        return (
            torch.amp.autocast(device, dtype=torch.bfloat16)
            if mixed_precision
            else nullcontext()
        )

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        else:
            pass

    def step_forward():
        with nvtx.annotate("forward", domain="benchmark", color="green"):
            with torch.no_grad():
                start = timeit.default_timer()
                with precision_context():
                    _ = model(x)
                sync()
                forward_takes = (timeit.default_timer() - start) * 1000
                return {"forward_takes": forward_takes, "total_takes": forward_takes}

    def step_forward_and_backward():
        optimizer.zero_grad()
        start = timeit.default_timer()

        with precision_context():
            with nvtx.annotate("forward", domain="benchmark", color="green"):
                out: torch.Tensor = model(x)
                sync()

            forward_time = timeit.default_timer()
            forward_takes = (forward_time - start) * 1000

            with nvtx.annotate("loss", domain="benchmark", color="yellow"):
                loss: torch.Tensor = lossfn(out.view(-1, vocab_size), y.view(-1))
                sync()

        loss_time = timeit.default_timer()
        loss_takes = (loss_time - forward_time) * 1000

        with nvtx.annotate("backward", domain="benchmark", color="red"):
            loss.backward()
            sync()

        backward_time = timeit.default_timer()
        backward_takes = (backward_time - loss_time) * 1000

        with nvtx.annotate("optimizer", domain="benchmark", color="purple"):
            optimizer.step()
            sync()

        optimizer_time = timeit.default_timer()
        optimizer_takes = (optimizer_time - backward_time) * 1000
        total_takes = (optimizer_time - start) * 1000

        return {
            "total_takes": total_takes,
            "forward_takes": forward_takes,
            "loss_takes": loss_takes,
            "backward_takes": backward_takes,
            "optimizer_takes": optimizer_takes,
        }

    step_fn = step_forward if mode == "forward" else step_forward_and_backward
    times = []

    logging.info(
        f"start warmup model: {model_type}, mode: {mode}, context_length: {context_length} ==============="
    )

    for warmup_step in range(warmup_steps):
        lr = get_cosine_lr(global_step, max_lr, min_lr, warmup_iters, total_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr
        try:
            time_spend = step_fn()
        except Exception as e:
            logging.error(e)
            raise
        logging.info(
            "warmup step_%s mode %s time_spend: %s", warmup_step, mode, time_spend
        )
        global_step += 1
    times = []
    logging.info(
        f"start benchmark model: {model_type}, mode: {mode}, context_length: {context_length}, mixed_precision: {mixed_precision} ==============="
    )

    for step in range(benchmark_steps):
        lr = get_cosine_lr(global_step, max_lr, min_lr, warmup_iters, total_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr
        try:
            if step == 0:
                with nvtx.annotate(
                    "profile_step",
                    domain="benchmark",
                    color="blue",
                ):
                    time_spend = step_fn()
            else:
                time_spend = step_fn()
        except torch.cuda.OutOfMemoryError as e:
            logging.error("train step_%s mode %s CUDA OOM: %s", step, mode, e)
            torch.cuda.empty_cache()
            raise
        global_step += 1
        times.append(time_spend)
        logging.info("train step_%s mode %s time_spend: %s", step, mode, time_spend)

    forward_time = []
    loss_time = []
    backward_time = []
    optimizer_time = []
    total_time = []
    for time_spend in times:
        forward_time.append(time_spend.get("forward_takes", 0))
        loss_time.append(time_spend.get("loss_takes", 0))
        backward_time.append(time_spend.get("backward_takes", 0))
        optimizer_time.append(time_spend.get("optimizer_takes", 0))
        total_time.append(time_spend.get("total_takes", 0))
    # Aggregate and return stats dict safely
    stats = {
        "mean_total": mean(total_time),
        "std_total": stdev(total_time),
        "mean_forward": mean(forward_time),
        "std_forward": stdev(forward_time),
        "mean_backward": mean(backward_time),
        "std_backward": stdev(backward_time),
        "mean_loss": mean(loss_time),
        "std_loss": stdev(loss_time),
        "mean_optimizer": mean(optimizer_time),
        "std_optimizer": stdev(optimizer_time),
    }
    logging.info(
        f"benchmark model_{model_type},mode_{mode},context_length_{context_length} mean_total: {stats['mean_total']} ms, std_total: {stats['std_total']} ms"
    )
    return stats


def main():
    logging.info(f"device type {device}")
    model_name = "small"
    modes = ["forward_backward"]
    configs_to_run = {}
    if model_name == "all":
        configs_to_run = model_configs
    else:
        if model_name not in model_configs:
            raise ValueError(f"Model type '{model_name}' not found")
        model_config = model_configs[model_name]
        configs_to_run = {model_name: model_config}

    for cfg in configs_to_run:
        logging.info(f"{cfg}")

    results = []

    for mode in modes:
        for model_type, config in configs_to_run.items():
            for context_length in context_lengths:
                logging.info(
                    f"Running {config['size']} model, mode: [{mode}], context_length: {context_length}..."
                )

                model = BasicsTransformerLM(
                    vocab_size=vocab_size,
                    context_length=context_length,
                    d_model=config["d_model"],
                    num_layers=config["num_layers"],
                    num_heads=config["num_heads"],
                    d_ff=config["d_ff"],
                    rope_theta=rope_theta,
                ).to(device=device)

                x = torch.randint(
                    0, vocab_size, (batch_size, context_length), device=device
                )
                y = torch.randint(
                    0, vocab_size, (batch_size, context_length), device=device
                )

                try:
                    stats = benchmark(model, x, y, mode, model_type, context_length)
                except Exception as e:
                    logging.error(
                        config["size"],
                        mode,
                        context_length,
                        e,
                    )
                    stats = {
                        "mean_total": float("nan"),
                        "std_total": float("nan"),
                        "mean_forward": float("nan"),
                        "std_forward": float("nan"),
                        "mean_backward": float("nan"),
                        "std_backward": float("nan"),
                        "mean_loss": float("nan"),
                        "std_loss": float("nan"),
                        "mean_optimizer": float("nan"),
                        "std_optimizer": float("nan"),
                    }

                logging.info(
                    f"model {config['size']} [{mode}]: Avg Time = {stats['mean_total']:.6f}ms, Std Dev = {stats['std_total']:.6f}ms"
                )
                del model, x, y
                if device == "cuda":
                    torch.cuda.empty_cache()
                elif device == "mps":
                    torch.mps.empty_cache()

                results.append(
                    {
                        "model_size": config["size"],
                        "Mode": mode,
                        "Context Length": context_length,
                        "Avg Time (ms)": round(stats["mean_total"], 6),
                        "Std Dev (ms)": round(stats["std_total"], 6),
                        "Avg Forward (ms)": round(stats["mean_forward"], 6),
                        "Std Dev Forward (ms)": round(stats["std_forward"], 6),
                        "Avg Backward (ms)": round(stats["mean_backward"], 6),
                        "Std Dev Backward (ms)": round(stats["std_backward"], 6),
                        "Avg Loss (ms)": round(stats["mean_loss"], 6),
                        "Std Dev Loss (ms)": round(stats["std_loss"], 6),
                        "Avg Optimizer (ms)": round(stats["mean_optimizer"], 6),
                        "Std Dev Optimizer (ms)": round(stats["std_optimizer"], 6),
                    }
                )

    logging.info("\nBenchmark Results:")
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    save_file = f"benchmark_baseline_results_{model_name}.md"
    # Save to file
    with open(save_file, "w") as f:
        f.write(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
