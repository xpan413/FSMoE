import argparse

import torch
import torch.nn.functional as F


def main(tokens, model, hidden):
    # Initialize random tensors on CUDA
    matrix_a = torch.randn((tokens, model), device="cuda:0")
    matrix_b = torch.randn((hidden, model), device="cuda:0")

    # Constants
    MEMORY_SIZE = 4096 * 14336 * 16
    NUM_ITERATIONS = 100

    # Allocate pinned and CUDA memory
    pinned_storage = torch.UntypedStorage(MEMORY_SIZE).pin_memory("cuda:0")
    cuda_storage = torch.UntypedStorage(MEMORY_SIZE, device="cuda:0")

    # Warm-up phase to stabilize CUDA performance
    for _ in range(20):
        F.linear(matrix_a, matrix_b)

    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Add a large CUDA task to avoid CPU launch bottleneck
    cuda_storage.copy_(pinned_storage, non_blocking=True)

    # Record the start time
    start_event.record()

    # Perform matrix multiplication for a set number of iterations
    for _ in range(NUM_ITERATIONS):
        F.linear(matrix_a, matrix_b)

    # Record the end time
    end_event.record()

    # Wait for all kernels to finish
    torch.cuda.synchronize()

    # Calculate and print the average elapsed time per iteration
    average_time = start_event.elapsed_time(end_event) / NUM_ITERATIONS
    print(f"Average time per iteration: {average_time:.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix multiplication benchmarking.")
    parser.add_argument(
        "--tokens", type=int, required=True, help="Number of tokens (rows of matrix_a)."
    )
    parser.add_argument(
        "--model",
        type=int,
        required=True,
        help="Size of model (columns of matrix_a and rows of matrix_b).",
    )
    parser.add_argument(
        "--hidden", type=int, required=True, help="Hidden size (columns of matrix_b)."
    )

    args = parser.parse_args()
    main(args.tokens, args.model, args.hidden)
