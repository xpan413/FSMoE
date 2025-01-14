import argparse
import torch
import torch.nn.functional as F

# Constants
MEMORY_SIZE = 4096 * 14336 * 16
NUM_ITERATIONS = 100
WARMUP_ITERATIONS = 20


def initialize_tensor(shape: tuple, device: str) -> torch.Tensor:
    """Helper function to initialize random tensors on the specified device."""
    return torch.randn(shape, device=device)


def benchmark_matrix_multiplication(tokens: int, model: int, hidden: int) -> None:
    # Initialize random tensors on CUDA
    matrix_a = initialize_tensor((tokens, model), device="cuda:0")
    matrix_b = initialize_tensor((hidden, model), device="cuda:0")

    # Allocate pinned and CUDA memory
    pinned_storage = torch.UntypedStorage(MEMORY_SIZE).pin_memory()
    cuda_storage = torch.UntypedStorage(MEMORY_SIZE, device="cuda:0")

    # Warm-up phase to stabilize CUDA performance
    for _ in range(WARMUP_ITERATIONS):
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


def main() -> None:
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
    benchmark_matrix_multiplication(args.tokens, args.model, args.hidden)


if __name__ == "__main__":
    main()
