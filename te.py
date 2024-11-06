import torch


class IndexSelectFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, index):
        # Store input for backward computation
        ctx.save_for_backward(index)
        ctx.input_shape = input.shape
        ctx.dim = dim

        # Perform index_select operation
        return torch.index_select(input, dim, index)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        (index,) = ctx.saved_tensors
        input_shape = ctx.input_shape
        dim = ctx.dim

        # Initialize gradient for the input tensor
        grad_input = torch.zeros(input_shape, dtype=grad_output.dtype, device=grad_output.device)

        # Scatter the gradients into the appropriate locations
        grad_input.index_add_(dim, index, grad_output)

        # No gradients for dim and index as they are not tensors
        return grad_input, None, None


# Example usage
input = torch.rand(((32, 8)), requires_grad=True)
index = torch.tensor([0, 2])
dim = 0

# Apply custom function
output = IndexSelectFunction.apply(input, dim, index)
output.sum().backward()

# Check gradients
print("Input:", input)
print("Index:", index)
print("Output:", output)
print("Gradients:", input.grad)
