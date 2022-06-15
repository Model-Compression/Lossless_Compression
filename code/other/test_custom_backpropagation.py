import torch  # import main library
from torch.autograd import \
    Function  # import Function to create custom activations


def sigma(x, a):
    '''
    define sigma function
    '''
    if x > a:
        y = 1
    else:
        y = 0
    return y


def binary_zero(s1, s2, b1):
    return lambda x: b1 * (sigma(x, s2) + sigma(-x, -s1))


def binary_last(s1, s2, b1, b2):
    return lambda x: b1 * (sigma(x, s2) + sigma(-x, -s1)) + \
                b2 * (sigma(x, s1) + sigma(-x, -s2) -1)


class binary_zero(Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, s1: int, s2: int, b1: int):
        f = binary_zero(s1, s2, b1)
        out = f(input)
        ctx.save_for_backward(input)
        ctx.s1 = s1
        ctx.s2 = s2
        ctx.b1 = b1
        return out

    '''
    here to define backpropogation style
    '''

    @staticmethod
    def backward(ctx, grad_out):
        input = ctx.saved_tensors
        return grad_out * input


if __name__ == "__main__":
    a = torch.tensor(1., requires_grad=True, dtype=torch.double)
    b = torch.tensor(2., requires_grad=True, dtype=torch.double)
    c = 4
    # d = Func.apply(a, b, c)
