from torchvision import transforms

def enc_advprop(x):
    normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    return normalize(x)

def dec_advprop(x):
    normalize = transforms.Lambda(lambda img: (img + 1.0) / 2.0)
    return normalize(x)

class NormalizeEf(Transform):
    "Normalize/denorm batch of `TensorImage`"
    order=99
    def __init__(self, axes=(0,2,3)): self.axes = axes

    @classmethod
    def from_advprop(cls, dim=1, ndim=4, cuda=True): return cls(*broadcast_vec(dim, ndim, cuda=cuda))

    def encodes(self, x:TensorImage): return enc_advprop(x)
    def decodes(self, x:TensorImage):
        f = to_cpu if x.device.type=='cpu' else noop
        return (dec_advprop(x))

    _docs=dict(encodes="Normalize batch", decodes="Denormalize batch") 