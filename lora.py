import math

from singa import tensor
from singa import autograd
from singa import layer
from singa import model



class LoRALinear(layer.Layer):
    r""" LoRA implemented in a linear layer
        in_features: the dimension of the input
        out_features: the dimension of the output
        r: rank
        lora_alpha:
        lora_dropout:
        **kwargs:
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            merge_weights: bool = True,
            bias: bool = True,
            **kwargs
    ):
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.merge_weights = merge_weights
        self.bias = bias
        self.merged = False

    def freeze_pretrained_weight(self, freeze: bool = True):
        self.W.requires_grad = not freeze
        self.W.stores_grad = not freeze
        if self.b is not None:
            self.b.requires_grad = not freeze
            self.b.stores_grad = not freeze

    def freeze_lora_weight(self, freeze: bool = True):
        if self.lora_A is not None:
            self.lora_A.requires_grad = not freeze
            self.lora_A.stores_grad = not freeze
        if self.lora_B is not None:
            self.lora_B.requires_grad = not freeze
            self.lora_B.stores_grad = not freeze


    def initialize(self, x):
        self.in_features = x.shape[1]
        w_shape = (self.in_features, self.out_features)
        b_shape = (self.out_features,)

        self.W = tensor.Tensor(shape=w_shape,
                              dtype=x.dtype,
                              requires_grad=True,
                              stores_grad=True)
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        self.W.gaussian(0.0, std)

        if self.bias:
            self.b = tensor.Tensor(shape=b_shape,
                            dtype=x.dtype,
                            requires_grad=True,
                            stores_grad=True)
            self.b.set_value(0.0)
        else:
            self.b = None

        # actual trainable parameters
        if self.r > 0:
            lora_A_shape = (self.r, self.in_features)
            lora_B_shape = (self.out_features, self.r)
            self.lora_A = tensor.Tensor(
                shape=lora_A_shape,
                dtype=x.dtype,
                requires_grad=True,
                stores_grad=True
            )
            self.lora_B = tensor.Tensor(
                shape=lora_B_shape,
                dtype=x.dtype,
                requires_grad=True,
                stores_grad=True
            )
            std = math.sqrt(2.0 / (self.in_features + self.out_features))
            # initialize A the same way as the default for nn.Linear and B to zero
            # this is different than what is described in the paper but should not affect performance
            self.lora_A.gaussian(0.0, std)
            # self.lora_B.gaussian(0.0, std)
            self.lora_B.set_value(0.0)
            self.scaling = tensor.Tensor(shape=(1,), requires_grad=False, stores_grad=False)
            self.scaling.set_value(1.0 * self.lora_alpha / self.r)
            # Freezing the pre-trained weight matrix
            self.freeze_pretrained_weight(freeze=True)

    def forward(self, x):
        if self.b:
            self.device_check(x, self.W, self.b)
            self.dtype_check(x, self.W, self.b)
        else:
            self.device_check(x, self.W)
            self.dtype_check(x, self.W)

        assert x.shape[1] == self.W.shape[0], (
                "Linear layer expects input features size %d received %d" %
                (self.W.shape[0], x.shape[1]))


        if self.r > 0 and not self.merged:
            y1 = autograd.matmul(x, self.W)
            if self.bias:
                y1 = autograd.add_bias(y1, self.b, axis=0)
            y2 = autograd.dropout(x, self.lora_dropout)
            y2 = autograd.matmul(y2, autograd.transpose(self.lora_A, (1, 0)))
            y2 = autograd.matmul(y2, autograd.transpose(self.lora_B, (1, 0)))
            y2 = autograd.mul(y2, self.scaling)
            y = autograd.add(y1, y2)
            return y
        else:
            y = autograd.matmul(x, self.W)
            if self.bias:
                y = autograd.add_bias(y, self.b, axis=0)
            return y

    def train(self, mode: bool = True):
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    delta = tensor.mult(self.lora_A.transpose((1, 0)), self.lora_B.transpose((1, 0))) * self.scaling
                    self.W.data -= delta.data
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    delta = tensor.mult(self.lora_A.transpose((1, 0)), self.lora_B.transpose((1, 0))) * self.scaling
                    self.W.data += delta.data
                self.merged = True

    def get_params(self):
        params = {self.W.name: self.W}
        if self.bias:
            params[self.b.name] = self.b
        if self.r > 0 :
            params[self.lora_A.name] = self.lora_A
            params[self.lora_B.name] = self.lora_B
        return params

    def set_params(self, parameters):
        self.W.copy_from(parameters[self.W.name])
        if self.bias:
            self.b.copy_from(parameters[self.b.name])
        if self.r > 0 :
            self.lora_A.copy_from(parameters[self.lora_A.name])
            self.lora_B.copy_from(parameters[self.lora_B.name])


def mark_only_lora_as_trainable(model: model.Model, requires_grad: bool = False) -> None:
    for n, p in model.get_params():
        if "lora_" not in n:
            p.requires_grad = requires_grad
        else:
            p.requires_grad = not requires_grad


def enable_lora(model: model.Model) -> None:
    mark_only_lora_as_trainable(model)

def disable_lora(model: model.Model) -> None:
    mark_only_lora_as_trainable(model, True)


if __name__ == '__main__':

    la = LoRALinear(2 ,3, 8)
    print("jjj")