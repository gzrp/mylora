from singa import layer
from singa import model
from singa import tensor
from singa import opt
from singa import device
import argparse
import numpy as np

import lora

np_dtype = {"float16": np.float16, "float32": np.float32}

singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}

class MLPLoRA(model.Model):
    def __init__(self, in_features=10, perceptron_size=16, num_classes=10):
        super(MLPLoRA, self).__init__()
        self.in_features = in_features
        self.perceptron_size = perceptron_size
        self.num_classes = num_classes
        self.relu = layer.ReLU()
        self.linear1 = lora.LoRALinear(in_features=in_features, out_features=perceptron_size, r=8)
        self.linear2 = lora.LoRALinear(in_features=perceptron_size, out_features=num_classes, r=8)
        self.softmax_cross_entropy = layer.SoftMaxCrossEntropy()

    def forward(self, inputs):
        y = self.linear1(inputs)
        y = self.relu(y)
        y = self.linear2(y)
        return y

    def train_one_batch(self, x, y, dist_option, spars):
        out = self.forward(x)
        loss = self.softmax_cross_entropy(out, y)

        if dist_option == 'plain':
            self.optimizer(loss)
        elif dist_option == 'half':
            self.optimizer.backward_and_update_half(loss)
        elif dist_option == 'partialUpdate':
            self.optimizer.backward_and_partial_update(loss)
        elif dist_option == 'sparseTopK':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=True,
                                                      spars=spars)
        elif dist_option == 'sparseThreshold':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=False,
                                                      spars=spars)
        return out, loss

    def train(self, mode=True):
        super().train(mode=mode)
        self.linear1.train(mode)
        self.linear2.train(mode)

    def eval(self):
        self.train(mode=False)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

if __name__ == '__main__':
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        choices=['float32', 'float16'],
                        default='float32',
                        dest='precision')
    parser.add_argument('-g',
                        '--disable-graph',
                        default='True',
                        action='store_false',
                        help='disable graph',
                        dest='graph')
    parser.add_argument('-m',
                        '--max-epoch',
                        default=1001,
                        type=int,
                        help='maximum epochs',
                        dest='max_epoch')
    args = parser.parse_args()

    # generate the boundary
    # generate the boundary
    f = lambda x: (5 * x + 1)
    bd_x = np.linspace(-1.0, 1, 200)
    bd_y = f(bd_x)

    # choose one precision
    precision = singa_dtype[args.precision]
    np_precision = np_dtype[args.precision]


    dev = device.get_default_device()
    sgd = opt.SGD(0.5, 0.9, 1e-5, dtype=singa_dtype[args.precision])
    tx = tensor.Tensor((400, 2), dev, precision)
    ty = tensor.Tensor((400,), dev, tensor.int32)
    model = MLPLoRA(in_features=2, perceptron_size=3, num_classes=2)
    model.set_optimizer(sgd)
    model.compile([tx], is_train=True, use_graph=args.graph, sequential=True)
    model.train()
    print(model.get_params())
    for i in range(10):
        # generate the training data
        x = np.random.uniform(-1, 1, 400)
        y = f(x) + 2 * np.random.randn(len(x))
        # convert training data to 2d space
        label = np.asarray([5 * a + 1 > b for (a, b) in zip(x, y)]).astype(np.int32)
        data = np.array([[a, b] for (a, b) in zip(x, y)], dtype=np_precision)
        tx.copy_from_numpy(data)
        ty.copy_from_numpy(label)
        out, loss = model(tx, ty, 'plain', spars=None)
        print("training loss = ", tensor.to_numpy(loss)[0])
        print(model.get_params())
    print(model.get_params())
    model.eval()
    print(model.get_params())
    model.train()
    print(model.get_params())
    model.eval()
    for i in range(3):
        # generate the training data
        x = np.random.uniform(-1, 1, 400)
        y = f(x) + 2 * np.random.randn(len(x))
        # convert training data to 2d space
        label = np.asarray([5 * a + 1 > b for (a, b) in zip(x, y)]).astype(np.int32)
        data = np.array([[a, b] for (a, b) in zip(x, y)], dtype=np_precision)
        tx.copy_from_numpy(data)
        ty.copy_from_numpy(label)
        out = model(tx)
        print("eval out = ", tensor.to_numpy(out)[0])
    print(model.get_params())


