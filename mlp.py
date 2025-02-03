from singa import layer
from singa import model
from singa import tensor
from singa import opt
from singa import device
import argparse
import numpy as np

np_dtype = {"float16": np.float16, "float32": np.float32}

singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}


class MLP(model.Model):

    def __init__(self, data_size=10, perceptron_size=100, num_classes=10):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.dimension = 2

        self.relu = layer.ReLU()
        self.linear1 = layer.Linear(perceptron_size)
        self.linear2 = layer.Linear(num_classes)
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

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


def create_model(pretrained=False, **kwargs):
    """Constructs a CNN model.

    Args:
        pretrained (bool): If True, returns a pre-trained model.

    Returns:
        The created CNN model.
    """
    model = MLP(**kwargs)

    return model


__all__ = ['MLP', 'create_model']

if __name__ == "__main__":
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
    f = lambda x: (5 * x + 1)
    bd_x = np.linspace(-1.0, 1, 200)
    bd_y = f(bd_x)



    # choose one precision
    precision = singa_dtype[args.precision]
    np_precision = np_dtype[args.precision]



    dev = device.get_default_device()
    sgd = opt.SGD(0.1, 0.9, 1e-5, dtype=singa_dtype[args.precision])
    tx = tensor.Tensor((400, 2), dev, precision)
    ty = tensor.Tensor((400,), dev, tensor.int32)
    model = MLP(data_size=2, perceptron_size=3, num_classes=2)

    # attach model to graph
    model.set_optimizer(sgd)
    model.compile([tx], is_train=True, use_graph=args.graph, sequential=True)
    model.train()

    for i in range(10):
        # generate the training data
        x = np.random.uniform(-1, 1, 400)
        y = f(x) + 2 * np.random.randn(len(x))
        # convert training data to 2d space
        label = np.asarray([5 * a + 1 > b for (a, b) in zip(x, y)]).astype(np.int32)
        data = np.array([[a, b] for (a, b) in zip(x, y)], dtype=np_precision)
        tx.copy_from_numpy(data)
        ty.copy_from_numpy(label)
        print(model.get_states())
        out, loss = model(tx, ty, 'plain', spars=None)

        print("training loss = ", tensor.to_numpy(loss)[0])


    # model.save_states("/tmp/aa.pt")
    # model.load_states("/tmp/aa.pt")

    print("hello")
    print("done")
