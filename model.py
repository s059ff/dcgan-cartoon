import chainer
import chainer.functions as F
import chainer.links as L


class Generator(chainer.Chain):

    def __init__(self):
        super(Generator, self).__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            # Deconvolution2D(input channels, output channels, kernel size, stride, padding)
            self.f0 = L.Linear(100, 4 * 4 * 240, initialW=w)
            self.dc1 = L.Deconvolution2D(240, 120, ksize=4, stride=2, pad=1, initialW=w)
            self.dc2 = L.Deconvolution2D(120, 60, ksize=4, stride=2, pad=1, initialW=w)
            self.dc3 = L.Deconvolution2D(60, 30, ksize=4, stride=2, pad=1, initialW=w)
            self.dc4 = L.Deconvolution2D(30, 3, ksize=4, stride=2, pad=1, initialW=w)
            self.bn0 = L.BatchNormalization(4 * 4 * 240)
            self.bn1 = L.BatchNormalization(120)
            self.bn2 = L.BatchNormalization(60)
            self.bn3 = L.BatchNormalization(30)
            self.bn4 = None     # Don't use batch normalization in output layer!

    def __call__(self, z):
        h = F.relu(self.bn0(self.f0(z)))
        h = F.reshape(h, (-1, 240, 4, 4))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        h = F.tanh(self.dc4(h))
        return h

class Discriminator(chainer.Chain):

    def __init__(self):
        super(Discriminator, self).__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            # Convolution2D(input channels, output channels, kernel size, stride, padding)
            self.c0 = L.Convolution2D(3, 30, ksize=4, stride=2, pad=1, initialW=w)
            self.c1 = L.Convolution2D(30, 60, ksize=4, stride=2, pad=1, initialW=w)
            self.c2 = L.Convolution2D(60, 120, ksize=4, stride=2, pad=1, initialW=w)
            self.c3 = L.Convolution2D(120, 240, ksize=4, stride=2, pad=1, initialW=w)
            self.c4 = L.Convolution2D(240, 1, ksize=3, stride=1, pad=1, initialW=w)
            self.bn0 = None      # Don't use batch normalization in input layer!
            self.bn1 = L.BatchNormalization(60)
            self.bn2 = L.BatchNormalization(120)
            self.bn3 = L.BatchNormalization(240)

    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        h = self.c4(h)
        return h
