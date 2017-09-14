import chainer
import chainer.functions as F
import chainer.links as L
import chainer.cuda
import cupy as xp
import datetime
import gzip
import numpy as np
import os
import urllib.request

from model import Generator
from model import Discriminator
from visualize import visualize

# Define constants
N = 1000    # Minibatch size
M = 70000
SNAPSHOT_INTERVAL = 10
REAL_LABEL = 1
FAKE_LABEL = 0


def main():

    # (Make directories)
    os.mkdir('dataset/') if not os.path.isdir('dataset') else None
    os.mkdir('train/') if not os.path.isdir('train') else None

    # (Download dataset)
    if not os.path.exists('dataset/mnist.npy'):
        url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        response = urllib.request.urlopen(url)
        with open('dataset/train-images-idx3-ubyte.gz', 'wb') as stream:
            stream.write(response.read())
        url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
        response = urllib.request.urlopen(url)
        with open('dataset/t10k-images-idx3-ubyte.gz', 'wb') as stream:
            stream.write(response.read())
        train = np.zeros((M, 1, 32, 32), dtype='f')
        with gzip.open('dataset/train-images-idx3-ubyte.gz') as stream:
            _ = np.frombuffer(stream.read(), dtype=np.uint8, offset=16).astype('f').reshape((-1, 1, 28, 28))
        with gzip.open('dataset/t10k-images-idx3-ubyte.gz') as stream:
            __ = np.frombuffer(stream.read(), dtype=np.uint8, offset=16).astype('f').reshape((-1, 1, 28, 28))
            _ = np.vstack((_, __))
        for i in range(M):
            train[i, 0, 2:30, 2:30] = _[i]
        train /= 255.
        np.save('dataset/mnist', train)
    os.remove('dataset/train-images-idx3-ubyte.gz') if os.path.exists('dataset/train-images-idx3-ubyte.gz') else None
    os.remove('dataset/t10k-images-idx3-ubyte.gz') if os.path.exists('dataset/t10k-images-idx3-ubyte.gz') else None

    # Create samples.
    train = np.load('dataset/mnist.npy').reshape((-1, 1, 32, 32))
    train = np.random.permutation(train)
    validation_z = xp.random.uniform(low=-1.0, high=1.0, size=(100, 100)).astype('f')

    # Create the model
    gen = Generator()
    dis = Discriminator()

    # (Use GPU)
    chainer.cuda.get_device(0).use()
    gen.to_gpu()
    dis.to_gpu()

    # Setup optimizers
    optimizer_gen = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5, beta2=0.9)
    optimizer_dis = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5, beta2=0.9)
    optimizer_gen.setup(gen)
    optimizer_dis.setup(dis)

    # (Change directory)
    os.chdir('train/')
    time = datetime.datetime.today().strftime("%Y-%m-%d %H.%M.%S")
    os.mkdir(time)
    os.chdir(time)

    # (Validate input images)
    visualize(train, 'real.png')

    # Training
    for epoch in range(1000):

        # (Validate generated images)
        if (epoch % SNAPSHOT_INTERVAL == 0):
            os.mkdir('%d' % epoch)
            os.chdir('%d' % epoch)
            fake = chainer.cuda.to_cpu(gen(validation_z).data)
            visualize(fake, 'fake.png')
            chainer.serializers.save_hdf5("gen.h5", gen)
            chainer.serializers.save_hdf5("dis.h5", dis)
            os.chdir('..')

        total_loss_dis = 0.0
        total_loss_gen = 0.0

        for n in range(0, M, N):

            ############################
            # (1) Update D network
            ###########################
            real = xp.array(train[n:n + N])
            z = xp.random.uniform(low=-1.0, high=1.0, size=(100, 100)).astype('f')
            fake = gen(z)
            y_real = dis(real)
            y_fake = dis(fake)
            loss_dis = (F.sum((y_real - REAL_LABEL) ** 2) + F.sum((y_fake - FAKE_LABEL) ** 2)) / np.prod(y_fake.shape)
            dis.cleargrads()
            loss_dis.backward()
            optimizer_dis.update()

            ###########################
            # (2) Update G network
            ###########################
            z = xp.random.uniform(low=-1.0, high=1.0, size=(100, 100)).astype('f')
            fake = gen(z)
            y_fake = dis(fake)
            loss_gen = F.sum((y_fake - REAL_LABEL) ** 2) / np.prod(y_fake.shape)
            gen.cleargrads()
            loss_gen.backward()
            optimizer_gen.update()

            total_loss_dis += loss_dis.data
            total_loss_gen += loss_gen.data

        # (View loss)
        total_loss_dis /= M / N
        total_loss_gen /= M / N
        print(epoch, total_loss_dis, total_loss_gen)


if __name__ == '__main__':
    main()
