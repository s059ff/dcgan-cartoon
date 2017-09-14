import matplotlib.pylab as plt


def visualize(X, fname):
    plt.figure(num=None, figsize=(20, 20), dpi=100, facecolor='w', edgecolor='k')
    for i in range(0, 100):
        plt.subplot(10, 10, i + 1)
        plt.tick_params(labelleft='off', top='off', bottom='off')
        plt.tick_params(labelbottom='off', left='off', right='off')
        plt.imshow(X[i].transpose(1, 2, 0).reshape((32, 32)), cmap='gray', vmin=0.0, vmax=1.0)
    plt.savefig(fname)
    plt.close()
    