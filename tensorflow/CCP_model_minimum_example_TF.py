#reference:
# https://www.tensorflow.org/tutorials/quickstart/advanced
# https://gist.github.com/carlosedp/295c9609f8c438b8b5a86d74202a3901
import pdb
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
tf.compat.v1.enable_eager_execution()
import numpy as np
tf.print(tf. __version__)

# gpu_fraction = 0.2
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
# session = tf.compat.v1.Session(config=config)

# from keras.backend import set_session
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# set_session( tf.compat.v1.Session(config=config))

def load_db(path='mnist.npz', batch_size=64, shuffle=True, valid_ratio=0.2):
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path)
    x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
    x_train, x_test = x_train / 255., x_test / 255.

    x_train_sampler, x_valid_sampler, y_train_sampler, y_valid_sampler = train_test_split(x_train, y_train,
                                                                                           stratify=y_train, test_size=valid_ratio)

    x_train_sampler = tf.keras.layers.Normalization(mean=0.5, variance=0.5**2)(x_train_sampler)
    x_valid_sampler = tf.keras.layers.Normalization(mean=0.5, variance=0.5**2)(x_valid_sampler)
    x_test = tf.keras.layers.Normalization(mean=0.5, variance=0.5**2)(x_test)

    train_loader = tf.data.Dataset.from_tensor_slices((x_train_sampler, y_train_sampler)).batch(batch_size)
    valid_loader = tf.data.Dataset.from_tensor_slices((x_valid_sampler, y_valid_sampler)).batch(batch_size)
    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    if shuffle:
        train_loader = train_loader.shuffle(5000)
        valid_loader = valid_loader.shuffle(5000)
        test_loader = test_loader.shuffle(5000)

    image_size, n_classes, channels_in = 28, 10, 1
    return train_loader, valid_loader, test_loader, image_size, n_classes, channels_in

class CCP(Model):
    def __init__(self, hidden_size, image_size=28, channels_in=1, n_degree=4, bias=False, n_classes=10):
        super(CCP, self).__init__()
        self.image_size = image_size
        self.channels_in = channels_in
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_degree = n_degree
        self.total_image_size = self.image_size * self.image_size * channels_in
        initializer1 = tf.keras.initializers.RandomUniform(minval=-1*np.sqrt(1/self.total_image_size),
                                                          maxval=np.sqrt(1/self.total_image_size))
        for i in range(1, self.n_degree + 1):
            setattr(self, 'U{}'.format(i), layers.Dense(self.hidden_size,
                                                        use_bias=bias,
                                                        kernel_initializer=initializer1,
                                                        bias_initializer=initializer1))

        initializer2 = tf.keras.initializers.RandomUniform(minval=-1*np.sqrt(1/self.hidden_size),
                                                          maxval=np.sqrt(1/self.hidden_size))
        self.C = layers.Dense(self.n_classes,use_bias=True,kernel_initializer=initializer2,bias_initializer=initializer2)

    def call(self, z):
        h = layers.Flatten(input_shape=(self.image_size, self.image_size,self.channels_in))(z)
        out = self.U1(h)
        for i in range(2, self.n_degree + 1):
            out = getattr(self, 'U{}'.format(i))(h) * out + out
        out = self.C(out)
        return out

train_loader, valid_loader, test_loader, image_size, n_classes, channels_in = load_db(batch_size=64)
# create the model.
net = CCP(16, image_size=image_size, n_classes=n_classes)

# # define the optimizer
opt = tf.keras.optimizers.SGD(learning_rate=0.001)

# # aggregate losses and accuracy.
train_losses, acc_list = [], []
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def train(train_loader, net, optimizer, criterion, epoch,):
    """ Perform single epoch of the training."""
    for idx, data_dict in enumerate(train_loader):
        img = data_dict[0]
        label = data_dict[1]
        with tf.GradientTape() as tape:
            predictions = net(img, training=True)
            loss = criterion(label, predictions)
        gradients = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))
        train_loss(loss)
        train_accuracy(label, predictions)
        if idx % 100 == 0 and idx > 0:
            m2 = ('Epoch: {}, Epoch iters: {} / {}\t'
                  'Loss: {:.04f}, Acc: {:.06f}')
            print(m2.format(epoch, idx, len(train_loader), float(train_loss.result()), train_accuracy.result()))
    return train_loss.result()


def test(test_loader):
    """ Perform testing, i.e. run net on test_loader data
      and return the accuracy. """
    for test_images, test_labels in test_loader:
        predictions = net(test_images, training=False)
        t_loss = criterion(test_labels, predictions)
        test_loss(t_loss)
        test_accuracy(test_labels, predictions)
    return test_accuracy.result()


acc = 0.
for epoch in range(0, 5):
    print('Epoch {} (previous validation accuracy: {:.03f})'.format(epoch, acc))
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    loss_tr = train(train_loader, net, opt, criterion, epoch)
    acc = test(test_loader)
    train_losses.append(loss_tr)
    acc_list.append(acc)



