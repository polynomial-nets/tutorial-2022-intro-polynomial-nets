"""Train CIFAR-10 with TensorFlow2.0."""
from tqdm import tqdm
import tensorflow as tf
from utils_tf import return_loaders,load_model
import logging
import yaml
import sys
from os.path import abspath, dirname, join, isdir
from os import curdir, makedirs
base = dirname(abspath(__file__))
sys.path.append(base)

class Model():
    def __init__(self, modc, decay_steps, lr):
        self.modc=modc
        self.model = load_model(modc['fn'], modc['name'], modc['args'])
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.weight_decay = 5e-4
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=tf.keras.experimental.CosineDecay(lr,decay_steps),momentum=0.9)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_acc = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            # Cross-entropy loss
            ce_loss = self.loss_object(labels, predictions)
            # L2 loss(weight decay)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables])
            loss = ce_loss + l2_loss * self.weight_decay
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_acc(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)
        self.test_loss(t_loss)
        self.test_acc(labels, predictions)

    def train(self, train_loader, test_loader, epoch):
        best_acc = tf.Variable(0.0)
        curr_epoch = tf.Variable(0)
        cur_path = abspath(curdir)
        # # define the output path
        out = join(cur_path, 'results_poly_tf', '')
        if not isdir(out):
            makedirs(out)
        logging.basicConfig(format='%(message)s', level=logging.INFO, datefmt='%m-%d %H:%M',
                            filename="%s/%s" % (out, 'res.log'), filemode='w+')
        print('Current path: {}'.format(cur_path))
        ckpt = tf.train.Checkpoint(curr_epoch=curr_epoch, best_acc=best_acc,
                                   optimizer=self.optimizer, model=self.model)
        manager = tf.train.CheckpointManager(ckpt, out, max_to_keep=1)

        for e in tqdm(range(int(curr_epoch), epoch)):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_acc.reset_states()
            self.test_loss.reset_states()
            self.test_acc.reset_states()
            for images, labels in train_loader:
                self.train_step(images, labels)
            for images, labels in test_loader:
                self.test_step(images, labels)
            msg = 'Epoch:{}.\tTrain_Loss: {:.3f}.\tTrain_Acc: {:.03f}.\tTest_Acc: {:.03f}.\tBest_Test_Acc:{:.03f} (epoch: {}).'
            msg = msg.format(int(e + 1),self.train_loss.result(), self.train_acc.result(),self.test_acc.result(),
                            best_acc.numpy(), curr_epoch.numpy())
            print(msg)
            logging.info(msg)
            # Save checkpoint
            if self.test_acc.result() > best_acc:
                print('Saving...')
                best_acc.assign(self.test_acc.result())
                curr_epoch.assign(e + 1)
                manager.save()

def main():
    yml = yaml.safe_load(open('model_ncp_tf.yml'))  # # file that includes the configuration.
    tinfo = yml['training_info']
    train_loader, test_loader = return_loaders(**yml['dataset'])
    decay_steps = int(tinfo['total_epochs'] * train_loader.cardinality().numpy())
    modc = yml['model']
    model = Model(modc, decay_steps,yml['learning_rate'])
    model.train(train_loader, test_loader,tinfo['total_epochs'])

if __name__ == "__main__":
    main()