"""Train CIFAR-10 with TensorFlow2.0."""
import os
from tqdm import tqdm
import tensorflow as tf
from utils_tf import return_loaders,load_model
import logging
import yaml

class Model():
    def __init__(self, modc,decay_steps,lr):
        self.modc=modc
        self.model = load_model(modc['fn'], modc['name'], modc['args'])
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.weight_decay = 5e-4
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=tf.keras.experimental.CosineDecay(lr,decay_steps),momentum=0.9)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

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
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def train(self, train_loader, test_loader, epoch):
        best_acc = tf.Variable(0.0)
        curr_epoch = tf.Variable(0)
        ckpt_path = './checkpoints/{:s}/'.format(self.modc['name'])
        if not os.path.isdir('./checkpoints/'):
            os.mkdir('./checkpoints/')
        if not os.path.isdir(ckpt_path):
            os.mkdir(ckpt_path)
        logging.basicConfig(format='%(message)s', level=logging.INFO, datefmt='%m-%d %H:%M',
                            filename="%s/%s" % (ckpt_path, 'res.log'), filemode='w+')
        ckpt = tf.train.Checkpoint(curr_epoch=curr_epoch, best_acc=best_acc,
                                   optimizer=self.optimizer, model=self.model)
        manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)

        for e in tqdm(range(int(curr_epoch), epoch)):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
            for images, labels in train_loader:
                self.train_step(images, labels)
            for images, labels in test_loader:
                self.test_step(images, labels)
            template = 'Epoch {:0}, Loss: {:.4f}, Accuracy: {:.2f}%, Test Loss: {:.4f}, Test Accuracy: {:.2f}%'
            outstr=template.format(e + 1,
                                   self.train_loss.result(),
                                   self.train_accuracy.result() * 100,
                                   self.test_loss.result(),
                                   self.test_accuracy.result() * 100)
            print(outstr)
            logging.info(outstr)
            # Save checkpoint
            if self.test_accuracy.result() > best_acc:
                print('Saving...')
                best_acc.assign(self.test_accuracy.result())
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