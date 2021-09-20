import tensorflow as tf
import os
import embedding_base
import numpy as np


class Linear(tf.keras.Model):
    def __init__(self, input_dims, output_dims):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.W = tf.Variable(
            initial_value=w_init(shape=(output_dims, input_dims), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.B = tf.Variable(initial_value=b_init(shape=(output_dims, 1), dtype="float32"), trainable=True)

    def call(self, inputs):
        return tf.nn.relu(tf.matmul(self.W, inputs) + self.B)


def compute_mse_loss(predictions, targets):
    return tf.reduce_mean(tf.square(predictions - targets))


def clear_directory(output_dir):
    print("Clearing directory: {}".format(output_dir))
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            os.remove(os.path.join(root, file))


class Embedding(embedding_base.Embedding):

    def __init__(self, input_dims, output_dims, checkpoint_dir=None, save_every=None, num_epoch=100, lr=0.01, step_size=10, weight_decay=0.99):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every

        self.forward = Linear(self.input_dims, self.output_dims)
        self.backward = Linear(self.output_dims, self.input_dims)

        # Setup the optimizers
        self.num_epoch = num_epoch
        lr = lr

        learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
            lr,
            decay_steps=step_size,
            decay_rate=weight_decay,
            staircase=True)
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.opt, forward=self.forward, backward=self.backward)

    def update(self, V, H):
        trainable_variables = self.forward.trainable_variables + self.backward.trainable_variables
        with tf.GradientTape() as tape:
            V_ = self.forward(V, training=True)
            proximity_loss = compute_mse_loss(V_, self.forward(H, training=True))
            reconstruction_loss = compute_mse_loss(self.backward(V_, training=True), V)
            loss_values = proximity_loss + reconstruction_loss
            grads = tape.gradient(loss_values, trainable_variables)
        self.opt.apply_gradients(zip(grads, trainable_variables))
        self.checkpoint.step.assign_add(1)
        return loss_values, self.checkpoint.step.numpy()

    def encode(self, c):
        to_numpy = type(c).__module__ == np.__name__
        result = self.forward(c, training=False)
        return result.numpy() if to_numpy else result

    def decode(self, h):
        to_numpy = type(h).__module__ == np.__name__
        result = self.backward(h, training=False)
        return result.numpy() if to_numpy else result

    def load(self):
        if self.checkpoint_dir is not None:
            self.checkpoint.restore(tf.train.latest_checkpoint(os.path.join(self.checkpoint_dir, "weights")))
        else:
            print("Cannot load weights, checkpoint_dir is None.")

    def save(self):
        if not os.path.exists(self.checkpoint_dir):
            print("Creating directory: {}".format(self.checkpoint_dir))
            os.makedirs(self.checkpoint_dir)
        self.checkpoint.save(os.path.join(self.checkpoint_dir, "weights", "ckpt"))

    def incrementally_learn(self, path):
        loss, iterations = self.update(path[:, :-1], path[:, 1:])
        return loss, iterations

    def bootstrap(self, path):
        clear_directory(self.checkpoint_dir)

        # Start training
        sum_loss = 0
        sum_count = 0

        writer = tf.summary.create_file_writer(os.path.join(self.checkpoint_dir, "logs"))

        V = np.reshape(path[:, :-1, :], [self.input_dims, -1])
        H = np.reshape(path[:, 1:, :], [self.input_dims, -1])
        while True:

            # Main training code
            loss, iterations = self.update(V, H)
            sum_loss = sum_loss + loss.numpy()
            sum_count = sum_count + 1

            with writer.as_default():
                tf.summary.scalar('Loss/train', loss, iterations)

            # Save network weights
            if (iterations + 1) % self.save_every == 0:
                print("Iteration: %08d/%08d, loss: %.8f" % (iterations + 1, self.num_epoch, sum_loss / sum_count))
                sum_loss = 0
                sum_count = 0
                self.save()

            if iterations >= self.num_epoch:
                break


if __name__ == '__main__':
    import numpy as np
    dir_path = os.path.dirname(os.path.realpath(__file__))

    model = Embedding(**{
        'input_dims': 8, 'output_dims': 4,
        'checkpoint_dir': os.path.join(dir_path, "..", "..", "artifacts", "tf_one_layer"),
        'save_every': 10, 'num_epoch': 100,
        'lr': 0.01, 'step_size': 10, 'weight_decay': 0.95
    })

    path = np.random.normal(0, 1.0, [8, 16, 2])
    model.bootstrap(path)
