import math

import tensorflow as tf

learning_rate = 1e-3

from tensorflow.keras.datasets import mnist


class NaiveDense:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation

        # cria uma matriz W com shape (input_size, output_size)
        # ele tem esse shape porque a camada espera que para cada
        # quantidade de inputs (input_size) ele vai produzir n outputs
        # com valores aleatorios
        w_shape = (input_size, output_size)
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
        self.W = tf.Variable(w_initial_value)

        # cria um vetor b com shape (output_size) com valores
        # iniciais zero
        b_shape = output_size
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)

    def __call__(self, inputs):
        # aplica o forward pass
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    # método conveniente para chamar os pesos da camada
    @property
    def weights(self):
        return [self.W, self.b]


class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights


learning_rate = 1e-3


def update_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        # atualiza os pesos com o gradiente
        # o learning rate é um hiperparâmetro que controla a magnitude da atualização
        w.assign_sub(learning_rate * g)  # assign_sub = '-=' do tensorflow


def one_training_step(model, images_batch, labels_batch):
    # roda o forward pass (computa as predições do modelo
    # dentro do scope do GradientTape)
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels_batch, predictions
        )
        average_loss = tf.reduce_mean(per_sample_losses)
    # computa o gradiente da perda com relação aos pesos. os gradientes de saida
    # sao uma lista onde cada entry correpsonde ao peso do model.weights list
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss


class BatchGenerator:
    def __init__(self, images, labels, batch_size):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels


def update_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        # atualiza os pesos com o gradiente
        # o learning rate é um hiperparâmetro que controla a magnitude da atualização
        w.assign_sub(learning_rate * g)  # assign_sub = '-=' do tensorflow


def one_training_step(model, images_batch, labels_batch):
    # roda o forward pass (computa as predições do modelo
    # dentro do scope do GradientTape)
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels_batch, predictions
        )
        average_loss = tf.reduce_mean(per_sample_losses)
    # computa o gradiente da perda com relação aos pesos. os gradientes de saida
    # sao uma lista onde cada entry correpsonde ao peso do model.weights list
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss


def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels, batch_size)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"Batch {batch_counter}, loss: {loss.numpy()}")


model = NaiveSequential(
    [
        NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
        NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax),
    ]
)
# modelo deve ter 4 pesos: 2 matrizes de w e 2 vetores de b
assert len(model.weights) == 4

# carrega o dataset MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# reshape para 2D, cada imagem é um vetor de 28*28 = 784 pixels
train_images = train_images.reshape((60000, 28 * 28))
test_images = test_images.reshape((10000, 28 * 28))
# normaliza os valores dos pixels para o intervalo [0, 1]
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255

fit(model, train_images, train_labels, epochs=5, batch_size=128)
