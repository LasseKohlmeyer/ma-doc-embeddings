from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np


class SimpleAutoEncoder:
    def __init__(self, latent_dim: int, input_data, epochs: int = None):
        try:
            self.input_data = input_data
            self.input_shape = [self.input_data.shape[1], self.input_data.shape[2]]
        except IndexError:
            self.input_data = np.expand_dims(input_data, axis=1)
            self.input_shape = [self.input_data.shape[1], self.input_data.shape[2]]
            print(self.input_shape)

        self.input_data = np.asarray(self.input_data).astype('float32')

        self.encoder = Sequential()
        self.encoder.add(Flatten(input_shape=self.input_shape))
        # self.encoder.add(Dense(1000, activation="relu"))
        # self.encoder.add(Dense(800, activation="relu"))
        # self.encoder.add(Dense(600, activation="relu"))
        # self.encoder.add(Dense(400, activation="relu"))
        self.encoder.add(Dense(latent_dim, activation="relu"))

        self.decoder = Sequential()
        self.decoder.add(Dense(self.input_shape[0] * self.input_shape[1], input_shape=[latent_dim], activation='relu'))
        # self.decoder.add(Dense(400, input_shape=[latent_dim], activation='relu'))
        # self.decoder.add(Dense(600, activation='relu'))
        # self.decoder.add(Dense(800, activation='relu'))
        # self.decoder.add(Dense(1000, activation='relu'))
        # self.decoder.add(Dense(self.input_shape[0] * self.input_shape[1], activation="relu"))
        self.decoder.add(Reshape(self.input_shape))

        self.autoencoder = Sequential([self.encoder, self.decoder])
        self.autoencoder.compile(loss="mse")

        self.history = None
        if epochs:
            self.fit(epochs)

    def fit(self, epochs: int):
        print(self.input_data.shape)
        history = self.autoencoder.fit(self.input_data, self.input_data, epochs=epochs)
        self.history = history
        return history

    def get_latent_representation(self, predict_data=None):
        if predict_data is None:
            predict_data = self.input_data
        else:
            predict_data = np.asarray(predict_data).astype('float32')
        try:
            return self.encoder.predict(predict_data)
        except ValueError:
            pred_data = np.expand_dims(predict_data, axis=1)
            return self.encoder.predict(pred_data)

    def plot_history(self):
        plt.plot(self.history.history['loss'])
        # plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()


if __name__ == '__main__':
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_train = X_train / 255.0
    # X_test = X_test / 255.0
    # print(X_test.shape)

    X_train = np.array([[.1,.2,.3,.4,.5], [.2,.5,.6,.7,.8], [.2,.4,.6,.8,.10]])
    print(X_train.shape)
    auto_encoder = SimpleAutoEncoder(latent_dim=300, input_data=X_train, epochs=2)
    # auto_encoder.fit(2)
    print(auto_encoder.get_latent_representation(X_train).shape)
    # print(auto_encoder.get_latent_representation(X_test).shape)
    auto_encoder.plot_history()

    auto_encoder = SimpleAutoEncoder(latent_dim=5, input_data=X_train, epochs=2)
    # auto_encoder.fit(2)
    print(auto_encoder.get_latent_representation(X_train).shape)
    # print(auto_encoder.get_latent_representation(X_test).shape)
    auto_encoder.plot_history()
