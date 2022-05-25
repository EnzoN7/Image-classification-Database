import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten


class BasicConvolutionalNetwork(tf.keras.Model):

    def __init__(self, _nbclasses, _imagesize):
        super().__init__()
        self.conv1 = Conv2D(32, 3, activation="relu", input_shape=[_imagesize, _imagesize, 3])
        self.conv2 = Conv2D(64, 3, activation="relu")
        self.conv3 = Conv2D(94, 3, activation="relu")
        self.conv4 = Conv2D(128, 3, activation="relu")
        self.maxpool = MaxPooling2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(_nbclasses, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)