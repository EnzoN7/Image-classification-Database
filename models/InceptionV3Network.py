import tensorflow as tf
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import InceptionV3

class InceptionV3Network(tf.keras.Model):

    def __init__(self, _nbclasses, _imagesize):
        super().__init__()
        base = InceptionV3(weights='imagenet',
                           include_top=False,
                           input_shape=(_imagesize, _imagesize, 3))
        base.trainable = False
        self.inceptionv3 = base
        self.globalAveragePooling2D = GlobalAveragePooling2D()
        self.dropout = Dropout(0.5)
        self.dense = Dense(_nbclasses, activation='softmax')

    def call(self, _inputs):
        x = self.inceptionv3(_inputs)
        x = self.globalAveragePooling2D(x)
        x = self.dropout(x)
        return self.dense(x)