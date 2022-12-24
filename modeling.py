import tensorflow as tf

def build_model(output_length=6, sequence_length=768):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(sequence_length,)),
        # convolutional layers
        tf.keras.layers.Reshape((sequence_length, 1)),
        tf.keras.layers.Conv1D(256, 12, activation='relu'),
        tf.keras.layers.Conv1D(128, 6, activation='relu'),
        # dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_length, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
