import tensorflow as tf
import matplotlib.pyplot as plt
import os

# load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()


model.compile(optimizer='adam',  # tf.keras.optimizers.Adam(),
                loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 'categorical_crossentropy',
                metrics=['accuracy'])

# history = model.fit(x_train, y_train, epochs=3)  # ,validation_data=(x_test, y_test)
history = model.fit(x_train, y_train, batch_size=50, epochs=20, validation_data=(x_test, y_test))

# save model
checkpoint_path = "model_teacher/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.save_weights(checkpoint_path.format(epoch=0))

# evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.ylim([0.8, 1])
plt.legend(loc='lower right')
plt.show()

