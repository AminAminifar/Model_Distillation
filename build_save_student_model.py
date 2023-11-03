import tensorflow as tf
import os

teacher_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

checkpoint_path = "model_teacher/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

latest = tf.train.latest_checkpoint(checkpoint_dir)
teacher_model.load_weights(latest)

student_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


def train_student_with_distillation(student_model, teacher_model, train_data, distillation_loss_fn, optimizer, epochs):
    for epoch in range(epochs):
        for x_batch, y_batch in train_data:
            with tf.GradientTape() as tape:
                student_logits = student_model(x_batch)
                teacher_logits = teacher_model(x_batch)

                loss = distillation_loss_fn(teacher_logits, student_logits, y_batch)

            grads = tape.gradient(loss, student_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, student_model.trainable_variables))

        # Calculate validation accuracy or other metrics as needed
        # Optionally, save the student model checkpoint

# load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# assert x_train.shape == (60000, 28, 28)
# assert x_test.shape == (10000, 28, 28)
# assert y_train.shape == (60000,)
# assert y_test.shape == (10000,)

