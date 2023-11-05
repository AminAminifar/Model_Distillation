import os
# Stop showing warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import numpy as np

import models
import Distiller


################################################################
# Knowledge Distillation
# https://keras.io/examples/vision/knowledge_distillation/
# https://github.com/keras-team/keras-io/blob/master/examples/vision/knowledge_distillation.py
################################################################

# set parameters
# batch_size = 15
model_num_teacher = 3  # check models file for details
model_num_student = 3  # check models file for details

batch_size_teacher = 100
epochs_teacher = 10

temperature = 2
alpha = 0.1
batch_size_student = 100
epochs_student = 10

check_student_scratch_flag = True
batch_size_student_scratch = 100
epochs__student_scratch = 10
################################################################
# Create student and teacher models
################################################################

teacher = models.import_teacher(model_num=model_num_teacher)
student = models.import_student(model_num=model_num_student)
# Clone student for later comparison
student_scratch = keras.models.clone_model(student)

################################################################
# Prepare the dataset
################################################################
# Prepare the train and test dataset.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# Normalize data
x_train = x_train.astype("float32") / 255.0
x_train = np.reshape(x_train, (-1, 28, 28, 1))

x_test = x_test.astype("float32") / 255.0
x_test = np.reshape(x_test, (-1, 28, 28, 1))
################################################################
# Train the teacher
################################################################
print("################################################################")
print("Train and evaluate the teacher")
print("################################################################")
# Train teacher as usual
teacher.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train and evaluate teacher on data.
teacher.fit(x_train, y_train, batch_size=batch_size_teacher, epochs=epochs_teacher)
print("evaluate teacher based on x_test, y_test: ")
teacher.evaluate(x_test, y_test)

################################################################
# Distill teacher to student
################################################################
print("################################################################")
print("Train and evaluate the distilled student model")
print("################################################################")
# Initialize and compile distiller
distiller = Distiller.Distiller(student=student, teacher=teacher)

distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=alpha,
    temperature=temperature,
)

# Distill teacher to student
distiller.fit(x_train, y_train, batch_size=batch_size_student, epochs=epochs_student)  # x_train, y_train, x_train_, y_train_, batch_size=64

# Evaluate student on test dataset
print("evaluate student based on x_test, y_test: ")
distiller.evaluate(x_test, y_test)

################################################################
# Train student from scratch for comparison
################################################################
if check_student_scratch_flag:
    print("################################################################")
    print("Train and evaluate the (non distilled) student model for comparison")
    print("################################################################")
    # Train student as doen usually
    student_scratch.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    # Train and evaluate student trained from scratch.
    student_scratch.fit(x_train, y_train, batch_size=batch_size_student_scratch, epochs=epochs__student_scratch)
    print("evaluate student_scratch based on x_test, y_test: ")
    student_scratch.evaluate(x_test, y_test)
