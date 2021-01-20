import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.preprocessing import image
from keras import callbacks
from PIL import Image
import matplotlib.pyplot as plt
import modules.images.utils as utils
import mlflow
import mlflow.keras
import tempfile
from keras import optimizers
from keras import metrics
from keras.callbacks import TensorBoard


# Model configuration
# initialize parameters
TRAIN_DATA_DIR = '../data/cifar-100/train'
TEST_DATA_DIR = '../data/cifar-100/test'
directory = "../data/cifar-100/train"
output_dir = "../data/out"
image_dir = "../data/images"
model_dir = "../data/models"
output_dir = "../data/out/"
pathdir = "../data/out"
categories = utils.get_categories(directory)
TRAIN_IMAGE_SIZE = 32
TRAIN_BATCH_SIZE = 25
batch_size = 25
img_width, img_height, img_num_channels = 32, 32, 3
validation_split = 0.3
verbosity = 1
input_shape = (img_width, img_height, img_num_channels)


def train_model(args, base_line=True):
    '''
    Train model function
    '''
    graph_label_loss = 'Baseline Model: Training and Validation Loss'
    graph_label_acc = 'Baseline Model: Training and Validation Accuracy'
    graph_image_loss_png = os.path.join(image_dir,'baseline_loss.png')
    graph_image_acc_png = os.path.join(image_dir, 'baseline_accuracy.png')

    if not base_line:
        graph_label_loss = 'Experimental: Training and Validation Loss'
        graph_label_acc = 'Experimental Model: Training and Validation Accuracy'
        graph_image_loss_png = os.path.join(image_dir, 'experimental_loss.png')
        graph_image_acc_png = os.path.join(image_dir,'experimental_accuracy.png')
        
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=validation_split)

    train_generator = image_data_generator.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE),
        batch_size=TRAIN_BATCH_SIZE,
        class_mode='categorical',
        subset='training')
    
    validation_generator = image_data_generator.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE),
        batch_size=TRAIN_BATCH_SIZE,
        class_mode='categorical',
        subset='validation')

    # Create the model
    model = Sequential()

    model.add(Conv2D(args.filters, kernel_size=args.kernel_size, activation='relu', padding='same', input_shape=(img_width, img_height, img_num_channels)))
    model.add(Flatten())
    model.add(Dense(args.output, activation='softmax'))

    # Compile the model
    model.compile(loss=args.loss, optimizer=args.optimizer, metrics=['accuracy'])

    history = model.fit_generator(train_generator, epochs=args.epochs, validation_data=validation_generator)

    model.summary()

    print_metrics(history)
    figure_loss = plot_loss_graph(history, graph_label_loss)
    figure_loss.savefig(graph_image_loss_png)
    figure_acc = plot_accuracy_graph(history, graph_label_acc)
    figure_acc.savefig(graph_image_acc_png)
    # print('==================================================')
    # predictions = model.predict(TEST_DATA_DIR)
    # print(predictions)
    # print('==================================================')
    
    #mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        # print out current run_uuid
        run_uuid = mlflow.active_run().info.run_uuid
        print("MLflow Run ID: %s" % run_uuid)

        # mlflow.create_experiment("Training CNN Model", artifact_location=None)

        # log parameters
        mlflow.log_param("Filters", args.filters)
        mlflow.log_param("Kernel Size", args.kernel_size)
        mlflow.log_param("Output", args.output)
        mlflow.log_param("Epochs", args.epochs)
        mlflow.log_param("Loss", args.loss)
        mlflow.log_param("Optimize", args.optimizer)

        # calculate metrics
        binary_loss = get_binary_loss(history)
        binary_acc = get_binary_acc(history)
        validation_loss = get_validation_loss(history)
        validation_acc = get_validation_acc(history)

        # log metrics
        mlflow.log_metric("binary_loss", binary_loss)
        mlflow.log_metric("binary_acc", binary_acc)
        mlflow.log_metric("validation_loss", validation_loss)
        mlflow.log_metric("validation_acc", validation_acc)

        # log artifacts
        mlflow.log_artifacts(image_dir, "images")

        # log model
        mlflow.keras.log_model(model, "models")

        # save model locally
        pathdir =  "../data/out/keras_models/" + run_uuid
        # keras_save_model(model, pathdir)


        # Write out TensorFlow events as a run artifact
        print("Uploading TensorFlow events as a run artifact.")
        mlflow.log_artifacts(output_dir, artifact_path="events")
        mlflow.end_run()

def keras_save_model(self, model, model_dir='../data/out/keras_models'):
        """
        Convert Keras estimator to TensorFlow
        :type model_dir: object
        """
        print("Model is saved locally to %s" % model_dir)
        mlflow.tensorflow.keras.save_model(model, model_dir, conda_env=None)
        

def get_binary_loss(hist):
    loss = hist.history['loss']
    loss_val = loss[len(loss) - 1]
    return loss_val

def get_binary_acc(hist):
    acc = hist.history['accuracy']
    acc_value = acc[len(acc) - 1]

    return acc_value

def get_validation_loss(hist):
    val_loss = hist.history['val_loss']
    val_loss_value = val_loss[len(val_loss) - 1]

    return val_loss_value

def get_validation_acc(hist):
    val_acc = hist.history['val_accuracy']
    val_acc_value = val_acc[len(val_acc) - 1]

    return val_acc_value

def plot_loss_graph(history, title):
    """
    Generate a matplotlib graph for the loss and accuracy metrics
    :param histroy:
    :return: instance of a graph
    """

    acc = history.history['accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    fig, ax = plt.subplots()
    ax.plot(epochs, loss, 'bo')

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    return fig

def plot_accuracy_graph(history, title):

    plt.clf()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs = range(1, len(acc) + 1)

    fig, ax = plt.subplots()
    ax.plot(epochs, acc, 'bo')

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    return fig

def print_metrics(hist):

    acc_value = get_binary_acc(hist)
    loss_value = get_binary_loss(hist)

    val_acc_value = get_validation_acc(hist)

    val_loss_value = get_validation_loss(hist)

    print("Final metrics: binary_loss:%6.4f" % loss_value)
    print("Final metrics: binary_accuracy=%6.4f" % acc_value)
    print("Final metrics: validation_binary_loss:%6.4f" % val_loss_value)
    print("Final metrics: validation_binary_accuracy:%6.4f" % val_acc_value)


def get_args():
    import argparse
    args = parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument("--filters", help="Number of Filters", action='store', nargs='?', default=16,
                        type=int)
    parser.add_argument("--hidden_layers", help="Number of Hidden Layers", action='store', nargs='?', default=1,
                        type=int)
    parser.add_argument("--output", help="Output from First & Hidden Layers", action='store', nargs='?', default=2,
                        type=int)
    parser.add_argument("--epochs", help="Number of epochs for training", nargs='?', action='store', default=20,
                        type=int)
    parser.add_argument("--kernel_size", help="Number of epochs for training", nargs='?', action='store', default=3,
                        type=int)
    parser.add_argument("--loss", help="Loss Function for the Gradients", nargs='?', action='store',
                        default='categorical_crossentropy', type=str)
    parser.add_argument("--optimizer", help="Optimizer", nargs='?', action='store', default='adam', type=str)
    parser.add_argument("--load_model_path", help="Load model path", nargs='?', action='store', default='/tmp', type=str)
    parser.add_argument("--my_review", help="Type in your review", nargs='?', action='store', default='this film was horrible, bad acting, even worse direction', type=str)
    parser.add_argument("--verbose", help="Verbose output", nargs='?', action='store', default=0, type=int)
    parser.add_argument("--run_uuid", help="Specify the MLflow Run ID", nargs='?', action='store', default=None, type=str)
    parser.add_argument("--tracking_server", help="Specify the MLflow Tracking Server", nargs='?', action='store', default=None, type=str)
    # parser.add_argument("--experiment_name", help="Name of the MLflow Experiment for the runs", nargs='?', action='store', default='Keras_CNN_Classifier', type=str)

    return parser.parse_args()


if __name__ == "__main__":
    
    args = get_args()

    print(args)

    print("hidden_layers:", args.filters)
    print("output:", args.output)
    print("epochs:", args.epochs)
    print("loss:", args.loss)
    print("kernel_size:", args.kernel_size)

    # filters = int(sys.argv[1])
    # output = int(sys.argv[2])
    # epochs = int(sys.argv[3])

    flag = len(sys.argv) == 1

    if flag:
        print("Using Default Baseline parameters")
    else:
        print("Using Experimental parameters")

    # print("filters:", filters)
    # print("output:", output)
    # print("epochs:", epochs)

    train_model(args, flag)
    