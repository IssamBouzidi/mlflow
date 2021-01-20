# import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Model
from keras.preprocessing import image


# def resize_image(img, size=(20,20)):

#     h, w = img.shape[:2]
    
#     if h == w: 
#         return cv2.resize(img, size, cv2.INTER_AREA)

#     dif = h if h > w else w


#     if dif > (size[0] + size[1]):
#         interpolation = cv2.INTER_AREA
#     else:
#         interpolation = cv2.INTER_CUBIC

#     x_pos = (dif - w)//2
#     y_pos = (dif - h)//2

#     mask = np.zeros((dif, dif), dtype=img.dtype)
#     mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]

#     return cv2.resize(mask, size, interpolation)



def get_categories(dir_path):
    categories = []
    for root, subdirectories, files in os.walk(dir_path):
        for subdirectory in subdirectories:
            categories.append(subdirectory)
    
    return categories

def loss_visualisation(training_loss, training_val_loss):
    plt.plot(training_loss, color='red', label='Training loss')
    plt.plot(training_val_loss,  color='green', label='Validation loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.legend()

    plt.show()


def accuracy_visualisation(training_accuracy, training_val_accuracy):
    plt.plot(training_accuracy, color='red', label='Training accuracy')
    plt.plot(training_val_accuracy, color='green', label='Validation accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.legend()

    plt.show()



# def get_binary_loss(hist):
#     loss = hist.history['loss']
#     loss_val = loss[len(loss) - 1]
#     return loss_val

# def get_binary_acc(hist):
#     acc = hist.history['binary_accuracy']
#     acc_value = acc[len(acc) - 1]

#     return acc_value

# def get_validation_loss(hist):
#     val_loss = hist.history['val_loss']
#     val_loss_value = val_loss[len(val_loss) - 1]

#     return val_loss_value

# def get_validation_acc(hist):
#     val_acc = hist.history['val_binary_accuracy']
#     val_acc_value = val_acc[len(val_acc) - 1]

#     return val_acc_value

# # def evaluate_model(self,model, x_test, y_test):
# #     return model.evaluate(x_test, y_test)


# def run_mlflow(hidden_layers, output, epochs, loss):
#     #mlflow.set_experiment(args.experiment_name)
#     with mlflow.start_run():
#         # print out current run_uuid
#         run_uuid = mlflow.active_run().info.run_uuid
#         print("MLflow Run ID: %s" % run_uuid)

#         # log parameters
#         mlflow.log_param("hidden_layers", hidden_layers)
#         mlflow.log_param("output", output)
#         mlflow.log_param("epochs", epochs)
#         mlflow.log_param("loss_function", loss)

#         # calculate metrics
#         binary_loss = get_binary_loss(history)
#         binary_acc = get_binary_acc(history)
#         validation_loss = get_validation_loss(history)
#         validation_acc = get_validation_acc(history)
#         average_loss = results[0]
#         average_acc = results[1]

#         # log metrics
#         mlflow.log_metric("binary_loss", binary_loss)
#         mlflow.log_metric("binary_acc", binary_acc)
#         mlflow.log_metric("validation_loss", validation_loss)
#         mlflow.log_metric("validation_acc", validation_acc)
#         mlflow.log_metric("average_loss", average_loss)
#         mlflow.log_metric("average_acc", average_acc)

#         # log artifacts
#         # mlflow.log_artifacts(image_dir, "images")

#         # log model
#         mlflow.keras.log_model(model, "models")

#         # save model locally
#         # pathdir = "keras_models/" + run_uuid
#         # model_dir = get_directory_path(pathdir, False)
#         # keras_save_model(model, model_dir)

#         # Write out TensorFlow events as a run artifact
#         print("Uploading TensorFlow events as a run artifact.")
#         mlflow.log_artifacts(output_dir, artifact_path="events")
#         mlflow.end_run()
