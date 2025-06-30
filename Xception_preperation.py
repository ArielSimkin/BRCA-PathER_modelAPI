#imports

import os

import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Conv2D, Multiply, Add, Activation, Concatenate, Dropout
import keras.losses, keras.optimizers, keras.metrics, keras.preprocessing.image, keras.applications
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.utils import load_img, img_to_array

from keras.applications import Xception
from keras.applications.xception import preprocess_input

from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc, confusion_matrix

print(tf.config.list_physical_devices('GPU'))

print("finished setup")

#setting up the Image Generation

batch_size = 12
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    vertical_flip=True
)

# Define the data generator for training data
train_generator = train_datagen.flow_from_directory(
    "/mnt/c/Users/owner/System_Progect/Processed_Data_split/train",
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

# Define a separate data generator for validation data (without data augmentation)
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    vertical_flip=True
)

validation_generator = validation_datagen.flow_from_directory(
    "/mnt/c/Users/owner/System_Progect/Processed_Data_split/val",
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

# Define a separate data generator for the test data (including rescaling)
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

test_generator = test_datagen.flow_from_directory(
    "/mnt/c/Users/owner/System_Progect/Processed_Data_split/test",
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

#setting up the model
class Xception_Hybrid_Builder:
    def __init__(self, include_top, weights, input_shape):
        self.base_model = Xception(include_top=include_top, weights=weights, input_shape=input_shape)

    def setup_structure(self):
        # Freeze all layers
        self.base_model.trainable = False

        # Unfreeze the last 10 layers for fine-tuning
        for layer in self.base_model.layers[-10:]:
            layer.trainable = True

        # Get output from a specific layer to apply CBAM
        x = self.base_model.get_layer('block14_sepconv2_act').output

        return x

    def cbam_block(self, input_feature, ratio=8):
        # Channel Attention
        channel = input_feature.shape[-1]
        shared_dense_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
        shared_dense_two = Dense(channel, kernel_initializer='he_normal', use_bias=True)

        avg_pool = GlobalAveragePooling2D()(input_feature)
        avg_pool = Reshape((1, 1, channel))(avg_pool)
        avg_pool = shared_dense_one(avg_pool)
        avg_pool = shared_dense_two(avg_pool)

        max_pool = GlobalMaxPooling2D()(input_feature)
        max_pool = Reshape((1, 1, channel))(max_pool)
        max_pool = shared_dense_one(max_pool)
        max_pool = shared_dense_two(max_pool)

        channel_attention = Add()([avg_pool, max_pool])
        channel_attention = Activation('sigmoid')(channel_attention)
        channel_refined = Multiply()([input_feature, channel_attention])

        # Spatial Attention
        avg_pool_spatial = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
        spatial_attention = Conv2D(filters=1, kernel_size=7, strides=1, padding='same',
                                activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
        refined_feature = Multiply()([channel_refined, spatial_attention])

        return refined_feature
    
    def finalise_arqitechture(self):

        #Setup the first model part
        x = self.setup_structure()
        
        # Apply CBAM
        x = self.cbam_block(x)

        # Add classification head
        x = GlobalAveragePooling2D()(x)
        x = Dense(2048, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)

        # Create the final model
        final_model = Model(inputs=self.base_model.input, outputs=predictions)

        # Compile the final model
        final_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=['binary_accuracy']
        )
        return final_model

builder = Xception_Hybrid_Builder(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
base_model = builder.finalise_arqitechture()

print(train_generator.class_indices)

# Train the model
tf.keras.backend.clear_session()
epochs = 10 # Adjust the number of epochs as needed
history = base_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
)

# Accessing the training accuracy from the history object
train_accuracy = history.history['binary_accuracy']  # Replace 'accuracy' with the appropriate key if using a custom metric
# Plotting the training accuracy over epochs
epochs = range(1, len(train_accuracy) + 1)
plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(range(1, len(train_accuracy) + 1, 5))
plt.savefig("training AUC")
# Evaluate the model on the validation data
loss, accuracy = base_model.evaluate(test_generator, verbose=1)
print(f"test Loss: {loss:.4f}")
print(f"test Accuracy: {accuracy:.4f}")

# Obtain true labels and predicted probabilities
true_labels = test_generator.classes
y_pred_probs = base_model.predict(test_generator)
predicted_labels = (y_pred_probs > 0.5).astype(int).flatten()

# Calculate F1 Score
f1 = f1_score(true_labels, predicted_labels)
print(f"F1 Score: {f1:.4f}")

# Calculate AUC-ROC
auc_roc = roc_auc_score(true_labels, y_pred_probs)
print(f"AUC-ROC: {auc_roc:.4f}")

# Generate ROC curve data
fpr, tpr, thresholds = roc_curve(true_labels, y_pred_probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % auc_roc)
plt.plot([0.0, 1.0], [0.0, 1.0], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.close()

# Generate the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

#saving the model
base_model.save('AXAHv1.keras')














#ADDITIONAL CODE FOR IMPROVMENT
# Path to your image
img_path = '/mnt/c/Users/owner/System_Progect/Processed_Data_split/val/ER-/TCGA-AO-A0J6-01A-01-TSA.d62f4c33-ae94-4321-b8' \
'33-53663942e846_tile_893.png'

# Load the image with the target size used during training
img = load_img(img_path, target_size=(299, 299))  # Adjust target_size as per your model's input

# Convert the image to a NumPy array
img_array = img_to_array(img)

# Expand dimensions to match the model's input shape (1, height, width, channels)
img_array = np.expand_dims(img_array, axis=0)

img_array = preprocess_input(img_array)

predictionnn = base_model.predict(img_array)
print(f"prediction: {predictionnn}")

# Load and preprocess image
def get_img_array(img_path, size):
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return tf.keras.applications.xception.preprocess_input(array)

# Create Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap, pred_index

# Prepare input image
img_array = get_img_array(img_path, size=(299, 299))

# Generate heatmap
heatmap, pred_index = make_gradcam_heatmap(img_array, base_model, 'block14_sepconv2_act')

# Resize heatmap to 299x299
heatmap_resized = cv2.resize(heatmap, (299, 299))

# Load original image for visualization
img = keras.preprocessing.image.load_img(img_path)
img = keras.preprocessing.image.img_to_array(img)

# Create color map
heatmap_color = np.uint8(255 * heatmap_resized)
heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)

# Superimpose heatmap on image
superimposed = cv2.addWeighted(np.uint8(img), 0.6, heatmap_color, 0.4, 0)

# Plot and save results
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(np.uint8(img))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Heatmap")
plt.imshow(heatmap_resized, cmap='jet')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.savefig("gradcam_result.png")
plt.show()
