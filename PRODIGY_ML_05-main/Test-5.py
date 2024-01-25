import pyforest
import random
import zipfile
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.image as mpim
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import keras_tuner as kt

# Access the directory
data_dir = '/kaggle/input/food-101/food-101/food-101/images'
class_names_file_path = '/kaggle/input/food-101/food-101/food-101/meta/classes.txt'

# Read class names from the file into a list
with open(class_names_file_path, 'r') as file:
    all_class_names = [line.strip() for line in file]

# Choose 10 classes randomly
chosen_classes = random.sample(all_class_names, 10)
print(chosen_classes)

# List of real class names
classes = chosen_classes

# Set the number of classes
num_classes = len(classes)

# Display a random image from each class folder
plt.figure(figsize=(15, 8))
for i in range(num_classes):
    class_folder = classes[i]
    class_path = os.path.join(data_dir, class_folder)
    
    # Get a random image from the class folder
    random_image = random.choice(os.listdir(class_path))
    image_path = os.path.join(class_path, random_image)
    
    # Load and display the image
    img = load_img(image_path, target_size=(224, 224))
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.title(f"Class: {class_folder}")
    plt.axis('off')

plt.show()

# Create a sequential model
model = models.Sequential()

# Add convolutional layers with activation and pooling
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output and add dense layers
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Split the data into training and validation
)

batch_size = 32

# Use the same generator for both training and validation
generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',  # For training data
    classes=classes
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',  # For validation data
    classes=classes
)

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(
    generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Evaluate the model on the validation set
evaluation = model.evaluate(validation_generator)
print("Loss:", evaluation[0])
print("Accuracy:", evaluation[1])


# Visualize the evaluation metrics with data labels
plt.figure(figsize=(3, 5))
bars = plt.bar(['Loss', 'Accuracy'], [evaluation[0], evaluation[1]])

# Add data labels to the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.title('Model Evaluation')
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.show()

# Visualize the training history (loss and accuracy over epochs)
history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

epochs = range(1, len(loss_values) + 1)

plt.figure(figsize=(12, 6))

# Plotting Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, loss_values, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss_values, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, acc_values, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_acc_values, 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Instantiate the VGG16 base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Define a model-building function
def build_model(hp):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1, default=0.5)),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Instantiate the tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='tuner_dir',
    project_name='food_classification'
)

# Search for the best hyperparameters
tuner.search(
    generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[early_stopping],
)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Best Hyperparameters:\n{best_hps.values}")

# Build the model with the best hyperparameters
best_model = tuner.hypermodel.build(best_hps)

# Train the best model with the best hyperparameters
history_best_model = best_model.fit(
    generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Save the best model and its weights
best_model.save('best_model.h5')
# Evaluate the best model on the validation set
evaluation = best_model.evaluate(validation_generator)
print("Best Model Validation Loss:", evaluation[0])
print("Best Model Validation Accuracy:", evaluation[1])

# Predictions on the validation set using the best model
y_true = validation_generator.classes
y_pred_probs = best_model.predict(validation_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculate accuracy on the validation set
validation_accuracy = accuracy_score(y_true, y_pred)
print("Validation Accuracy with Best Model:", validation_accuracy)

# Visualize the training history for the best model
history_dict_best_model = history_best_model.history

loss_values_best_model = history_dict_best_model['loss']
val_loss_values_best_model = history_dict_best_model['val_loss']
acc_values_best_model = history_dict_best_model['accuracy']
val_acc_values_best_model = history_dict_best_model['val_accuracy']

epochs_best_model = range(1, len(loss_values_best_model) + 1)

plt.figure(figsize=(12, 6))

# Plotting Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_best_model, loss_values_best_model, 'bo-', label='Training Loss')
plt.plot(epochs_best_model, val_loss_values_best_model, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss (Best Model)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_best_model, acc_values_best_model, 'bo-', label='Training Accuracy')
plt.plot(epochs_best_model, val_acc_values_best_model, 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy (Best Model)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

loaded_model = load_model('best_model.h5')

def predict_and_display(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make predictions
    predictions = loaded_model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # Display the image with the predicted label
    plt.imshow(img)
    plt.title(f"Predicted Class: {classes[predicted_class]}")
    plt.axis('off')
    plt.show()
# Example usage
image_path = '/kaggle/input/test-image/rice.jpg'
predict_and_display(image_path)

image_path = '/kaggle/input/test-image/shr_gri.jpg'
predict_and_display(image_path)

image_path = '/kaggle/input/test-image/cho-mou.jpg'
predict_and_display(image_path)
