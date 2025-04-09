#importing libraries for model training
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import VGG16, EfficientNetB0
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import sys 
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.preprocessing import * 

input_shape = (224, 224, 3)
IMG_SIZE = (224, 224)
NUM_CLASSES = 2

# train_generator, test_generator = train_test_generators()

def train_model_VGG16(train_generator, test_generator):
    # Load the base model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze all layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top of the base model for the classification task of cataract images
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer='l2'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model with an initial learning rate
    # model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=[
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision'),
            'accuracy'
        ]
    )

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model_vgg16.h5', monitor='val_loss', save_best_only=True)

    # Train the model with frozen base layers
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=test_generator,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Load the best model from training
    model.load_weights('/Users/muskaan2/ML_Dev/models/best_model_vgg16.h5')

    # Fine-tune the model: Unfreeze the last few layers of the base model
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    # Train model on the data again to finetune last few layers
    model.compile(
        optimizer=Adam(learning_rate=0.0001), 
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=[
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision'),
            'accuracy'
        ]
    )

    # Continue training with fine-tuned layers
    history_fine_tuning = model.fit(
        train_generator,
        epochs=10,
        validation_data=test_generator,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Save the final model (architecture + weights) in .h5 format
    final_model_path = '/Users/muskaan2/ML_Dev/models/final_model_vgg16.h5'
    model.save(final_model_path)

    return history_fine_tuning, model

# def plot_model_eval_history(history, test_generator, model):
#     plt.figure(figsize=(12, 4))
#     #plot training and validation accuracy
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'], label='Training Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.title('Training and Validation Accuracy')
#     plt.legend()

#     #plot training and validation loss
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()
#     plt.show()

#     test_loss, test_accuracy = model.evaluate(test_generator)
#     print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
#     return test_loss, test_accuracy

#plot the evaluation history of the fine-tuned model
# test_loss, test_accuracy  = plot_model_eval_history(history_fine_tuning, test_generator, model)
