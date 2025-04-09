import kagglehub
import pandas as pd 
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# filepaths 
data_dir = "/Users/muskaan2/ML_Dev/data"
train_filepath = data_dir + '/processed_images/train/'
test_filepath = data_dir + '/processed_images/test/'

# Function to display sample images
def display_sample_images(category, num_samples=5):
    category_path = os.path.join(data_dir+'/processed_images', 'train', category)
    print(category_path)
    image_files = os.listdir(category_path)[:num_samples]
    
    plt.figure(figsize=(10, 3))
    for idx, image_file in enumerate(image_files):
        img_path = os.path.join(category_path, image_file)
        img = Image.open(img_path)
        
        plt.subplot(1, num_samples, idx + 1)
        plt.imshow(img)
        plt.title(f'{category}\n{image_file}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_combined_distribution():
    # Get counts for training data
    train_normal_count = len(os.listdir(train_filepath+'/normal'))
    train_cataract_count = len(os.listdir(train_filepath+'/cataract'))
    
    # Get counts for test data
    test_normal_count = len(os.listdir(test_filepath+'/normal'))
    test_cataract_count = len(os.listdir(test_filepath+'/cataract'))

    fig = go.Figure(data=[
        go.Bar(
            name='Training Set',
            x=['Normal', 'Cataract'],
            y=[train_normal_count, train_cataract_count],
            text=[train_normal_count, train_cataract_count],
            textposition='auto',
        ),
        go.Bar(
            name='Test Set', 
            x=['Normal', 'Cataract'],
            y=[test_normal_count, test_cataract_count],
            text=[test_normal_count, test_cataract_count],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title='Distribution of Images in Training and Test Sets',
        yaxis_title='Number of Images',
        barmode='group'
    )

    fig.show()

def train_test_generators():
    # Define the optimal data augmentation parameters for cataract detection
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,               # Subtle rotations (±10°)
            width_shift_range=0.1,           # Minor horizontal translations (10%)
            height_shift_range=0.1,          # Minor vertical translations (10%)
            brightness_range=[0.85, 1.15],   # Subtle brightness adjustments (±15%)
            zoom_range=0.1,                  # Slight zoom variations (±10%)
            horizontal_flip=True,            # Horizontal flips are anatomically valid
            fill_mode='nearest',             # Fill mode for any empty pixels after transformations
        )

    # Initialize the ImageDataGenerator for testing (without augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)

    #define batch size
    batch_size = 32

    # Create training and testing data generators
    print("Training Categories Labelled as:", os.listdir(train_filepath))
    train_generator = train_datagen.flow_from_directory(
        train_filepath,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )

    print("\nTesting Categories Labelled as:", os.listdir(test_filepath))
    test_generator = test_datagen.flow_from_directory(
        test_filepath,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False  
    )

    # After creating your train_generator
    print("\nClass indices mapping:", train_generator.class_indices)
    return train_generator, test_generator

# Display sample images for both categories
# print("Sample Normal Images:")
# display_sample_images('normal')

# print("\nSample Cataract Images:")
# display_sample_images('cataract')

#Visualise the distribution of images in the training and test sets
# plot_combined_distribution() 

# train_generator, test_generator = train_test_generators()