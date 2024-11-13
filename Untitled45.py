#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.layers import Reshape, BatchNormalization, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


# In[2]:


def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,    # Reduced from 20
        width_shift_range=0.1,  # Reduced from 0.2
        height_shift_range=0.1,  # Reduced from 0.2
        zoom_range=0.1,      # Reduced from 0.2
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    return train_datagen, test_datagen

def load_datasets(train_dir, test_dir, img_size=(28, 28), batch_size=4):
    train_datagen, test_datagen = create_data_generators()
    
    training_set = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=True
    )
    
    test_set = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False
    )
    
    return training_set, test_set


# In[3]:


def build_generator(latent_dim):
    noise_input = Input(shape=(latent_dim,))
    
    # Reduced layer sizes
    x = Dense(64)(noise_input)  # Reduced from 128
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    
    x = Dense(128)(x)  # Reduced from 256
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    
    x = Dense(256)(x)  # Reduced from 512
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    
    x = Dense(28 * 28, activation='tanh')(x)  # Adjusted for new image size
    output = Reshape((28, 28, 1))(x)
    
    return Model(noise_input, output, name='generator')


# In[4]:


def build_discriminator(img_shape):
    img_input = Input(shape=img_shape)
    
    # Simplified architecture
    x = Conv2D(32, kernel_size=3, strides=2, padding='same')(img_input)  # Reduced filters
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(x)
    
    return Model(img_input, output, name='discriminator')


# In[5]:


class GAN(Model):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        
    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
    
    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, 50])  # Adjusted latent_dim
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            
            gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss
            
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.g_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return {"d_loss": disc_loss, "g_loss": gen_loss}


# In[6]:


def train_gan(gan_model, dataset, epochs):
    for epoch in range(epochs):
        for batch in dataset:
            losses = gan_model.train_step(batch[0])
            
        if (epoch + 1) % 10 == 0:  # More frequent updates
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"D loss: {losses['d_loss']:.4f}")
            print(f"G loss: {losses['g_loss']:.4f}")
            generate_and_save_images(gan_model.generator, epoch + 1)

def generate_and_save_images(generator, epoch, num_examples=4):  # Reduced examples
    noise = tf.random.normal([num_examples, 50])  # Adjusted latent_dim
    generated_images = generator(noise, training=False)
    generated_images = (generated_images + 1) / 2.0
    
    plt.figure(figsize=(4, 1))  # Reduced figure size
    for i in range(num_examples):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(generated_images[i, :, :, 0].numpy(), cmap='gray')
        plt.axis('off')
    
    plt.savefig(f'gan_output_epoch_{epoch}.png')
    plt.close()


# In[ ]:


# Configuration
IMG_SIZE = (28, 28)
BATCH_SIZE = 4
LATENT_DIM = 50
GAN_EPOCHS = 30
CNN_EPOCHS = 10

# Set your data directories
train_dir = r"C:\Proj\ML\Media player\data\train"  # Update this
test_dir = r"C:\Proj\ML\Media player\data\test"    # Update this

# Load datasets
training_set, test_set = load_datasets(train_dir, test_dir, IMG_SIZE, BATCH_SIZE)

# Build and compile GAN
generator = build_generator(LATENT_DIM)
discriminator = build_discriminator((IMG_SIZE[0], IMG_SIZE[1], 1))

gan_model = GAN(generator, discriminator)
gan_model.compile(
    g_optimizer=Adam(learning_rate=0.0001),  # Reduced learning rate
    d_optimizer=Adam(learning_rate=0.0001),
    loss_fn=tf.keras.losses.BinaryCrossentropy()
)

# Train GAN
train_gan(gan_model, training_set, GAN_EPOCHS)
generator.save('gesture_generator_model.h5')


# In[ ]:


# Build and train gesture recognition model
num_classes = len(training_set.class_indices)
gesture_model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(16, (3, 3), activation='relu'),  # Reduced filters
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    
    Flatten(),
    Dense(128, activation='relu'),  # Reduced units
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

gesture_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train with early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,  # Reduced patience
    restore_best_weights=True
)

# Train the model
history = gesture_model.fit(
    training_set,
    validation_data=test_set,
    epochs=CNN_EPOCHS,
    callbacks=[early_stopping]
)

# Evaluate and save
loss, accuracy = gesture_model.evaluate(test_set)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
gesture_model.save('gesture_recognition_model.h5')

