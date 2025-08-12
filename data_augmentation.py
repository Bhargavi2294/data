# data_augmentation.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Example function to demonstrate augmentation
def demonstrate_augmentation():
    # Create an image data generator with augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest',
        brightness_range=(0.8, 1.2)
    )
    
    # Load an example image (you'll need to replace with an actual PCB image path)
    example_img_path = "dataset/single_sided/PCB001.jpg"  # Replace with an actual image
    
    # For demonstration, create a simple colored rectangle if no image exists
    if not os.path.exists(example_img_path):
        img = Image.new('RGB', (224, 224), color=(150, 150, 150))
        # Add some shapes to simulate PCB features
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 174, 174], fill=(100, 100, 100), outline=(200, 200, 200))
        # Add some "components"
        for i in range(10):
            x, y = np.random.randint(60, 164, 2)
            draw.rectangle([x, y, x+10, y+10], fill=(50, 50, 50))
        img.save("example_pcb.jpg")
        example_img_path = "example_pcb.jpg"
    
    img = tf.keras.preprocessing.image.load_img(
        example_img_path, target_size=(224, 224)
    )
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    
    # Display original and augmented images
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 3, 1)
    plt.title("Original Image")
    plt.imshow(x[0].astype(np.uint8))
    
    i = 1
    for batch in datagen.flow(x, batch_size=1):
        i += 1
        plt.subplot(3, 3, i)
        plt.title(f"Augmentation {i-1}")
        plt.imshow(batch[0].astype(np.uint8))
        if i >= 9:
            break
    
    plt.tight_layout()
    plt.savefig("pcb_augmentation_examples.png")
    plt.close()
    print("Augmentation examples saved to 'pcb_augmentation_examples.png'")

if __name__ == "__main__":
    demonstrate_augmentation()
