# train_models.py

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from quality_check_model import PCBQualityCheckModel
from certification_model import PCBCertificationModel

def create_directory_structure():
    """Create the necessary directory structure for the project."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)
    
    # Create PCB type directories
    pcb_types = [
        "single_sided", "double_sided", "multilayer",
        "flexible", "rigid_flex", "high_frequency", "high_power"
    ]
    
    for pcb_type in pcb_types:
        os.makedirs(f"dataset/{pcb_type}", exist_ok=True)
        
    print("Directory structure created successfully!")

def create_dummy_images():
    """Create dummy PCB images for demonstration."""
    # Check if dataset already has images
    if len(os.listdir("dataset/single_sided")) > 0:
        print("Dummy images already exist. Skipping creation.")
        return
        
    print("Creating dummy PCB images for demonstration...")
    
    # Sample PCB dataset from CSV
    try:
        df = pd.read_csv("data/pcb_dataset.csv")
    except FileNotFoundError:
        print("Error: pcb_dataset.csv not found. Run dataset_creator.py first.")
        return
        
    # Create a dummy image for each entry in the dataset
    for _, row in df.iterrows():
        image_path = row["image_path"]
        pcb_type = row["pcb_type"]
        components_density = row["components_density"]
        defect_type = row["defect_type"]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # Create a colored rectangle as a dummy PCB
        img = Image.new('RGB', (224, 224), color=(150, 150, 150))
        
        # Add some simulated PCB features based on type and density
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Draw different patterns based on PCB type
        if pcb_type == "single_sided":
            draw.rectangle([20, 20, 204, 204], fill=(130, 130, 130), outline=(200, 200, 200))
        elif pcb_type == "double_sided":
            draw.rectangle([20, 20, 204, 204], fill=(120, 120, 120), outline=(200, 200, 200))
            draw.rectangle([40, 40, 184, 184], fill=(140, 140, 140), outline=(180, 180, 180))
        elif pcb_type == "multilayer":
            draw.rectangle([20, 20, 204, 204], fill=(110, 110, 110), outline=(200, 200, 200))
            for i in range(3):
                inset = 20 + (i * 20)
                draw.rectangle(
                    [inset, inset, 224 - inset, 224 - inset],
                    outline=(180 - (i * 20), 180 - (i * 20), 180 - (i * 20))
                )
        elif pcb_type == "flexible":
            draw.rectangle([20, 20, 204, 204], fill=(150, 150, 100), outline=(200, 200, 150))
            # Add curved lines to represent flexibility
            for i in range(5):
                y = 50 + (i * 30)
                for x in range(30, 194, 2):
                    draw.point((x, y + int(10 * np.sin(x * 0.1))), fill=(180, 180, 130))
        elif pcb_type == "rigid_flex":
            # Rigid section
            draw.rectangle([20, 20, 204, 100], fill=(120, 120, 120), outline=(200, 200, 200))
            # Flexible section
            draw.rectangle([20, 120, 204, 204], fill=(150, 150, 100), outline=(200, 200, 150))
        elif pcb_type == "high_frequency":
            draw.rectangle([20, 20, 204, 204], fill=(100, 120, 140), outline=(180, 200, 220))
            # Add microstrip lines
            for i in range(5):
                y = 50 + (i * 30)
                draw.line([(30, y), (194, y)], fill=(70, 90, 110), width=3)
        elif pcb_type == "high_power":
            draw.rectangle([20, 20, 204, 204], fill=(130, 110, 110), outline=(210, 190, 190))
            # Add thick power traces
            for i in range(3):
                y = 60 + (i * 50)
                draw.line([(30, y), (194, y)], fill=(90, 70, 70), width=8)
        
        # Add "components" based on density
        num_components = {
            "low": 5,
            "medium": 15,
            "high": 30,
            "very_high": 50
        }.get(components_density, 10)
        
        # Add random "components"
        np.random.seed(int(row["image_id"].replace("PCB", "")))  # For reproducibility
        for _ in range(num_components):
            x, y = np.random.randint(30, 194, 2)
            size = np.random.randint(5, 15)
            color = tuple(np.random.randint(50, 150, 3))
            draw.rectangle([x, y, x + size, y + size], fill=color)
        
        # Add a simulated defect if specified
        if defect_type != "none":
            if defect_type == "solder_bridge":
                x, y = np.random.randint(50, 174, 2)
                draw.line([(x, y), (x + 20, y)], fill=(220, 220, 220), width=3)
            elif defect_type == "open_circuit":
                x, y = np.random.randint(50, 174, 2)
                draw.line([(x, y), (x + 20, y)], fill=(80, 80, 80), width=2)
                # Break in the middle
                draw.line([(x + 8, y - 1), (x + 12, y + 1)], fill=(150, 150, 150), width=4)
            elif defect_type == "solder_quality":
                for _ in range(3):
                    x, y = np.random.randint(50, 174, 2)
                    draw.ellipse([x, y, x + 8, y + 8], fill=(200, 200, 200))
            elif defect_type == "copper_exposure":
                x, y = np.random.randint(50, 174, 2)
                draw.rectangle([x, y, x + 15, y + 15], fill=(180, 120, 80))
            elif "misalignment" in defect_type:
                for _ in range(3):
                    x, y = np.random.randint(50, 174, 2)
                    draw.rectangle([x, y, x + 10, y + 10], fill=(100, 100, 100))
                    # Misaligned pad
                    draw.rectangle([x + 3, y - 3, x + 13, y + 7], fill=(170, 170, 170))
            elif "delamination" in defect_type:
                x, y = np.random.randint(50, 174, 2)
                # Create a bubble effect
                draw.ellipse([x, y, x + 30, y + 20], fill=(180, 180, 180), outline=(200, 200, 200))
                
        # Save the image
        img.save(image_path)
        
    print(f"Created {len(df)} dummy PCB images!")

def prepare_training_datasets(img_size=(224, 224), batch_size=32):
    """
    Prepare training datasets for both models.
    
    Returns:
        tuple: (quality_train_ds, quality_val_ds, cert_train_ds, cert_val_ds)
    """
    print("Preparing training datasets...")
    
    # Load the dataset
    try:
        df = pd.read_csv("data/pcb_dataset.csv")
    except FileNotFoundError:
        print("Error: pcb_dataset.csv not found. Run dataset_creator.py first.")
        return None, None, None, None
        
    # Check if images exist
    if not os.path.exists(df.iloc[0]["image_path"]):
        print("Error: Images not found. Run create_dummy_images() first.")
        return None, None, None, None
        
    # Split into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    
    # Prepare quality check labels
    quality_encoder = LabelEncoder()
    quality_encoder.fit(df["quality_check_required"])
    
    train_quality_labels = quality_encoder.transform(train_df["quality_check_required"])
    val_quality_labels = quality_encoder.transform(val_df["quality_check_required"])
    
    # Save quality check class names
    quality_classes = quality_encoder.classes_.tolist()
    os.makedirs("models", exist_ok=True)
    with open("models/quality_check_classes.json", "w") as f:
        import json
        json.dump(quality_classes, f)
    
    # Prepare certification labels (multi-label)
    mlb = MultiLabelBinarizer()
    all_certs = df["certification_needed"].apply(lambda x: x.split(";")).tolist()
    mlb.fit(all_certs)
    
    train_cert_labels = mlb.transform(train_df["certification_needed"].apply(lambda x: x.split(";")))
    val_cert_labels = mlb.transform(val_df["certification_needed"].apply(lambda x: x.split(";")))
    
    # Save certification class names
    cert_classes = mlb.classes_.tolist()
    with open("models/certification_classes.json", "w") as f:
        import json
        json.dump(cert_classes, f)
    
    # Create data generators
    def preprocess_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img
    
    # Create quality check datasets
    def create_quality_dataset(df, labels):
        paths = df["image_path"].values
        
        def generator():
            for i in range(len(df)):
                yield preprocess_image(paths[i]), labels[i]
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(img_size[0], img_size[1], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int64)
            )
        )
        
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create certification datasets
    def create_cert_dataset(df, labels):
        paths = df["image_path"].values
        
        def generator():
            for i in range(len(df)):
                yield preprocess_image(paths[i]), labels[i]
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(img_size[0], img_size[1], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(len(cert_classes)), dtype=tf.int64)
            )
        )
        
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create the datasets
    quality_train_ds = create_quality_dataset(train_df, train_quality_labels)
    quality_val_ds = create_quality_dataset(val_df, val_quality_labels)
    
    cert_train_ds = create_cert_dataset(train_df, train_cert_labels)
    cert_val_ds = create_cert_dataset(val_df, val_cert_labels)
    
    print("Training datasets prepared successfully!")
    
    return quality_train_ds, quality_val_ds, cert_train_ds, cert_val_ds, quality_classes, cert_classes

def train_models(epochs=10):
    """Train both PCB analysis models."""
    print("\n=== Training PCB Analysis Models ===\n")
    
    # Ensure directory structure exists
    create_directory_structure()
    
    # Create dummy images if needed
    create_dummy_images()
    
    # Prepare training datasets
    quality_train_ds, quality_val_ds, cert_train_ds, cert_val_ds, quality_classes, cert_classes = prepare_training_datasets()
    
    if quality_train_ds is None:
        print("Failed to prepare training datasets. Exiting.")
        return
    
    # 1. Train Quality Check Model
    print("\n=== Training Quality Check Model ===\n")
    quality_model = PCBQualityCheckModel(num_classes=len(quality_classes))
    quality_model.class_names = quality_classes  # Set class names
    quality_model.build_model()
    
    # Train the model
    quality_model.train(quality_train_ds, quality_val_ds, epochs=epochs)
    
    # Fine-tune the model
    print("\n=== Fine-tuning Quality Check Model ===\n")
    quality_model.fine_tune(quality_train_ds, quality_val_ds, epochs=5)
    
    # Plot training history
    quality_model.plot_training_history()
    
    # Save the model
    quality_model.save_model()
    
    # 2. Train Certification Model
    print("\n=== Training Certification Model ===\n")
    cert_model = PCBCertificationModel(num_classes=len(cert_classes))
    cert_model.class_names = cert_classes  # Set class names
    cert_model.build_model()
    
    # Train the model
    cert_model.train(cert_train_ds, cert_val_ds, epochs=epochs)
    
    # Fine-tune the model
    print("\n=== Fine-tuning Certification Model ===\n")
    cert_model.fine_tune(cert_train_ds, cert_val_ds, epochs=5)
    
    # Plot training history
    cert_model.plot_training_history()
    
    # Save the model
    cert_model.save_model()
    
    print("\n=== Model Training Complete! ===\n")
    print("Trained models saved to the 'models' directory:")
    print("- Quality Check Model: models/pcb_quality_check_model.h5")
    print("- Certification Model: models/pcb_certification_model.h5")

if __name__ == "__main__":
    train_models(epochs=10)
