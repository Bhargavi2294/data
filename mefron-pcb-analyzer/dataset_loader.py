# dataset_loader.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split

def load_dataset(csv_path="data/pcb_dataset.csv"):
    """Load the PCB dataset from CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {len(df)} entries")
    return df

def prepare_labels(df, target_column):
    """Prepare labels for classification."""
    if target_column == "quality_check_required":
        # Single-label classification for quality check
        encoder = LabelEncoder()
        labels = encoder.fit_transform(df[target_column])
        num_classes = len(encoder.classes_)
        class_names = encoder.classes_
        
        return labels, num_classes, class_names, encoder
    
    elif target_column == "certification_needed":
        # Multi-label classification for certifications
        # Split the semicolon-separated values into lists
        cert_lists = df[target_column].apply(lambda x: x.split(';'))
        
        # Use MultiLabelBinarizer to convert to one-hot encoding
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(cert_lists)
        num_classes = len(mlb.classes_)
        class_names = mlb.classes_
        
        return labels, num_classes, class_names, mlb
    
    else:
        raise ValueError(f"Unsupported target column: {target_column}")

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image."""
    # This function assumes images are available at the paths specified in the CSV
    # In practice, you'll need to adjust paths based on your actual image locations
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        # Normalize pixel values to [0, 1]
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # Return a blank image if loading fails
        return np.zeros((*target_size, 3))

def prepare_dataset_for_training(df, target_column, batch_size=32, img_size=(224, 224)):
    """Prepare the dataset for model training."""
    # Get labels and metadata
    labels, num_classes, class_names, encoder = prepare_labels(df, target_column)
    
    # Split into training and validation sets
    train_df, val_df, train_labels, val_labels = train_test_split(
        df, labels, test_size=0.2, random_state=42, stratify=labels if target_column == "quality_check_required" else None
    )
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    
    # Create TensorFlow datasets
    def create_dataset(dataframe, labels):
        # This is a simplified example - in practice, you'd need to handle
        # loading actual images from your filesystem
        
        def generator():
            for i in range(len(dataframe)):
                img_path = dataframe.iloc[i]["image_path"]
                label = labels[i]
                img = load_and_preprocess_image(img_path, img_size)
                yield img, label
        
        # Create dataset with the right types and shapes
        output_shapes = ((img_size[0], img_size[1], 3), 
                         () if target_column == "quality_check_required" else (num_classes,))
        output_types = (tf.float32, tf.int32 if target_column == "quality_check_required" else tf.float32)
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=output_types,
            output_shapes=output_shapes
        )
        
        # Batch and prefetch for performance
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
    
    train_dataset = create_dataset(train_df, train_labels)
    val_dataset = create_dataset(val_df, val_labels)
    
    return train_dataset, val_dataset, num_classes, class_names, encoder

# Example usage
if __name__ == "__main__":
    # Load the dataset
    df = load_dataset()
    
    # Display some statistics
    print("\nPCB Type Distribution:")
    print(df["pcb_type"].value_counts())
    
    print("\nQuality Check Requirements Distribution:")
    print(df["quality_check_required"].value_counts())
    
    print("\nCommon Certifications:")
    all_certs = []
    for cert_list in df["certification_needed"].str.split(';'):
        all_certs.extend(cert_list)
    print(pd.Series(all_certs).value_counts())
    
    print("\nPreparing dataset for 'quality_check_required' classification...")
    train_ds, val_ds, num_classes, class_names, encoder = prepare_dataset_for_training(
        df, "quality_check_required"
    )
    
    print(f"\nPrepared for classification with {num_classes} classes: {class_names}")
    
    # Note: In practice, you'd need to make sure images are available at the paths
    # specified in the CSV before running this code.
