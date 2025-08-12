# pcb_model_trainer.py

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from dataset_loader import load_dataset, prepare_dataset_for_training
import matplotlib.pyplot as plt
import os

def create_model(num_classes, input_shape=(224, 224, 3), is_multilabel=False):
    """Create a transfer learning model based on MobileNetV2."""
    # Load the pretrained model
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Final layer depends on the task (single-label vs multi-label)
    if is_multilabel:
        # For multi-label classification (e.g., certifications)
        outputs = Dense(num_classes, activation='sigmoid')(x)
    else:
        # For single-label classification (e.g., quality check required)
        outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    # Compile the model
    if is_multilabel:
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
    else:
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def train_and_evaluate(target_column, epochs=10, batch_size=32):
    """Train and evaluate a model for the specified target."""
    # Load the dataset
    df = load_dataset()
    
    # Determine if this is a multi-label task
    is_multilabel = target_column == "certification_needed"
    
    # Prepare the dataset
    train_ds, val_ds, num_classes, class_names, encoder = prepare_dataset_for_training(
        df, target_column, batch_size=batch_size
    )
    
    # Create the model
    model = create_model(num_classes, is_multilabel=is_multilabel)
    
    # Create a directory for saving models
    os.makedirs("models", exist_ok=True)
    
    # Create a callback for saving the best model
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        f"models/pcb_{target_column}_model.h5",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_callback]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model Accuracy ({target_column})')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss ({target_column})')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"pcb_{target_column}_training_history.png")
    plt.close()
    
    print(f"Model for {target_column} trained and saved to models/pcb_{target_column}_model.h5")
    print(f"Training history saved to pcb_{target_column}_training_history.png")
    
    return model, encoder, class_names

if __name__ == "__main__":
    # Train models for both tasks
    print("Training model for quality check requirements...")
    quality_model, quality_encoder, quality_classes = train_and_evaluate(
        "quality_check_required", epochs=10
    )
    
    print("\nTraining model for certification needs...")
    cert_model, cert_encoder, cert_classes = train_and_evaluate(
        "certification_needed", epochs=10
    )
    
    print("\nTraining complete!")
