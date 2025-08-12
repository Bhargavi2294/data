# certification_model.py

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import os
import json

class PCBCertificationModel:
    """Model for predicting certification requirements for PCBs."""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=7):
        """
        Initialize the PCB certification model.
        
        Args:
            input_shape: Input image dimensions (height, width, channels)
            num_classes: Number of certification classes
                         (default: 7 - CE, RoHS, UL, FCC, ISO9001, IEC60950, IATF16949)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.class_names = ['CE', 'RoHS', 'UL', 'FCC', 'ISO9001', 'IEC60950', 'IATF16949']
        self.history = None
        
    def build_model(self):
        """Build the model architecture using transfer learning with MobileNetV2."""
        # Load the pretrained base model
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Create the model architecture
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        # Use sigmoid activation for multi-label classification
        outputs = Dense(self.num_classes, activation='sigmoid')(x)
        
        self.model = Model(inputs, outputs)
        
        # Compile the model with binary crossentropy for multi-label classification
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        print("Certification model built successfully!")
        return self.model
    
    def train(self, train_dataset, validation_dataset, epochs=10, callbacks=None):
        """
        Train the model.
        
        Args:
            train_dataset: TensorFlow dataset for training
            validation_dataset: TensorFlow dataset for validation
            epochs: Number of training epochs
            callbacks: List of Keras callbacks for training
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
            
        # Default callbacks if none provided
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=3,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.2,
                    patience=2
                )
            ]
            
            # Create directory for model checkpoints if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Add model checkpoint callback
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='models/pcb_certification_model.h5',
                    monitor='val_auc',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                )
            )
        
        # Train the model
        self.history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the class names
        with open('models/certification_classes.json', 'w') as f:
            json.dump(self.class_names, f)
        
        return self.history
    
    def fine_tune(self, train_dataset, validation_dataset, epochs=5, learning_rate=0.0001):
        """
        Fine-tune the model by unfreezing the top layers of the base model.
        
        Args:
            train_dataset: TensorFlow dataset for training
            validation_dataset: TensorFlow dataset for validation
            epochs: Number of fine-tuning epochs
            learning_rate: Learning rate for fine-tuning
            
        Returns:
            Fine-tuning history
        """
        # Unfreeze the top layers of the base model
        base_model = self.model.layers[1]  # Get the base model
        
        # Unfreeze the top 30 layers
        for layer in base_model.layers[-30:]:
            layer.trainable = True
            
        # Recompile the model with a lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        # Create callback for model checkpoint
        os.makedirs('models', exist_ok=True)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='models/pcb_certification_model_finetuned.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Fine-tune the model
        finetune_history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=[checkpoint_callback],
            verbose=1
        )
        
        # Combine the training history
        if self.history is not None:
            for metric in self.history.history:
                self.history.history[metric].extend(finetune_history.history[metric])
        else:
            self.history = finetune_history
            
        return finetune_history
    
    def plot_training_history(self, save_path='certification_training_history.png'):
        """
        Plot the training history.
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
            
        # Create the plot
        plt.figure(figsize=(16, 6))
        
        # Plot accuracy
        plt.subplot(1, 3, 1)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot loss
        plt.subplot(1, 3, 2)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot AUC
        plt.subplot(1, 3, 3)
        plt.plot(self.history.history['auc'])
        plt.plot(self.history.history['val_auc'])
        plt.title('Model AUC')
        plt.ylabel('AUC')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Training history plot saved to {save_path}")
    
    def save_model(self, filepath='models/pcb_certification_model.h5'):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            print("No model to save. Build the model first.")
            return
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        self.model.save(filepath)
        
        # Save the class names
        class_names_path = os.path.join(os.path.dirname(filepath), 'certification_classes.json')
        with open(class_names_path, 'w') as f:
            json.dump(self.class_names, f)
            
        print(f"Model saved to {filepath}")
        print(f"Class names saved to {class_names_path}")
    
    def load_model(self, filepath='models/pcb_certification_model.h5'):
        """
        Load the model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        # Load the model
        self.model = tf.keras.models.load_model(filepath)
        
        # Load the class names
        class_names_path = os.path.join(os.path.dirname(filepath), 'certification_classes.json')
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        
        print(f"Model loaded from {filepath}")
        if os.path.exists(class_names_path):
            print(f"Class names loaded from {class_names_path}")
        
        return self.model
    
    def predict(self, image, threshold=0.5):
        """
        Make a prediction for a single image.
        
        Args:
            image: Image array with shape matching input_shape or a path to an image file
            threshold: Probability threshold for positive classification (default: 0.5)
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not initialized. Build or load a model first.")
            
        # Handle different input types
        if isinstance(image, str):
            # Load image from file
            img = tf.keras.preprocessing.image.load_img(
                image, target_size=self.input_shape[:2]
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
        elif isinstance(image, np.ndarray):
            # Ensure the image has the right shape
            if image.ndim == 3:  # Single image without batch dimension
                img_array = np.expand_dims(image, axis=0)
            else:
                img_array = image
        else:
            raise ValueError("Unsupported image format. Provide a file path or numpy array.")
            
        # Make prediction
        predictions = self.model.predict(img_array)
        
        # Get the predicted certifications (multi-label)
        predicted_certifications = []
        probabilities = {}
        
        for i, prob in enumerate(predictions[0]):
            class_name = self.class_names[i]
            probability = float(prob) * 100
            probabilities[class_name] = probability
            
            if prob >= threshold:
                predicted_certifications.append(class_name)
        
        # Return the results
        return {
            'certification_needed': predicted_certifications,
            'probabilities': probabilities
        }


# Example usage
if __name__ == "__main__":
    # Create the model
    model = PCBCertificationModel()
    model.build_model()
    
    # Print model summary
    model.model.summary()
    
    print("\nThis script defines the PCB Certification Model.")
    print("To train the model, you need to:")
    print("1. Prepare a dataset using dataset_loader.py")
    print("2. Call model.train(train_dataset, validation_dataset)")
    print("3. Optionally fine-tune with model.fine_tune()")
    print("4. Save the model with model.save_model()")
