"""
Food Classification Model Training Script
==========================================
Trains a CNN model to classify food types from images extracted from videos.

This script uses transfer learning with MobileNetV2 as the base model,
which is efficient and works well for food classification tasks.

Usage:
    python scripts/train_model.py [--data-dir training_data] [--epochs 20] [--batch-size 32]
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_model(num_classes: int, input_shape=(224, 224, 3)):
    """
    Create a food classification model using MobileNetV2 transfer learning.
    
    Args:
        num_classes: Number of food classes to classify
        input_shape: Input image shape (default: 224x224x3)
    
    Returns:
        Compiled Keras model
    """
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    
    # Load pre-trained MobileNetV2 without top layers
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model


def create_data_generators(data_dir: Path, batch_size: int = 32, validation_split: float = 0.2):
    """
    Create training and validation data generators with augmentation.
    
    Args:
        data_dir: Directory containing class subdirectories with images
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
    
    Returns:
        Tuple of (train_generator, validation_generator, class_names)
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Validation data - only rescaling, no augmentation
    val_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=validation_split
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator, train_generator.class_indices


def fine_tune_model(model, base_model, train_generator, validation_generator, epochs: int = 10):
    """
    Fine-tune the model by unfreezing some base model layers.
    
    Args:
        model: The compiled model
        base_model: The base MobileNetV2 model
        train_generator: Training data generator
        validation_generator: Validation data generator
        epochs: Number of fine-tuning epochs
    
    Returns:
        Training history
    """
    from tensorflow.keras.optimizers import Adam
    
    # Unfreeze the top layers of the base model
    base_model.trainable = True
    
    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        verbose=1
    )
    
    return history


def save_labels(class_indices: dict, output_path: Path):
    """
    Save class labels in the format expected by food_classifier.py.
    
    Args:
        class_indices: Dictionary mapping class names to indices
        output_path: Path to save the labels file
    """
    # Invert the dictionary to get index -> class name
    index_to_class = {v: k for k, v in class_indices.items()}
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx in sorted(index_to_class.keys()):
            class_name = index_to_class[idx]
            f.write(f"{idx} {class_name}\n")
    
    print(f"Labels saved to {output_path}")


def plot_training_history(history, output_path: Path):
    """
    Plot and save training history graphs.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        axes[0].plot(history.history['accuracy'], label='Training')
        axes[0].plot(history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss plot
        axes[1].plot(history.history['loss'], label='Training')
        axes[1].plot(history.history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"Training history plot saved to {output_path}")
    except ImportError:
        print("matplotlib not available, skipping plot generation")


def main():
    parser = argparse.ArgumentParser(
        description="Train food classification model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="training_data",
        help="Directory containing training data organized by class (default: training_data)",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="keras_model.h5",
        help="Output model filename (default: keras_model.h5)",
    )
    parser.add_argument(
        "--output-labels",
        type=str,
        default="labels.txt",
        help="Output labels filename (default: labels.txt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=10,
        help="Number of fine-tuning epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )
    parser.add_argument(
        "--no-fine-tune",
        action="store_true",
        help="Skip fine-tuning phase",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = PROJECT_ROOT / args.data_dir
    output_model_path = PROJECT_ROOT / args.output_model
    output_labels_path = PROJECT_ROOT / args.output_labels
    
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        print("Run extract_frames.py first to extract training data from videos")
        sys.exit(1)
    
    # Check for class subdirectories
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not class_dirs:
        print(f"Error: No class subdirectories found in {data_dir}")
        print("Expected structure: training_data/<class_name>/*.jpg")
        sys.exit(1)
    
    print("=" * 60)
    print("Food Classification Model Training")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    print(f"Output model: {output_model_path}")
    print(f"Output labels: {output_labels_path}")
    print(f"Epochs: {args.epochs} + {args.fine_tune_epochs if not args.no_fine_tune else 0} fine-tune")
    print(f"Batch size: {args.batch_size}")
    print("-" * 60)
    
    # Import TensorFlow after parsing args (slow import)
    print("\nLoading TensorFlow...")
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Set memory growth to avoid GPU memory issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU available: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU detected, using CPU")
    
    # Create data generators
    print("\nCreating data generators...")
    train_gen, val_gen, class_indices = create_data_generators(
        data_dir,
        batch_size=args.batch_size,
        validation_split=args.validation_split
    )
    
    num_classes = len(class_indices)
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Classes: {class_indices}")
    
    # Create model
    print("\nCreating model...")
    model, base_model = create_model(num_classes)
    model.summary()
    
    # Callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            str(output_model_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Initial training with frozen base
    print("\n" + "=" * 60)
    print("Phase 1: Training with frozen base model")
    print("=" * 60)
    
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning phase
    if not args.no_fine_tune and args.fine_tune_epochs > 0:
        print("\n" + "=" * 60)
        print("Phase 2: Fine-tuning top layers")
        print("=" * 60)
        
        fine_tune_history = fine_tune_model(
            model, base_model, train_gen, val_gen, epochs=args.fine_tune_epochs
        )
        
        # Combine histories
        for key in history.history:
            if key in fine_tune_history.history:
                history.history[key].extend(fine_tune_history.history[key])
    
    # Save the final model
    print("\nSaving model...")
    model.save(str(output_model_path))
    print(f"Model saved to {output_model_path}")
    
    # Save labels
    save_labels(class_indices, output_labels_path)
    
    # Plot training history
    plot_path = PROJECT_ROOT / "training_history.png"
    plot_training_history(history, plot_path)
    
    # Save training metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "num_classes": num_classes,
        "classes": class_indices,
        "training_samples": train_gen.samples,
        "validation_samples": val_gen.samples,
        "epochs": len(history.history['accuracy']),
        "final_accuracy": float(history.history['accuracy'][-1]),
        "final_val_accuracy": float(history.history['val_accuracy'][-1]),
        "batch_size": args.batch_size,
    }
    
    metadata_path = PROJECT_ROOT / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Training metadata saved to {metadata_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"\nModel saved to: {output_model_path}")
    print(f"Labels saved to: {output_labels_path}")


if __name__ == "__main__":
    main()
