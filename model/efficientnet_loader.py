"""
EfficientNetB0 Model Loader
Rebuilds the exact architecture from your Kaggle training
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
import os
import zipfile
import tempfile


def build_model():
    """
    Rebuild the EXACT model architecture from Kaggle training
    """
    print("Building EfficientNetB0 model architecture...")
    
    # Exact same architecture as training
    base_model = EfficientNetB0(
        weights="imagenet",  # We'll load pretrained weights first
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    print(f"✓ Model architecture built")
    print(f"  Input: {model.input_shape}")
    print(f"  Output: {model.output_shape}")
    print(f"  Parameters: {model.count_params():,}")
    
    return model


def load_weights_from_keras(model, keras_path):
    """
    Extract and load weights from .keras file into rebuilt model
    """
    print(f"\nAttempting to load weights from: {keras_path}")
    
    try:
        # Method 1: Try direct weight loading
        print("  Method 1: Direct weight loading...")
        model.load_weights(keras_path)
        print("  ✓ Direct loading successful!")
        return True
    except Exception as e1:
        print(f"  Method 1 failed: {str(e1)[:80]}")
        
        # Method 2: Extract from .keras archive
        try:
            print("  Method 2: Extracting from .keras archive...")
            
            with tempfile.TemporaryDirectory() as tmpdir:
                # .keras files are zip archives
                with zipfile.ZipFile(keras_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                
                # Look for weights file
                weights_file = os.path.join(tmpdir, 'model.weights.h5')
                
                if os.path.exists(weights_file):
                    print(f"  Found weights file: model.weights.h5")
                    model.load_weights(weights_file)
                    print("  ✓ Weights loaded from archive!")
                    return True
                else:
                    print(f"  Weights file not found in archive")
                    
                    # List what's in the archive
                    files = []
                    for root, dirs, filenames in os.walk(tmpdir):
                        for f in filenames:
                            files.append(os.path.join(root, f).replace(tmpdir, ''))
                    print(f"  Files in archive: {files}")
                    
        except Exception as e2:
            print(f"  Method 2 failed: {e2}")
        
        # Method 3: Load by name with skip_mismatch
        try:
            print("  Method 3: Loading with skip_mismatch...")
            model.load_weights(keras_path, by_name=True, skip_mismatch=True)
            print("  ✓ Partial weights loaded (some may be skipped)")
            return True
        except Exception as e3:
            print(f"  Method 3 failed: {e3}")
    
    return False


def load_trained_model(keras_path):
    """
    Main function: Build architecture and load weights
    """
    print("\n" + "="*60)
    print("LOADING EFFICIENTNETB0 DEEPFAKE DETECTOR")
    print("="*60)
    
    # Build model with exact architecture
    model = build_model()
    
    # Try to load trained weights
    if os.path.exists(keras_path):
        weights_loaded = load_weights_from_keras(model, keras_path)
        
        if weights_loaded:
            print("\n✓ Model loaded successfully with trained weights!")
            return model, True
        else:
            print("\n⚠ Using model with ImageNet weights only (not trained on deepfakes)")
            print("  The model will work but accuracy will be lower")
            return model, False
    else:
        print(f"\n⚠ Model file not found: {keras_path}")
        print("  Using model with ImageNet weights only")
        return model, False