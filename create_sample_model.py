"""
create_sample_model.py
----------------------
Utility script to create and save a sample MobileNetV2-based Transfer Learning model
for the 7 HAM10000 skin disease classes.

Run this script to initialize `model/model.keras` for local testing:
    python create_sample_model.py
"""

import os
import sys
import json

def create_sample_model():
    print("=" * 60)
    print("Initializing Sample AI Model for Skin Disease Detection...")
    print("=" * 60)

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        print(f"[*] TensorFlow version detected: {tf.__version__}")
    except ImportError:
        print("[!] Error: TensorFlow is not installed in the current environment.")
        print("[!] Please run: pip install tensorflow keras")
        return False

    # Target directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    target_model_path = os.path.join(model_dir, "model.keras")

    # Load classes metadata to get output class count
    classes_file = os.path.join(model_dir, "classes.json")
    num_classes = 7
    if os.path.exists(classes_file):
        try:
            with open(classes_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                num_classes = len(data.get("classes", [])) or 7
        except Exception as e:
            print(f"[*] Note: Could not parse classes.json ({e}), defaulting to 7 classes.")

    print(f"[*] Building MobileNetV2 architecture with {num_classes} output classes...")

    try:
        # Build transfer learning model with MobileNetV2 backbone
        input_shape = (224, 224, 3)
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False  # Freeze feature extractor

        inputs = keras.Input(shape=input_shape, name="skin_image_input")
        # Standard MobileNetV2 preprocessing: scales pixel values to [-1, 1]
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.Dropout(0.3, name="dropout")(x)
        x = layers.Dense(128, activation="relu", name="dense_features")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="disease_prediction")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="SkinDisease_MobileNetV2")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        print("[*] Model Architecture Summary:")
        model.summary(line_length=70)

        # Save model
        print(f"[*] Saving model to {target_model_path}...")
        model.save(target_model_path)

        print("=" * 60)
        print("[+] SUCCESS: Model saved successfully at:")
        print(f"    {target_model_path}")
        print("    You can now run 'python app.py' to start the application with a connected model.")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"[!] Error creating sample model: {e}")
        # Fallback: create a simple sequential CNN if downloading weights fails
        print("[*] Attempting fallback to lightweight custom CNN...")
        try:
            model = keras.Sequential([
                layers.Input(shape=(224, 224, 3)),
                layers.Rescaling(1./255),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.2),
                layers.Dense(64, activation="relu"),
                layers.Dense(num_classes, activation="softmax")
            ], name="SkinDisease_CNN_Fallback")
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            model.save(target_model_path)
            print(f"[+] Fallback CNN saved successfully to {target_model_path}")
            return True
        except Exception as err2:
            print(f"[!] Fallback also failed: {err2}")
            return False

if __name__ == "__main__":
    success = create_sample_model()
    sys.exit(0 if success else 1)
