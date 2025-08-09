import argparse
import json
import logging
import os
import random
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(input_dim: int, num_classes: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(1470, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(832, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(428, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(264, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def save_labels(label_encoder: LabelEncoder, labels_path: str) -> None:
    classes: List[str] = list(label_encoder.classes_)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, ensure_ascii=False, indent=2)
    logging.info("Saved labels to %s (%d classes)", labels_path, len(classes))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ISL keypoint classifier")
    parser.add_argument("--csv", default="hand_keypoints.csv", help="Path to keypoints CSV")
    parser.add_argument("--model_out", default="model.keras", help="Output path for Keras model (.keras)")
    parser.add_argument(
        "--h5_out", default="model_v2.h5", help="Optional legacy H5 output path"
    )
    parser.add_argument("--labels_out", default="labels.json", help="Where to save label classes")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    set_global_seeds(args.seed)

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    logging.info("Loading data from %s", args.csv)
    df = pd.read_csv(args.csv, header=None, dtype={0: str}, low_memory=False)

    # Features and labels
    X = df.iloc[:, 1:].astype(np.float32).fillna(0.0).to_numpy()
    y_str = df[0].astype(str).to_numpy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)
    num_classes = len(label_encoder.classes_)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=args.seed, stratify=y
    )

    logging.info("Input dim: %d | Classes: %d | Train: %d | Val: %d", X.shape[1], num_classes, X_train.shape[0], X_val.shape[0])

    model = build_model(input_dim=X.shape[1], num_classes=num_classes)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=args.patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(1, args.patience // 2), verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=args.model_out, monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    val_probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(val_probs, axis=1)

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

    logging.info("Validation Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f", acc, prec, rec, f1)
    logging.debug("\n%s", classification_report(y_val, y_pred, zero_division=0))

    # Persist labels and model
    save_labels(label_encoder, args.labels_out)

    # Ensure best model saved by checkpoint; also save H5 for compatibility
    try:
        model.save(args.h5_out)
        logging.info("Saved legacy H5 model to %s", args.h5_out)
    except Exception as exc:
        logging.warning("Could not save H5 model: %s", exc)

    # If the checkpoint already wrote the best .keras, ensure it exists; if not, save current
    if not os.path.exists(args.model_out):
        model.save(args.model_out)
        logging.info("Saved Keras model to %s", args.model_out)


if __name__ == "__main__":
    main()
