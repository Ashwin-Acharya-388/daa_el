"""
densenet_model.py — Module C: DenseNet Architecture for Tabular Data.

Implements a DenseNet adapted from the image-classification variant
(Huang et al., 2017) for loan default / financial distress prediction
on structured (tabular) data.

Key DenseNet principles retained:
  - Dense connectivity: each layer receives concatenated outputs
    from ALL preceding layers in the same Dense Block.
  - Bottleneck layers: two-stage projection for efficiency.
  - Transition layers: BatchNorm + compression to reduce dimensions.
  - Growth rate: fixed number of new features per layer.

Reference: Sayed et al. (2024) IEEE Access — DenseNet for loan prediction.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from sklearn.utils.class_weight import compute_class_weight

import config

# ─────────────────── Reproducibility ─────────────────────────────────
tf.random.set_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)


# ════════════════════════════════════════════════════════════════════════
#  DenseNet Building Blocks
# ════════════════════════════════════════════════════════════════════════

class BottleneckDenseLayer(layers.Layer):
    """
    A single bottleneck dense layer within a Dense Block.

    Structure: BN → ReLU → Dense(4*k) → BN → ReLU → Dense(k)
    where k = growth_rate.

    Each layer RECEIVES the concatenation of all previous layer outputs.
    """

    def __init__(self, growth_rate: int, bottleneck_factor: int = 4,
                 dropout_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.growth_rate = growth_rate
        inner_size = bottleneck_factor * growth_rate

        # Bottleneck path
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.dense1 = layers.Dense(inner_size, use_bias=False)

        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.dense2 = layers.Dense(growth_rate, use_bias=False)

        self.drop = layers.Dropout(dropout_rate) if dropout_rate > 0 else None

    def call(self, inputs, training=False):
        x = self.bn1(inputs, training=training)
        x = self.relu1(x)
        x = self.dense1(x)

        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.dense2(x)

        if self.drop is not None:
            x = self.drop(x, training=training)

        # Concatenate input + new features (dense connectivity)
        return layers.concatenate([inputs, x])


class DenseBlock(layers.Layer):
    """
    A Dense Block containing `num_layers` bottleneck dense layers.

    Each layer receives the concatenation of all preceding outputs
    within the block, growing the feature dimension by `growth_rate`
    at each step.
    """

    def __init__(self, num_layers: int, growth_rate: int,
                 bottleneck_factor: int = 4, dropout_rate: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.dense_layers = []
        for i in range(num_layers):
            self.dense_layers.append(
                BottleneckDenseLayer(
                    growth_rate=growth_rate,
                    bottleneck_factor=bottleneck_factor,
                    dropout_rate=dropout_rate,
                    name=f"bottleneck_{i}"
                )
            )

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x, training=training)
        return x


class TransitionLayer(layers.Layer):
    """
    Transition Layer between Dense Blocks.

    Reduces feature dimensionality via:
      BN → ReLU → Dense(compressed_size) → Dropout

    compressed_size = int(input_features * compression_factor)
    """

    def __init__(self, compression: float = 0.5, dropout_rate: float = 0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.compression = compression
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.dropout = layers.Dropout(dropout_rate)
        self._dense = None  # Built lazily to infer input size

    def build(self, input_shape):
        compressed_size = max(1, int(input_shape[-1] * self.compression))
        self._dense = layers.Dense(compressed_size, use_bias=False)
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.bn(inputs, training=training)
        x = self.relu(x)
        x = self._dense(x)
        x = self.dropout(x, training=training)
        return x


# ════════════════════════════════════════════════════════════════════════
#  Full DenseNet Model
# ════════════════════════════════════════════════════════════════════════

def build_densenet(n_features: int,
                   growth_rate: int = None,
                   num_blocks: int = None,
                   layers_per_block: int = None,
                   bottleneck_factor: int = None,
                   compression: float = None,
                   dropout_rate: float = None) -> Model:
    """
    Build a DenseNet model adapted for tabular binary classification.

    Parameters
    ----------
    n_features : int
        Number of input features.
    growth_rate : int
        New features added per dense layer (default: config.GROWTH_RATE).
    num_blocks : int
        Number of dense blocks (default: config.NUM_DENSE_BLOCKS).
    layers_per_block : int
        Layers within each dense block (default: config.LAYERS_PER_BLOCK).
    bottleneck_factor : int
        Bottleneck width multiplier (default: config.BOTTLENECK_FACTOR).
    compression : float
        Transition compression factor (default: config.COMPRESSION).
    dropout_rate : float
        Dropout rate (default: config.DROPOUT_RATE).

    Returns
    -------
    keras.Model — compiled DenseNet model.
    """
    growth_rate = growth_rate or config.GROWTH_RATE
    num_blocks = num_blocks or config.NUM_DENSE_BLOCKS
    layers_per_block = layers_per_block or config.LAYERS_PER_BLOCK
    bottleneck_factor = bottleneck_factor or config.BOTTLENECK_FACTOR
    compression = compression or config.COMPRESSION
    dropout_rate = dropout_rate or config.DROPOUT_RATE

    # ── Input ──
    inputs = keras.Input(shape=(n_features,), name="features")

    # Initial projection to a reasonable dimensionality
    x = layers.Dense(growth_rate * 2, use_bias=False, name="initial_projection")(inputs)
    x = layers.BatchNormalization(name="initial_bn")(x)
    x = layers.ReLU(name="initial_relu")(x)

    # ── Dense Blocks + Transition Layers ──
    for i in range(num_blocks):
        x = DenseBlock(
            num_layers=layers_per_block,
            growth_rate=growth_rate,
            bottleneck_factor=bottleneck_factor,
            dropout_rate=dropout_rate * 0.5,  # lighter dropout inside blocks
            name=f"dense_block_{i+1}"
        )(x)

        # Add transition layer after every block except the last
        if i < num_blocks - 1:
            x = TransitionLayer(
                compression=compression,
                dropout_rate=dropout_rate,
                name=f"transition_{i+1}"
            )(x)

    # ── Classification Head ──
    x = layers.BatchNormalization(name="final_bn")(x)
    x = layers.ReLU(name="final_relu")(x)
    x = layers.Dense(128, activation="relu", name="head_dense_1")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    x = layers.Dense(64, activation="relu", name="head_dense_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="DenseNet_Tabular")
    return model


def compile_model(model: Model, learning_rate: float = None) -> Model:
    """Compile the model with Adam optimizer and binary cross-entropy loss."""
    lr = learning_rate or config.LEARNING_RATE
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )
    return model


class MinEpochEarlyStopping(callbacks.EarlyStopping):
    """
    Custom EarlyStopping that only activates after a minimum number of epochs.
    This guarantees the model trains for at least `min_epochs` before
    early stopping can trigger.
    """

    def __init__(self, min_epochs: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.min_epochs = min_epochs

    def on_epoch_end(self, epoch, logs=None):
        # epoch is 0-indexed; don't allow stopping before min_epochs
        if epoch + 1 < self.min_epochs:
            return
        super().on_epoch_end(epoch, logs)


def get_callbacks() -> list:
    """
    Return training callbacks:
      - MinEpochEarlyStopping (monitor val_loss, only after MIN_EPOCHS)
      - ReduceLROnPlateau (only after MIN_EPOCHS)
      - ModelCheckpoint (save best model)
    """
    checkpoint_path = os.path.join(config.MODEL_DIR, "densenet_best.h5")
    min_epochs = getattr(config, "MIN_EPOCHS", 20)

    print(f"  Minimum epochs (guaranteed): {min_epochs}")

    return [
        MinEpochEarlyStopping(
            min_epochs=min_epochs,
            monitor="val_loss",
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_REDUCE_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        )
    ]


def compute_weights(y_train: np.ndarray) -> dict:
    """
    Compute class weights to handle remaining imbalance after SMOTE-ENN.

    Returns
    -------
    dict — {0: weight_0, 1: weight_1}
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = dict(zip(classes.astype(int), weights))
    print(f"[Model] Class weights: {weight_dict}")
    return weight_dict


def train_model(model: Model,
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                class_weight: dict = None) -> dict:
    """
    Train the DenseNet model.

    Parameters
    ----------
    model : compiled Keras Model
    X_train, y_train : training data
    X_val, y_val : validation / test data (used for val_loss monitoring)
    class_weight : dict — per-class sample weights

    Returns
    -------
    history : keras History object (as dict)
    """
    print(f"\n[Model] Training DenseNet...")
    print(f"  Epochs: {config.EPOCHS} (early stopping patience={config.EARLY_STOP_PATIENCE})")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weight,
        callbacks=get_callbacks(),
        verbose=1
    )

    # Save final model
    final_path = os.path.join(config.MODEL_DIR, "densenet_model.h5")
    model.save(final_path)
    print(f"\n[Model] Final model saved to: {final_path}")

    return history.history


# ─────────────────────── Standalone Test ─────────────────────────────
if __name__ == "__main__":
    # Quick architecture test with dummy data
    n_feat = 83
    model = build_densenet(n_feat)
    model = compile_model(model)
    model.summary()

    # Smoke test with random data
    dummy_X = np.random.randn(100, n_feat).astype(np.float32)
    dummy_y = np.random.randint(0, 2, 100)
    pred = model.predict(dummy_X[:5], verbose=0)
    print(f"\nSmoke test predictions: {pred.flatten()}")
