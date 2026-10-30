"""
PlantX Detector - shared model definition.

This file is the single source of truth for the Swin-style Transformer used by
both the training notebook and the Flask backend, so the two can never drift.
Compatible with TensorFlow 2.7.0 / tensorflow-addons 0.14.0.
"""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------------------------------------------------------
# Hyper-parameters
# --------------------------------------------------------------------------
IMG_SIDE = 64                       # images are resized to IMG_SIDE x IMG_SIDE
INPUT_SHAPE = (IMG_SIDE, IMG_SIDE, 3)
PIXEL_SCALE = 255.0                 # pixel normalisation divisor
PATCH_SIZE = (4, 4)                 # size of each image patch
EMBED_DIM = 256                     # token embedding width
NUM_HEADS = 4                       # attention heads
MLP_UNITS = 256                     # hidden units in the transformer MLP
QKV_BIAS = True                     # bias in query/key/value projections
WINDOW = 2                          # attention window size
SHIFT = 2                           # shifted-window offset
DROP_RATE = 0.2                     # dropout / stochastic-depth rate

PATCHES_X = INPUT_SHAPE[0] // PATCH_SIZE[0]
PATCHES_Y = INPUT_SHAPE[1] // PATCH_SIZE[1]

DEFAULT_CLASSES = ["bacterial_blight", "curl_virus", "fussarium_wilt", "healthy"]


# --------------------------------------------------------------------------
# Image helper (identical for training and inference)
# --------------------------------------------------------------------------
def prepare_image(bgr_image):
    """Resize a BGR (cv2) image and scale it to [0, 1]. Returns float32 HxWx3."""
    resized = cv2.resize(bgr_image, (IMG_SIDE, IMG_SIDE))
    return resized.astype("float32") / PIXEL_SCALE


# --------------------------------------------------------------------------
# Window helpers
# --------------------------------------------------------------------------
def split_into_windows(feature_map, win):
    _, h, w, c = feature_map.shape
    rows, cols = h // win, w // win
    feature_map = tf.reshape(feature_map, (-1, rows, win, cols, win, c))
    feature_map = tf.transpose(feature_map, (0, 1, 3, 2, 4, 5))
    return tf.reshape(feature_map, (-1, win, win, c))


def merge_windows(windows, win, h, w, c):
    rows, cols = h // win, w // win
    merged = tf.reshape(windows, (-1, rows, cols, win, win, c))
    merged = tf.transpose(merged, (0, 1, 3, 2, 4, 5))
    return tf.reshape(merged, (-1, h, w, c))


class StochasticDepth(layers.Layer):
    """Drops whole residual branches - ONLY while training."""

    def __init__(self, rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if not training or not self.rate:
            return inputs
        batch = tf.shape(inputs)[0]
        shape = (batch,) + (1,) * (inputs.shape.rank - 1)
        keep = 1.0 - self.rate
        mask = tf.floor(keep + tf.random.uniform(shape, dtype=inputs.dtype))
        return tf.math.divide(inputs, keep) * mask


# --------------------------------------------------------------------------
# Attention + transformer block
# --------------------------------------------------------------------------
class PlantXWindowAttention(layers.Layer):
    def __init__(self, dim, window, heads, qkv_bias=True, drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window = window
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv_dense = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.attn_drop = layers.Dropout(drop)
        self.out_dense = layers.Dense(dim)

    def build(self, input_shape):
        table_rows = (2 * self.window[0] - 1) * (2 * self.window[1] - 1)
        self.rel_bias_table = self.add_weight(
            name="rel_bias_table",
            shape=(table_rows, self.heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )
        ys = np.arange(self.window[0])
        xs = np.arange(self.window[1])
        grid = np.stack(np.meshgrid(ys, xs, indexing="ij"))
        flat = grid.reshape(2, -1)
        rel = flat[:, :, None] - flat[:, None, :]
        rel = rel.transpose([1, 2, 0])
        rel[:, :, 0] += self.window[0] - 1
        rel[:, :, 1] += self.window[1] - 1
        rel[:, :, 0] *= 2 * self.window[1] - 1
        self.rel_index = tf.Variable(
            initial_value=tf.convert_to_tensor(rel.sum(-1)),
            trainable=False, name="rel_index",
        )

    def call(self, tokens, mask=None, training=None):
        _, n_tok, channels = tokens.shape
        head_dim = channels // self.heads
        qkv = self.qkv_dense(tokens)
        qkv = tf.reshape(qkv, (-1, n_tok, 3, self.heads, head_dim))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q * self.scale) @ tf.transpose(k, (0, 1, 3, 2))

        win_area = self.window[0] * self.window[1]
        bias = tf.gather(self.rel_bias_table, tf.reshape(self.rel_index, (-1,)))
        bias = tf.reshape(bias, (win_area, win_area, -1))
        bias = tf.transpose(bias, (2, 0, 1))
        scores = scores + tf.expand_dims(bias, 0)

        if mask is not None:
            n_win = mask.get_shape()[0]
            mask_f = tf.cast(tf.expand_dims(tf.expand_dims(mask, 1), 0), tf.float32)
            scores = tf.reshape(scores, (-1, n_win, self.heads, n_tok, n_tok)) + mask_f
            scores = tf.reshape(scores, (-1, self.heads, n_tok, n_tok))
        weights = keras.activations.softmax(scores, axis=-1)
        weights = self.attn_drop(weights, training=training)

        out = weights @ v
        out = tf.transpose(out, (0, 2, 1, 3))
        out = tf.reshape(out, (-1, n_tok, channels))
        out = self.out_dense(out)
        return self.attn_drop(out, training=training)


class PlantXSwinBlock(layers.Layer):
    def __init__(self, dim, grid, heads, window=7, shift=0, mlp_units=1024,
                 qkv_bias=True, drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.grid = grid
        self.heads = heads
        self.window = window
        self.shift = shift
        self.mlp_units = mlp_units

        self.norm_a = layers.LayerNormalization(epsilon=1e-5)
        self.attention = PlantXWindowAttention(
            dim, (self.window, self.window), heads, qkv_bias=qkv_bias, drop=drop
        )
        self.path_drop = StochasticDepth(drop)
        self.norm_b = layers.LayerNormalization(epsilon=1e-5)
        self.mlp_dense_1 = layers.Dense(mlp_units)
        self.mlp_act = layers.Activation(keras.activations.gelu)
        self.mlp_drop_1 = layers.Dropout(drop)
        self.mlp_dense_2 = layers.Dense(dim)
        self.mlp_drop_2 = layers.Dropout(drop)

        if min(self.grid) < self.window:
            self.shift = 0
            self.window = min(self.grid)

    def build(self, input_shape):
        if self.shift == 0:
            self.attn_mask = None
            return
        h, w = self.grid
        h_cuts = (slice(0, -self.window), slice(-self.window, -self.shift),
                  slice(-self.shift, None))
        w_cuts = h_cuts
        canvas = np.zeros((1, h, w, 1))
        region = 0
        for hs in h_cuts:
            for ws in w_cuts:
                canvas[:, hs, ws, :] = region
                region += 1
        canvas = tf.convert_to_tensor(canvas)
        win_ids = split_into_windows(canvas, self.window)
        win_ids = tf.reshape(win_ids, (-1, self.window * self.window))
        diff = tf.expand_dims(win_ids, 1) - tf.expand_dims(win_ids, 2)
        diff = tf.where(diff != 0, -100.0, diff)
        diff = tf.where(diff == 0, 0.0, diff)
        self.attn_mask = tf.Variable(initial_value=diff, trainable=False, name="attn_mask")

    def _mlp(self, x, training):
        x = self.mlp_dense_1(x)
        x = self.mlp_act(x)
        x = self.mlp_drop_1(x, training=training)
        x = self.mlp_dense_2(x)
        return self.mlp_drop_2(x, training=training)

    def call(self, x, training=None):
        h, w = self.grid
        _, _, channels = x.shape
        shortcut = x
        x = self.norm_a(x)
        x = tf.reshape(x, (-1, h, w, channels))
        if self.shift > 0:
            x = tf.roll(x, shift=[-self.shift, -self.shift], axis=[1, 2])

        wins = split_into_windows(x, self.window)
        wins = tf.reshape(wins, (-1, self.window * self.window, channels))
        wins = self.attention(wins, mask=self.attn_mask, training=training)
        wins = tf.reshape(wins, (-1, self.window, self.window, channels))
        x = merge_windows(wins, self.window, h, w, channels)

        if self.shift > 0:
            x = tf.roll(x, shift=[self.shift, self.shift], axis=[1, 2])

        x = tf.reshape(x, (-1, h * w, channels))
        x = shortcut + self.path_drop(x, training=training)
        shortcut = x
        x = self.norm_b(x)
        x = self._mlp(x, training)
        return shortcut + self.path_drop(x, training=training)


# --------------------------------------------------------------------------
# Patch layers
# --------------------------------------------------------------------------
class PatchSlicer(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.px = patch_size[0]
        self.py = patch_size[0]

    def call(self, images):
        batch = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.px, self.py, 1),
            strides=(1, self.px, self.py, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        depth = patches.shape[-1]
        count = patches.shape[1]
        return tf.reshape(patches, (batch, count * count, depth))


class PatchProjector(layers.Layer):
    def __init__(self, n_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.n_patches = n_patches
        self.linear = layers.Dense(embed_dim)
        self.position = layers.Embedding(input_dim=n_patches, output_dim=embed_dim)

    def call(self, patches):
        idx = tf.range(start=0, limit=self.n_patches, delta=1)
        return self.linear(patches) + self.position(idx)


class PatchDownsample(layers.Layer):
    def __init__(self, grid, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.grid = grid
        self.embed_dim = embed_dim
        self.reduce = layers.Dense(2 * embed_dim, use_bias=False)

    def call(self, x):
        h, w = self.grid
        _, _, c = x.get_shape().as_list()
        x = tf.reshape(x, (-1, h, w, c))
        merged = tf.concat(
            (x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :],
             x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]), axis=-1)
        merged = tf.reshape(merged, (-1, (h // 2) * (w // 2), 4 * c))
        return self.reduce(merged)


# --------------------------------------------------------------------------
# Model factory
# --------------------------------------------------------------------------
def build_plantx_model(n_classes):
    """Builds the PlantX Swin-style classifier (uncompiled)."""
    inp = layers.Input(INPUT_SHAPE)
    x = inp
    x = PatchSlicer(PATCH_SIZE)(x)
    x = PatchProjector(PATCHES_X * PATCHES_Y, EMBED_DIM)(x)
    x = PlantXSwinBlock(
        dim=EMBED_DIM, grid=(PATCHES_X, PATCHES_Y), heads=NUM_HEADS,
        window=WINDOW, shift=0, mlp_units=MLP_UNITS,
        qkv_bias=QKV_BIAS, drop=DROP_RATE)(x)
    x = PlantXSwinBlock(
        dim=EMBED_DIM, grid=(PATCHES_X, PATCHES_Y), heads=NUM_HEADS,
        window=WINDOW, shift=SHIFT, mlp_units=MLP_UNITS,
        qkv_bias=QKV_BIAS, drop=DROP_RATE)(x)
    x = PatchDownsample((PATCHES_X, PATCHES_Y), embed_dim=EMBED_DIM)(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="plantx_swin")
