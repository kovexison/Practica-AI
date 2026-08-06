import tensorflow as tf

# Define data augmentation block using Keras layers.
# Mirrors ImageDataGenerator(rotation_range=20, width_shift_range=0.1,
# height_shift_range=0.1, zoom_range=0.15, horizontal_flip=True, fill_mode='nearest').
# Rescaling by 1/255 is handled downstream in the pipeline, not here.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(20 / 360, fill_mode='nearest'),
    tf.keras.layers.RandomTranslation(
        height_factor=0.1, width_factor=0.1, fill_mode='nearest'
    ),
    tf.keras.layers.RandomZoom(0.15, fill_mode='nearest'),
])

def create_dataset_pipeline(directory_path, img_size=(224, 224), batch_size=32, shuffle=True, augment=False, color_mode='rgb', rescale=True, seed=42):
    """
    Constructs a high-performance tf.data.Dataset pipeline from a directory of images.

    This pipeline handles loading, resizing, and optionally normalizing pixel values to the [0, 1] range.
    It optionally supports dynamic, on-the-fly data augmentation and grayscale processing.

    Args:
        directory_path (str): Path to the target image directory.
        img_size (tuple): Target size for resizing images (width, height). Default is (224, 224).
        batch_size (int): Number of images per batch. Default is 32.
        shuffle (bool): Whether to shuffle the data (recommended for training). Default is True.
        augment (bool): If True, applies random geometric augmentations. Default is False.
        color_mode (str): 'rgb' for 3-channel color or 'grayscale' for 1-channel. Default is 'rgb'.
        rescale (bool): If True, normalizes pixel values to [0, 1]. Set to False when the model
            already applies its own preprocessing (e.g. transfer learning with preprocess_input).
            Default is True.

    Returns:
        tf.data.Dataset: The configured dataset, ready for model training or evaluation.
    """
    
    # 1. Load the dataset from the specific folder
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory_path,
        image_size=img_size,
        batch_size=batch_size,
        color_mode=color_mode,
        shuffle=shuffle,
        label_mode = 'categorical',
        seed = seed
    )

    # 2. Apply on-the-fly data augmentation if enabled
    if augment:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # 3. Apply normalization using map (skip when the model handles its own preprocessing)
    if rescale:
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        dataset = dataset.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # 4. Optimize performance by prefetching data batches in the background
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset