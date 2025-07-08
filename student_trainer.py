# File: pipeline/student_trainer.py (Versi Final yang Paling Sederhana)

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

# Impor deklarasi fitur dari file transform
from student_transform import NUMERIC_FEATURES, LABEL_KEY, transformed_name

def _input_fn(file_pattern, tf_transform_output, batch_size=64):
    """Fungsi untuk memuat data TFRecord yang sudah di-transform."""
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=feature_spec,
        reader=lambda filenames: tf.data.TFRecordDataset(filenames, compression_type="GZIP"),
        label_key=transformed_name(LABEL_KEY)
    )
    return dataset

def _build_keras_model():
    """Membangun arsitektur model Keras."""
    inputs = {
        transformed_name(key): tf.keras.layers.Input(shape=(1,), name=transformed_name(key)) 
        for key in NUMERIC_FEATURES
    }
    
    concatenated_features = layers.concatenate(list(inputs.values()))
    
    x = layers.Dense(128, activation='relu')(concatenated_features)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(1, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    model.summary()
    return model

def run_fn(fn_args: FnArgs):
    """Fungsi utama yang dijalankan oleh TFX Trainer."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output)

    model = _build_keras_model()
    
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=15
    )
    
    # Simpan model dengan cara paling standar
    model.save(fn_args.serving_model_dir, save_format='tf')