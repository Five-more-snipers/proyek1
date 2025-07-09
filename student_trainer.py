import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.v1.components import TunerFnResult

from student_transform import NUMERIC_FEATURES, LABEL_KEY, transformed_name

def _input_fn(file_pattern: str,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 64) -> tf.data.Dataset:
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=feature_spec,
        reader=lambda filenames: tf.data.TFRecordDataset(filenames, compression_type="GZIP"),
        label_key=transformed_name(LABEL_KEY)
    )
    return dataset

def _build_keras_model(hparams: dict) -> tf.keras.Model:
    inputs = {
        **{transformed_name(key): tf.keras.layers.Input(shape=(1,), name=transformed_name(key))
           for key in NUMERIC_FEATURES},
    }
    
    concatenated_features = layers.concatenate(list(inputs.values()))
    x = layers.BatchNormalization()(concatenated_features)
    
    for i in range(hparams['num_layers']):
        x = layers.Dense(units=hparams[f'units_{i}'], activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(hparams['dropout'])(x)
        
    output = layers.Dense(1, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hparams['learning_rate']),
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

    hparams = fn_args.hyperparameters['values']
    print("Hyperparameters Terbaik yang Digunakan:", hparams)
    
    model = _build_keras_model(hparams)
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_root_mean_squared_error',
        mode='min',
        verbose=1,
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_root_mean_squared_error',
        mode='min',
        verbose=1,
        factor=0.5,
        patience=3
    )

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=75,
        callbacks=[early_stopping_cb, reduce_lr_cb]
    )
    
    model.save(fn_args.serving_model_dir, save_format='tf')