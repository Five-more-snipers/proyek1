import keras_tuner as kt
import tensorflow as tf
from tfx.components.trainer.fn_args_utils import FnArgs
from typing import NamedTuple, Dict, Text, Any
from keras_tuner.engine import base_tuner
from tensorflow.keras import layers
from student_transform import NUMERIC_FEATURES, LABEL_KEY, transformed_name
import tensorflow_transform as tft

TunerFnResult = NamedTuple("TunerFnResult", [("tuner", base_tuner.BaseTuner), ("fit_kwargs", Dict[Text, Any])])

def _input_fn(file_pattern, tf_transform_output, batch_size=64):
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=tf_transform_output.transformed_feature_spec(),
        reader=lambda filenames: tf.data.TFRecordDataset(filenames, compression_type="GZIP"),
        label_key=transformed_name(LABEL_KEY)
    )
    return dataset

def model_builder(hp):
    inputs = {transformed_name(key): layers.Input(shape=(1,), name=transformed_name(key)) for key in NUMERIC_FEATURES}
    concatenated_features = layers.concatenate(list(inputs.values()))
    x = concatenated_features
    for i in range(hp.Int('num_layers', 1, 3)):
        x = layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32), activation='relu')(x)
    x = layers.Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1))(x)
    output = layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Fungsi utama yang dijalankan oleh komponen Tuner TFX."""
    
    # Setup Tuner
    tuner = kt.RandomSearch(
        model_builder,
        objective=kt.Objective("val_root_mean_squared_error", direction="min"),
        max_trials=10,
        directory=fn_args.working_dir,
        project_name="student_performance_tuning"
    )
    
    # --- PERBAIKAN KUNCI ADA DI SINI ---
    # Gunakan fn_args.transform_graph_path yang berisi path langsung
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    # ------------------------------------
    
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output)
    
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "epochs": 30
        },
    )