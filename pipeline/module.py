import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

NUMERIC_FEATURES = [
    'age', 
    'study_hours_per_day', 
    'social_media_hours', 
    'netflix_hours',
    'attendance_percentage', 
    'sleep_hours', 
    'mental_health_rating',
    'exercise_frequency'
]
LABEL_KEY = 'exam_score'
# ------------------------------------------------

def preprocessing_fn(inputs):
    outputs = {}
    
    for key in NUMERIC_FEATURES:
        dense_feature = tf.sparse.to_dense(inputs[key], default_value=0) if isinstance(inputs[key], tf.SparseTensor) else inputs[key]
        outputs[key] = tft.scale_to_z_score(tf.cast(dense_feature, tf.float32))

    outputs[LABEL_KEY] = tf.cast(inputs[LABEL_KEY], tf.float32)
    
    return outputs

def _build_keras_model():
    inputs = {key: tf.keras.layers.Input(shape=(1,), name=key, dtype=tf.float32) 
              for key in NUMERIC_FEATURES}
    
    concatenated_features = tf.keras.layers.concatenate(list(inputs.values()))
    
    x = tf.keras.layers.Dense(128, activation='relu')(concatenated_features)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    model.summary()
    return model

def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    feature_spec = tf_transform_output.transformed_feature_spec().copy()

    def _parse_function(example_proto):
        parsed_features = tf.io.parse_single_example(example_proto, feature_spec)
        label = parsed_features.pop(LABEL_KEY)
        return (parsed_features, label)

    train_dataset = tf.data.TFRecordDataset(
        filenames=fn_args.train_files, compression_type='GZIP'
    ).map(_parse_function).batch(32).repeat()
    
    eval_dataset = tf.data.TFRecordDataset(
        filenames=fn_args.eval_files, compression_type='GZIP'
    ).map(_parse_function).batch(32).repeat()

    model = _build_keras_model()

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=10
    )
    
    model.save(fn_args.serving_model_dir, save_format='tf')