# File: pipeline/module.py

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

# Definisikan fitur dan label berdasarkan kolom di CSV Anda
NUMERIC_FEATURES = ['Hours_Studied', 'Previous_Scores', 'Sleep_Hours', 'Sample_Papers_Practiced']
CATEGORICAL_FEATURES = ['Extracurricular_Activities']
LABEL_KEY = 'Performance_Index'

def preprocessing_fn(inputs):
    """Fungsi untuk prapemrosesan data (Feature Engineering)."""
    outputs = {}
    
    # Normalisasi fitur numerik menggunakan Z-score
    for key in NUMERIC_FEATURES:
        outputs[key] = tft.scale_to_z_score(inputs[key])
    
    # Konversi fitur kategorikal 'Yes'/'No' menjadi angka (0 atau 1)
    for key in CATEGORICAL_FEATURES:
        outputs[key] = tft.compute_and_apply_vocabulary(
            tf.strings.lower(inputs[key]),
            vocab_filename=key
        )
    
    # Biarkan kolom target (label) apa adanya
    outputs[LABEL_KEY] = inputs[LABEL_KEY]
    
    return outputs

def _build_keras_model():
    """Membangun arsitektur model Keras."""
    
    # Tentukan layer input untuk setiap fitur
    inputs = {
        key: tf.keras.layers.Input(shape=(1,), name=key, dtype=tf.float32) for key in NUMERIC_FEATURES
    }
    inputs.update({
        key: tf.keras.layers.Input(shape=(1,), name=key, dtype=tf.int64) for key in CATEGORICAL_FEATURES
    })
    
    # Gabungkan semua input menjadi satu vektor fitur
    concatenated_features = tf.keras.layers.concatenate(list(inputs.values()))
    
    # Bangun arsitektur Deep Neural Network (DNN)
    x = tf.keras.layers.Dense(128, activation='relu')(concatenated_features)
    x = tf.keras.layers.Dropout(0.3)(x) # Mencegah overfitting
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='linear')(x) # Output layer untuk prediksi regresi
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    # Compile model dengan loss function dan metrik untuk regresi
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    
    model.summary() # Menampilkan ringkasan arsitektur model di log
    return model

def run_fn(fn_args: FnArgs):
    """Fungsi utama yang akan dijalankan oleh komponen Trainer TFX."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    # Siapkan dataset training dan evaluasi
    train_dataset = tf_transform_output.transform_raw_features(
        tf.data.TFRecordDataset(fn_args.train_files, compression_type="GZIP")
    )
    eval_dataset = tf_transform_output.transform_raw_features(
        tf.data.TFRecordDataset(fn_args.eval_files, compression_type="GZIP")
    )
    
    model = _build_keras_model()
    
    # Latih model
    model.fit(
        train_dataset.batch(64).repeat(),
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset.batch(64).repeat(),
        validation_steps=fn_args.eval_steps,
        epochs=10 # Jumlah epoch bisa disesuaikan
    )
    
    # Simpan model yang sudah dilatih
    model.save(fn_args.serving_model_dir, save_format='tf')