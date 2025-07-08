import tensorflow as tf

NUMERIC_FEATURES = [
    'age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
    'attendance_percentage', 'sleep_hours', 'mental_health_rating', 'exercise_frequency'
]
LABEL_KEY = 'exam_score'

def transformed_name(key):
    return f"{key}_xf"

def preprocessing_fn(inputs):
    outputs = {}
    for key in NUMERIC_FEATURES:
        outputs[transformed_name(key)] = tf.cast(inputs[key], tf.float32)
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.float32)
    return outputs