# File: main.py

import os
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import trainer_pb2, pusher_pb2
from tfx.components import (
    CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator,
    Transform, Trainer, Evaluator, Pusher
)
from tfx.orchestration import pipeline

# --- Konfigurasi Pipeline ---
PIPELINE_NAME = "student-score-predictor"
DATA_ROOT = "data"
MODULE_FILE = os.path.join("pipeline", "module.py")

# --- GANTI DENGAN USERNAME ANDA ---
PIPELINE_ROOT = os.path.join(os.getcwd(), "raffihakim-pipeline") 
# ---------------------------------

METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata", "metadata.db")
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, "serving_model", PIPELINE_NAME)

def create_pipeline(pipeline_name, pipeline_root, data_root, module_file, serving_model_dir, metadata_path):
    """Mendefinisikan dan membuat TFX pipeline."""
    
    # 1. Mengambil data dari file CSV.
    example_gen = CsvExampleGen(input_base=data_root)
    
    # 2. Menghitung statistik deskriptif dari data.
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    
    # 3. Membuat skema data secara otomatis.
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)
    
    # 4. Memvalidasi data untuk anomali.
    example_validator = ExampleValidator(statistics=statistics_gen.outputs['statistics'], schema=schema_gen.outputs['schema'])
    
    # 5. Melakukan feature engineering sesuai 'preprocessing_fn'.
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=module_file
    )
    
    # 6. Melatih model machine learning.
    trainer = Trainer(
        module_file=module_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=5000),
        eval_args=trainer_pb2.EvalArgs(num_steps=1000)
    )
    
    # 7. Mengevaluasi performa model (Resolver sudah implisit di sini).
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model']
    )
    
    # 8. Mendorong (deploy) model ke folder 'serving_model' jika performanya bagus.
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(base_directory=serving_model_dir)
        )
    )

    components = [
        example_gen, statistics_gen, schema_gen, example_validator,
        transform, trainer, evaluator, pusher
    ]
    
    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=f'sqlite:///{metadata_path}',
        components=components
    )

if __name__ == '__main__':
    # Menjalankan pipeline menggunakan Apache Beam
    BeamDagRunner().run(
        create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_root=DATA_ROOT,
            module_file=MODULE_FILE,
            serving_model_dir=SERVING_MODEL_DIR,
            metadata_path=METADATA_PATH
        )
    )