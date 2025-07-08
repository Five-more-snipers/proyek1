# Submission 1: Machine Learning Pipeline - Student Score Predictor via Habits
Nama: Muhammad Raffi Hakim

Username dicoding: raffihakim

![Sumber Gambar](./img/dataset-cover.png)

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [student_habits_performance](https://www.kaggle.com/code/ahmednasra1/student-habits-performance/input) |
| Masalah | Banyak faktor di luar ruang kelas yang diduga memengaruhi kinerja akademis seorang mahasiswa, mulai dari durasi belajar, intensitas penggunaan media sosial, hingga gaya hidup seperti pola tidur dan frekuensi olahraga. Namun, mengukur dampak kuantitatif dari setiap kebiasaan ini terhadap hasil akhir, seperti skor ujian, merupakan sebuah tantangan. Institusi pendidikan dan mahasiswa sering kali kesulitan mengidentifikasi secara pasti kebiasaan mana yang paling signifikan berkontribusi pada keberhasilan atau kegagalan akademis. Oleh karena itu, masalah utamanya adalah bagaimana kita dapat memodelkan hubungan kompleks antara berbagai atribut dan kebiasaan sehari-hari mahasiswa untuk memprediksi skor ujian (exam_score) mereka secara akurat. |
| Solusi machine learning | Solusi yang diajukan adalah membangun pipeline machine learning end-to-end menggunakan TensorFlow Extended (TFX). Pipeline ini mengotomatiskan seluruh proses mulai dari penyerapan data, validasi, transformasi fitur, pencarian hyperparameter terbaik (tuning), pelatihan model, evaluasi, hingga penyiapan model untuk deployment (serving). Model yang digunakan adalah Neural Network yang dibangun dengan TensorFlow/Keras. |
| Metode pengolahan | Data diolah menggunakan komponen TFX Transform. Secara spesifik, fungsi preprocessing_fn dalam student_transform.py digunakan untuk memastikan semua fitur numerik dan label dikonversi ke tipe data tf.float32. Fitur numerik yang diproses antara lain: age, study_hours_per_day, social_media_hours, netflix_hours, attendance_percentage, sleep_hours, mental_health_rating, dan exercise_frequency. |
| Arsitektur model | Arsitektur model final ditentukan oleh Keras Tuner. Berdasarkan hasil tuning, arsitektur terbaik yang ditemukan adalah sebagai berikut: <br> • Input Layer: Menerima 8 fitur numerik. <br>  • Hidden Layer 1: Dense dengan 96 unit dan aktivasi relu.<br>  • Hidden Layer 2: Dense dengan 224 unit dan aktivasi relu.<br>  • Dropout: Regularisasi Dropout dengan rate 0.5. • Output Layer: Dense dengan 1 unit dan aktivasi linear.<br>  • Optimizer: Adam dengan learning rate 0.001.<br>  • Loss Function: Mean Squared Error (MSE). |
| Metrik evaluasi | Metrik utama yang digunakan untuk mengevaluasi dan mengoptimalkan model adalah Root Mean Squared Error (RMSE). Metrik ini dipilih karena cocok untuk masalah regresi dan memberikan gambaran error dalam satuan yang sama dengan label (skor ujian). Selama evaluasi, metrik MeanSquaredError dan ExampleCount juga dihitung untuk analisis lebih lanjut. |
| Performa model | Berdasarkan hasil dari TFX Tuner, performa terbaik yang dicapai oleh model pada data validasi adalah val_root_mean_squared_error sekitar 4.65. Ini menunjukkan bahwa model, dengan hyperparameter terbaiknya, mampu memprediksi skor ujian dengan rata-rata kesalahan sekitar 4.65 poin. |
