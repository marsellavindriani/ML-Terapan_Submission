# Laporan Proyek Machine Learning - Marsella Vindriani

## Domain Proyek

Proyek ini menggunakan model pembelajaran mesin untuk memprediksi diagnosis kanker payudara (jinak atau ganas) berdasarkan fitur input dari kumpulan data. Dataset ini mencakup atribut klinis utama yang dapat membantu mendeteksi kanker payudara secara dini.

## Latar Belakang
Mengapa Masalah Kanker Payudara Harus Segera Diselesaikan?
Kanker payudara merupakan penyakit dimana sel - sel payudara abnormal tumbuh diluar kendali dan membentuk tumor. Jika dibiarkan, tumor tersebut dapat menyebar ke seluruh tubuh termasuk jaringan payudara terdekat (invasi) dimana nantinya tumor akan menyebabkan benjolan dan penebalan dan berakibat fatal. 

Kanker payudara menjadi jenis kanker yang menempati posisi penyumbang kematian terbesar di Indonesia, dengan angka kematian mencapai lebih dari 22 ribu jiwa. Berdasarkan data Globocan tahun 2020, jumlah kasus baru kanker payudara mencapai 68.858 kasus (16,6%) dari total 396.914 kasus baru kanker di Indonesia. [Kemenkes RI](https://sehatnegeriku.kemkes.go.id/baca/umum/20220202/1639254/kanker-payudaya-paling-banyak-di-indonesia-kemenkes-targetkan-pemerataan-layanan-kesehatan/?utm_source=chatgpt.com) 
  
  
## Business Understanding

### Problem Statements
berdasarkan latar belakang diatas, berikut ini rumusan masalah yang dapat diselesaikan pada proyek ini:
1. Bagaimana cara melakukan pra-pemrosesan pada data penyakit kanker payudara yang akan digunakan untuk membuat model yang baik?
2. Fitur apa saja yang memiliki korelasi tinggi terhadap target?
3. Model Machine Learning apa yang paling efektif untuk klasifikasi data Breast Cancer ini?

### Goals
1. Mengetahui cara pra-pemrosesan dengan baik agar dapat digunakan dalam pembuatan model.
2. Mengetahui fitur apa saja yang memiliki korelasi mendekati 1 sehingga berpengaruh terhadap target.
3. Mengetahui cara membuat model machine learning untuk memprediksi penyakit kanker payudara.

### Solution Statements
Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :

Untuk pra-pemrosesan data dapat dilakukan beberapa teknik, diantaranya :
- Melakukan drop kolom pada kolom ID.
- Melakukan Encoding terhadap kolom yang bertipe object.
- Melakukan pembagian dataset menjadi dua bagian dengan rasio 80% untuk data latih dan 20% untuk data uji.
- Melakukan Standard Scaler.

- Melakukan korelasi untuk mengetahui fitur mana saja yang nilai korelasinya mendekati 1 terhadap target yang ada pada tahap EDA

- Untuk pembuatan model dipilih penggunaan model dengan algoritma SVM dan Logistic Regression. Algoritma tersebut dipilih karena mudah digunakan dan juga cocok untuk kasus ini. 


## Data Understanding
## Dataset
| Jenis | Keterangan |
| ------ | ------ |
| Sumber | [Kaggle : Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)|
| Lisensi | CC0 : Public Domain|
| Kategori | Kesehatan|
| Jumlah data | 569 data|


### Variabel-variabel pada Breast Cancer dataset adalah sebagai berikut:
- **id** merupakan parameter bernilai unique. Parameter ini tidak penting untuk dimasukkan kedalam model, oleh karena itu parameter ini di drop.
- **Diagnosis** Klasifikasi tumor, dengan nilai 'M' untuk Malignant (ganas) dan 'B' untuk Benign (jinak).
#### Karakteristik Sel:
Setiap karakteristik dihitung dalam tiga kategori: Mean (rata-rata), Standard Error (SE), dan Worst (nilai terburuk). Total ada 10 karakteristik inti, masing-masing dengan 3 jenis nilai, sehingga mencakup 30 fitur numerik.
- **Radius**: Rata-rata jarak dari pusat ke titik pada perimeter sel.
- **Texture**: Standar deviasi intensitas skala abu-abu pada inti sel.
- **Perimeter** :Keliling inti sel, mencerminkan ukuran keseluruhan.
- **Area** : Luas inti sel, dihitung dari perimeter dan radius.
- **Smoothness**: Variasi lokal dalam panjang radius sel, mencerminkan ketidakrataan.
- **Compactness**: Dihitung sebagai (perimeter² / area - 1.0), menggambarkan kepadatan bentuk inti sel.
- **Concavity**: Keparahan cekungan pada kontur sel.
- **Concave Points**: Jumlah titik cekung pada kontur sel.
- **Symmetry**: Tingkat simetri inti sel, menggambarkan perbedaan dari bentuk simetris sempurna.
- **Fractal Dimension**: Mengukur kompleksitas tepi kontur dengan menggunakan dimensi fraktal (1     dimensi fraktal).

**Tiga Kategori Penghitungan:**
- **Mean**: Nilai rata-rata dari fitur tertentu untuk setiap sampel.
- **Standard Error (SE)**: Kesalahan standar dari pengukuran, menunjukkan variasi antar-sampel.
- **Worst**: Nilai maksimum yang diamati untuk fitur tertentu.
**Total Fitur:**
30 Fitur Numerik (10 karakteristik × 3 kategori): Menggambarkan rincian struktur dan tekstur inti sel.
1 Fitur Diagnostik: Menentukan klasifikasi kanker.
1 Fitur Identitas: Untuk penandaan data.

#### Explortory Data Analysis
- Tidak ada data yang kosong/NaN/Duplikat



## Data Preparation
 1. Drop kolom **id** karena mempunyai karakteristik unik dan memiliki nilai korelasi yang kecil dengan target sehingga dapat dibuang.
 2. Melihat apakah ada data NaN dan Null. Setelah melihat dataset ini bersih dari data baik NaN ataupun Null.
 3. Pada fitur "diagnosis", Mengubah M menjadi angka 1 dan B menjadi angka 0 agar mempermudah encoding.
 4. Melakukan Label Encoder dan Melihat korelasi antara target dan fitur-fitur lainnya.
5. Melakukan pembagian dataset menjadi dengan 80% untuk data latih dan 20% untuk data uji Setelah melakukan pra-pemrosesan ke dataset, selanjutnya adalah membagi dataset untuk data latih dan data uji dengan rasio 80:20. Data latih adalah data yang hanya untuk melatih model, sedangkan data uji adalah data yang hanya sebagai ujicoba model. Pembagian dataset ini menggunakan modul train_test_split dari scikit-learn.
6. Melakukan standardisasi data pada semua fitur data. Tahap terakhir yaitu melakukan standarisasi data. Hal ini dilakukan untuk membuat semua fitur berada dalam skala data yang sama yaitu dengan range 0-1.

**Heatmap korelasi**
![image](https://github.com/user-attachments/assets/f3bd9c04-43cc-4fdc-af90-0b025514d12c)


## Modeling
Berikut cara kerja, kelebihan dan kekurangan algoritma SVM dan Logistic Regression:

### Support Vector Machine (SVM)
Cara Kerja:
Menggunakan kernel trick (seperti RBF, polynomial, sigmoid) untuk memetakan data ke dimensi yang lebih tinggi, sehingga menjadi lebih mudah dipisahkan.
Optimasi dilakukan untuk memaksimalkan margin antara dua kelas sambil meminimalkan kesalahan klasifikasi.

**Kelebihan SVM**:
- Efisien pada Dimensi Tinggi: Cocok untuk dataset dengan banyak fitur.
- Robust terhadap Overfitting: Karena memaksimalkan margin, SVM cenderung generalisasi dengan baik.
- Fleksibel: Kernel trick memungkinkan menangani data non-linear.
- Akurasi Tinggi: Sangat baik untuk masalah klasifikasi kompleks.

**Kekurangan SVM**:
- Waktu Komputasi Tinggi: Kurang efisien untuk dataset besar.
- Pemilihan Hyperparameter Sulit: Kernel, C (regularisasi), dan gamma membutuhkan tuning.
- Kurang Interpretatif: Hasil model lebih sulit dipahami.
- Tidak Memberikan Probabilitas Langsung: Probabilitas kelas memerlukan estimasi tambahan.

### Logistic Regression (LR)
Cara Kerja:
Menentukan threshold (biasanya 0.5) untuk memutuskan kelas.

**Kelebihan Logistic Regression:**
- Sederhana dan Cepat: Cocok untuk dataset besar dan menghasilkan hasil dalam waktu singkat.
- Interpretatif: Koefisien fitur memberikan wawasan tentang hubungan dengan variabel target.
- Probabilitas Kelas: Memberikan probabilitas prediksi, mempermudah pengambilan keputusan.
- Tahan terhadap Noise: Model cukup stabil terhadap data yang tidak terlalu kompleks.

**Kekurangan Logistic Regression:**
- Linear Assumption: Tidak cocok untuk data yang tidak dapat dipisahkan secara linear tanpa transformasi.
- Rentan terhadap Multikolinearitas: Korelasi tinggi antar-fitur dapat memengaruhi performa.
- Kurang Baik pada Dimensi Tinggi: Tidak seefisien SVM untuk data dengan banyak fitur.
- Tidak Optimal pada Data Non-linear: Harus diubah dengan fitur tambahan atau teknik polinomial.

## Evaluation
Setelah dilakukan pra-pemrosesan pada dataset, langkah selanjutnya adalah modeling terhadap data. Pada tahap ini menggunakan 2 algoritma yaitu SVM dan Logistik Regression dengan tanpa parameter tambahan. Pertama-tama kedua model ini dilatih menggunakan data latih. Setelah itu kedua model akan diuji dengan data uji. Terakhir kedua model akan diukur nilai akurasinya. Perbandingan Hasil dari kedua model adalah berikut : 
![image](https://github.com/user-attachments/assets/0c6f4fec-1466-4a74-add1-4f1a7a1958a4)


### Support Vector Machine
![image](https://github.com/user-attachments/assets/4985b540-2401-44d9-bfba-f4f5419ddd19)


### Logistic Regression
![image](https://github.com/user-attachments/assets/56b93750-38f2-4300-a83b-65b86e105b4b)


**Akurasi** merupakan metrik untuk menghitung persentase dari total data yang diidentifikasi dan dinilai benar. Rumus akurasi sebagai berikut: 
![Alt text](https://tse3.mm.bing.net/th?id=OIP.QfdpFcS8HzN-LleoF0LxMgAAAA&pid=Api&P=0&h=180)

dimana :
True Positive (TP) : Kasus dimana model memprediksi nilai 0 dan jawaban yang benar juga nilai 0.
True Negative (TN): Kasus dimana model memprediksi nilai 0 tetapi jawaban yang benar adalah nilai 1.
False Positive (FP) : Kasus dimana model memprediksi nilai 1 dan jawaban yang benar juga nilai 1.
False Negative (FN): Kasus dimana model memprediksi nilai 1 tetapi jawaban yang benar adalah nilai 0.

**Precision** merupakan metrik untuk memprediksi benar positif dari keseluruhan hasil yang diprediksi positf. Rumus precision sebagai berikut: 
![Alt text](https://tse3.mm.bing.net/th?id=OIP.h99V_t8My9Nwk5c0uYgpKQHaB8&pid=Api&P=0&h=180)

**Recall** merupakan metrik untuk memprediksi benar positif dibandingkan dengan keseluruhan data yang benar positif. Rumus precision sebagai berikut: 
![Alt text](https://tse2.mm.bing.net/th?id=OIP.vCpYnNVQTcOX7Qp6QJuUgAHaCk&pid=Api&P=0&h=180)

**f1-score** merupakan metrik untuk perbandingan rata-rata precision dan recall yang dibobotkan. Rumus f1-score sebagai berikut: 
![Alt text](https://tse2.mm.bing.net/th?id=OIP.A0Lu2dZfWsCMqWlhw1ZNfQHaB3&pid=Api&P=0&h=180)


**---Terima kasih---**

