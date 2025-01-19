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

1. Untuk pra-pemrosesan data dapat dilakukan beberapa teknik, diantaranya :
- Melakukan drop kolom pada kolom ID.
- Melakukan Encoding terhadap kolom yang bertipe object.
- Melakukan pembagian dataset menjadi dua bagian dengan rasio 80% untuk data latih dan 20% untuk data uji.
- Melakukan Standard Scaler.
2. Melakukan korelasi untuk mengetahui fitur mana saja yang nilai korelasinya mendekati 1 terhadap target yang ada pada tahap EDA
3. Untuk pembuatan model dipilih penggunaan model dengan algoritma SVM dan Logistic Regression. Algoritma tersebut dipilih karena mudah digunakan dan juga cocok untuk kasus ini. 


## Data Understanding
Tahap ini memberikan informasi seperti jumlah data, kondisi data, dan informasi mengenai data yang digunakan, tautan sumber data (link download), dan menguraikan seluruh variabel atau fitur pada data.

### Dataset
| Jenis | Keterangan |
| ------ | ------ |
| Sumber | [Kaggle : Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)|
| Lisensi | CC0 : Public Domain|
| Kategori | Kesehatan|
| Jumlah data | 569 data|

### Explortory Data Analysis
Data berisi 569 data dengan 32 fitur

### Fitur pada Breast Cancer dataset adalah sebagai berikut:
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

- Tidak ada data yang kosong/NaN/Duplikat



**Visualisasi Fitur Diagnosis (Target)**

![image](https://github.com/user-attachments/assets/d656b6d5-b3f6-4257-b835-46e3993f64fb)

Target analisis data ini adalah fitur "diagnosis" yang memiliki 357 data Benign (Jinak) dan 212 data Malignant (Ganas).


**Heatmap korelasi fitur terhadap Target**

![image](https://github.com/user-attachments/assets/144e29df-f941-4321-a85c-6e816e4412bd)


**8 Fitur yang memiiki nilai korelasi mendekati 1 terhadap Target yang artinya mempengaruhi target**

![image](https://github.com/user-attachments/assets/75f08175-ecf5-423e-ac69-8c5ce810f7f0)

Fitur yang memiliki nilai korelasi mendekati 1 terhadap target antara lain :
radius_mean, parameter_mean, area_mean, concave_points_mean, radius_worst, perimeter_worst, area_worst, dan convcave_points_worst.



## Data Preparation
Tahap ini dilakukan untuk mempersiapkan data untuk memasuki tahap modeling. Adapun beberapa langkah yang dilakukan pada data preparation antara lain :
1. Drop kolom "id"
2. Melakukan label encoding untuk mengurangi dimensi data
3. Mendefinisikan kolom X sebagai target dan y sebagai fitur
4. Split data training dan testing 
5. Melakukan standarisari data

   
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

**Confussion Matrix SVM Model**

![image](https://github.com/user-attachments/assets/2b5907e9-b129-4b6f-a249-1d2e2468e257)

Hasil confussion matriks SVM Classifier adalah 68 (True Positive) 2 (False Negative) 3 (False Positive) dan 41 (True Negative)

**Akurasi SVM Model**

![image](https://github.com/user-attachments/assets/ddf63f51-b33d-4c1c-9534-8d01b0b78194)


**Classification Report SVM Model**

![image](https://github.com/user-attachments/assets/13e7ee96-848a-4ab5-852e-ad3856e8d204)




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

**Confussion Matrix Logistic Regression Model**

![image](https://github.com/user-attachments/assets/a56e30b7-1a68-4683-a6e5-f3f781bd2195)

Hasil confussion matriks LogReg Classifier adalah 70 (True Positive) 2 (False Negative) 1 (False Positive) dan 41 (True Negative)

**Akurasi Logistic Regression Model**

![image](https://github.com/user-attachments/assets/32347ff9-2530-4a96-9a10-b8672a06a509)


**Classification Report Logistic Regression Model**

![image](https://github.com/user-attachments/assets/b361b652-23e5-43c1-aea7-435b783bf94e)




## Evaluation
Setelah dilakukan pra-pemrosesan pada dataset, langkah selanjutnya adalah modeling terhadap data. Pada tahap ini menggunakan 2 algoritma yaitu SVM dan Logistik Regression dengan tanpa parameter tambahan. Pertama-tama kedua model ini dilatih menggunakan data latih. Setelah itu kedua model akan diuji dengan data uji. Terakhir kedua model akan diukur nilai akurasinya. Perbandingan Hasil dari kedua model adalah berikut :

![image](https://github.com/user-attachments/assets/7bd2085e-70f2-42c6-a9e6-26e0c2090335)


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

# Kesimpulan
Dengan ini dapat disimpulkan bahwa :    
1. Untuk pra-pemrosesan data dapat dilakukan beberapa teknik, diantaranya :
- Melakukan drop kolom pada kolom ID.
- Melakukan Encoding terhadap kolom yang bertipe object.
- Melakukan pembagian dataset menjadi dua bagian dengan rasio 80% untuk data latih dan 20% untuk data uji.
- Melakukan Standard Scaler.

2. Pada data understanding didapatkan insight dari melakukan EDA pada tiap fitur serta mengetahui fitur mana saja yang memiliki nilai korelasi mendekati 1 terhadap target antara lain :     
radius_mean, parameter_mean, area_mean, concave_points_mean, radius_worst, perimeter_worst, area_worst, dan convcave_points_worst.

3. Setelah dilakukan proses klasifikasi menggunakan model SVM dan Logistic Regression, dapat disimpulkan bahwa model dengan akurasi terbaik adalah Logistic Regression dengan akurasi mencapai 97%

**---Terima kasih---**

