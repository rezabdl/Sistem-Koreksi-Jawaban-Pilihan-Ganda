# Sistem-Koreksi-Jawaban-Pilihan-Ganda
Sistem ini adalah sebuah alat koreksi jawaban pilihan ganda yang bisa digunakan pada lambar jawaban yang berjumlah 40 nomor dan diisi dengan silangan

Untuk sistem ini dibangun dengan menggunkana TKinter untuk UI nya, UI ini saya bentuk hanya sekedar untuk model YOLO ini bisa digunakan dengan mudah oleh end-user saja. Model YOLOv11 dipilih sebagai model pendeteksian karena model YOLOv11 terkenal dengan sangat baik dalam mendeteksi objek kecil, karena itu saya terpikirkan untuk menggunakan model YOLOv11 ini.

Tentu, Reza! Berikut ini adalah penjelasan **langkah demi langkah (step-by-step)** penggunaan sistem pengoreksian otomatis berbasis **YOLOv11** yang kamu kembangkan. Penulisan ini cocok kamu tampilkan dalam dokumentasi GitHub, bagian tutorial aplikasi, atau bisa juga disisipkan di caption sebagai “cara penggunaan”:

---

## **Langkah-langkah Menggunakan Sistem Pengoreksian Otomatis Lembar Jawaban Pilihan Ganda (YOLOv11)**

Berikut panduan penggunaan sistem mulai dari memuat model hingga melihat hasil koreksi:

---

### 🔹 **1. Memasukkan Model YOLOv11**

* Pastikan model **YOLOv11** yang telah dilatih tersedia dalam format `.pt` atau sesuai struktur yang digunakan.
* Model akan digunakan untuk mendeteksi bulatan jawaban pada lembar soal pilihan ganda.
* Model ini dimuat otomatis saat aplikasi dijalankan, atau dapat dipilih secara manual di halaman awal.

---

### 🔹 **2. Menentukan Kunci Jawaban**

Tersedia dua metode yang bisa dipilih untuk menentukan kunci jawaban:

#### 🔸 **a. Deteksi Otomatis dari Lembar Kunci Jawaban**

* Unggah gambar lembar kunci jawaban ke sistem.
* Sistem akan melakukan deteksi visual terhadap pilihan yang ditandai pada lembar tersebut menggunakan YOLOv11.
* Hasil deteksi akan otomatis tersimpan sebagai kunci jawaban.

#### 🔸 **b. Input Manual**

* Jika tidak memiliki lembar kunci jawaban dalam bentuk gambar, pengguna dapat mengisi **kunci jawaban secara manual** melalui kolom isian.
* Format pengisian disesuaikan dengan urutan soal (contoh: `A B D C B A`).

---

### 🔹 **3. Koreksi Lembar Jawaban Siswa**

* Klik tombol untuk mengunggah **gambar lembar jawaban siswa**.
* Sistem akan mendeteksi pilihan jawaban yang diisi siswa menggunakan YOLOv11.
* Jawaban siswa akan **dibandingkan otomatis dengan kunci jawaban** yang telah disimpan sebelumnya.

---

### 📊 **Hasil Koreksi yang Ditampilkan**

Setelah proses deteksi dan pencocokan selesai, sistem akan menampilkan:

* ✅ **Jumlah jawaban benar**
* ❌ **Jumlah jawaban salah**
* 📌 **Jumlah jawaban yang berhasil terdeteksi**
* ⏱️ **Waktu koreksi per lembar (dalam detik)**

Setiap hasil juga dapat di-*export* atau disimpan untuk dokumentasi.




