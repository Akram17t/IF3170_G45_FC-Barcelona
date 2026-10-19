# Penataan Paket dalam Truk

Tugas Besar 1 IF3170 Inteligensi Artifisial, kelompok 45. Program mencari susunan paket yang memenuhi batasan truk dan memaksimalkan total nilai paket yang diangkut. Algoritma yang tersedia adalah Hill Climbing, Simulated Annealing, dan Genetic Algorithm.

## Menjalankan program

Gunakan Python 3.10 atau lebih baru. Jalankan perintah berikut dari folder utama proyek.

```bash
python -m pip install -r requirements.txt
python -m src.web
```

Program membuka antarmuka di `http://127.0.0.1:8000`. Untuk memakai port lain, jalankan `python -m src.web --port 8001`. Grafik pada halaman web memerlukan koneksi internet untuk memuat Plotly.

Pilih contoh kasus atau unggah file JSON, kemudian pilih algoritma dan parameternya. Hasil yang disimpan melalui pilihan simpan hasil akan ditulis ke folder `output/`.

Hill Climbing juga dapat dijalankan melalui terminal, misalnya:

```bash
python -m src.algorithms.hillclimbing --input examples/tiny.json --variant sideways --max-sideways-moves 20 --seed 17
```

Pada varian sideways, batas tersebut menghitung total gerakan dengan nilai yang sama selama satu pencarian. Hitungannya tetap bertambah meskipun diselingi kenaikan nilai.

## Format input kasus

File kasus berisi objek JSON dengan dua field wajib, yaitu `truck` dan `packages`. Nama field membedakan huruf besar dan kecil. Field tambahan yang tidak dikenal ditolak oleh pembaca input Python.

Berikut contoh kasus lengkap dengan satu paket:

```json
{
  "truck": {
    "dimensions": {"w": 3, "l": 2, "h": 2},
    "maxCapacity": 5
  },
  "packages": [
    {
      "id": "A",
      "dimensions": {"w": 2, "l": 1, "h": 1},
      "orientation": "xyz",
      "value": 10,
      "weight": 2,
      "isFragile": false,
      "ETA": 1
    }
  ]
}
```

Contoh dengan beberapa paket tersedia di [examples/tiny.json](examples/tiny.json).

### Data truk

| Field | Isi dan syarat |
|---|---|
| `dimensions` | Objek dengan tepat tiga field `w`, `l`, dan `h`. Semuanya bilangan bulat positif |
| `maxCapacity` | Batas total berat paket yang dimuat. Angka hingga yang tidak negatif, boleh desimal |

Satu truk memiliki satu ruang muat berbentuk balok. Sumbu `x`, `y`, dan `z` masing-masing menyatakan lebar, panjang, dan tinggi ruang muat.

### Data paket

`packages` harus berupa array. Array kosong diperbolehkan. Setiap paket wajib memiliki semua field berikut.

| Field | Isi dan syarat |
|---|---|
| `id` | String yang berisi setidaknya satu karakter selain spasi. Harus unik dalam satu kasus |
| `dimensions` | Objek dengan tepat tiga field `w`, `l`, dan `h`. Semuanya bilangan bulat positif |
| `orientation` | Salah satu dari enam orientasi yang dijelaskan di bawah |
| `value` | Nilai paket. Angka hingga yang tidak negatif, boleh desimal |
| `weight` | Berat paket. Angka hingga yang tidak negatif, boleh desimal |
| `isFragile` | Boolean JSON, yaitu `true` atau `false` |
| `ETA` | Bilangan bulat yang tidak negatif |

Gunakan angka JSON untuk field numerik, bukan string atau boolean. Nilai `NaN` dan tak hingga tidak diterima. Kapasitas, nilai, berat, dan ETA boleh bernilai nol. Tidak ada batas jumlah paket yang ditetapkan program, tetapi kasus yang besar akan membutuhkan waktu pencarian lebih lama.

Paket yang terlalu besar atau terlalu berat untuk dimuat tetap boleh tercantum dalam input. Paket tersebut dapat berada di luar truk dan tidak menambah nilai maupun berat muatan.

### Orientasi

Orientasi memetakan sisi asli paket `w`, `l`, dan `h` ke sumbu truk. Bentuk string yang diterima adalah `xyz`, `xzy`, `yxz`, `yzx`, `zxy`, dan `zyx`. Urutan huruf selalu mengikuti sisi asli `w`, `l`, lalu `h`.

Program juga menerima bentuk objek. Dua contoh berikut memiliki arti yang sama:

```json
"orientation": "yxz"
```

```json
"orientation": {"w": "y", "l": "x", "h": "z"}
```

Artinya, sisi `w` diarahkan ke sumbu `y`, sisi `l` ke sumbu `x`, dan sisi `h` ke sumbu `z`. Jika dimensi asli paket adalah `w = 3`, `l = 4`, dan `h = 1`, ukurannya pada sumbu truk menjadi `x = 4`, `y = 3`, dan `z = 1`.

Pada bentuk objek, field harus tepat `w`, `l`, dan `h`. Setiap sumbu harus digunakan satu kali. Pemetaan berulang seperti `{"w":"x","l":"x","h":"z"}` tidak valid. Web mengubah orientasi yang diunggah menjadi bentuk string agar pilihan orientasi dan pratinjau tetap konsisten. Ekspor dari Python juga memakai bentuk string.

Orientasi pada input menjadi orientasi awal paket. Selama pencarian, algoritma dapat mengubahnya melalui rotasi.

## Representasi state dan syarat kelayakan

File kasus hanya memuat data truk dan paket. Posisi awal paket dibangkitkan oleh program. State hasil pencarian mencatat semua ID paket, termasuk paket yang berada di luar truk.

Setiap penempatan memiliki `position` dan `orientation`. Posisi berupa array tiga bilangan bulat `[x, y, z]` yang menunjukkan sudut bawah paket. `position: null` berarti paket berada di luar truk. Orientasi tetap dicatat untuk paket yang berada di luar.

Contoh state untuk kasus satu paket di atas:

```json
{
  "placements": {
    "A": {"position": [0, 0, 0], "orientation": "xyz"}
  }
}
```

State merupakan bagian dari hasil pencarian, bukan file kasus yang diunggah melalui tombol unggah kasus. File hasil lengkap juga memuat konfigurasi, riwayat nilai, dan metrik pencarian.

State dinyatakan layak jika memenuhi semua syarat berikut:

- Semua ID paket dalam kasus tercatat tepat sekali, tanpa ID tambahan.
- Seluruh bagian paket yang dimuat berada di dalam batas ruang muat.
- Total berat paket di dalam truk tidak melebihi `maxCapacity`.
- Paket tidak saling tumpang tindih. Permukaan yang bersentuhan diperbolehkan.
- Paket berada di lantai atau memiliki penopang tepat di bawah sebagian alasnya. Dukungan tidak harus menutupi seluruh alas, tetapi kontak pada tepi saja tidak cukup.
- Paket rapuh boleh dimuat dan ditopang, tetapi tidak boleh menjadi penopang paket lain.

Validitas data input berbeda dari kelayakan suatu susunan. Contohnya, paket yang dimensinya melebihi ruang muat tetap merupakan data paket yang valid, tetapi state yang menempatkannya menembus dinding akan ditolak.

## Fungsi objektif dan penggunaan ETA

Fungsi objektif menjumlahkan `value` seluruh paket yang berada di dalam truk. Paket di luar truk tidak dihitung. Batasan susunan diperiksa oleh `src/validation.py` sebelum nilainya digunakan.

`ETA` tetap wajib dibaca, divalidasi, dan disimpan bersama data paket. ETA tidak dipakai untuk memilih langkah pencarian atau menghitung nilai karena tujuan dasar tugas adalah memaksimalkan nilai muatan, sementara spesifikasi tidak menetapkan prioritas, tenggat pengiriman, atau aturan penempatan berdasarkan ETA. Karena itu, program tidak menambahkan bobot atau batasan ETA yang belum didefinisikan dalam persoalan.

## Pengujian

Jalankan pengujian model dari folder utama proyek:

```bash
python -m unittest discover -s tests
```
