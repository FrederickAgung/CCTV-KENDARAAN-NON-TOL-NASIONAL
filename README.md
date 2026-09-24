# CCTV Traffic Analytics Dashboard

Website yang menggabungkan 2 kamera CCTV (dari `traffic_analytics_smooth_final_fixed2.py`
dan `traffic_monitor_fixed_v5.py`) menjadi satu dashboard yang bisa dibuka lewat browser,
di komputer manapun yang satu jaringan (LAN) — misalnya `http://10.10.91.160:5000/`.

## Struktur file

```
webapp/
├── app.py              -> web server Flask (routing, MJPEG stream, API stats)
├── camera_engine.py     -> mesin deteksi/tracking/hitung kendaraan (headless, per kamera)
├── templates/index.html -> tampilan dashboard (2 video live + panel statistik)
├── requirements.txt
└── data/                 (dibuat otomatis saat pertama dijalankan)
    ├── gunung_pasir/      -> db, csv, snapshot khusus kamera Gunung Pasir
    └── foni_kb_sayur/     -> db, csv, snapshot khusus kamera Foni KB Sayur
```

## Cara menjalankan

1. Install dependencies (idealnya di virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
   Kalau komputer punya GPU NVIDIA, install `torch` versi CUDA dulu sebelum
   `pip install -r requirements.txt` biar deteksi jauh lebih cepat (lihat
   https://pytorch.org/get-started/locally/).

2. Jalankan:
   ```bash
   python app.py
   ```
   Model `yolov8s.pt` akan otomatis terunduh saat pertama kali dijalankan.

3. Buka di browser:
   - Di komputer yang sama: `http://localhost:5000/`
   - Dari HP/laptop lain di jaringan yang sama: `http://<IP-LAN-komputer-ini>:5000/`
     (cek IP dengan `ipconfig` di Windows / `ip addr` di Linux/Mac, contoh: `10.10.91.160`)

   Server sudah di-set `host="0.0.0.0"` di `app.py`, jadi otomatis bisa diakses dari
   perangkat lain — tidak cuma `localhost`.

## Kalau mau dijalankan terus-menerus (hosting beneran)

`python app.py` (Flask dev server) cukup untuk pemakaian internal/LAN, tapi kalau mau
lebih stabil untuk jalan 24/7:

- **Linux**: jalankan lewat `systemd` service, atau pakai `gunicorn` + `nginx` sebagai
  reverse proxy (catatan: MJPEG streaming butuh worker yang tidak mem-buffer respons,
  jadi kalau pakai gunicorn gunakan `--worker-class gthread` dengan beberapa thread).
- **Windows**: jalankan sebagai Windows Service (mis. pakai NSSM - Non-Sucking Service
  Manager) supaya otomatis start saat komputer nyala/restart.
- Pastikan firewall komputer mengizinkan port 5000 (atau port lain yang kamu pilih) untuk
  koneksi masuk dari jaringan lokal.
- Untuk ganti port, ubah baris terakhir `app.py`: `app.run(host="0.0.0.0", port=5000, ...)`.

## Menambah kamera lain

Tinggal tambah satu `CameraConfig(...)` baru di `CAMERA_CONFIGS` (list) dalam `app.py`,
kasih `cam_id` unik, `name`, dan `stream_url` HLS-nya. Dashboard & routing otomatis
menyesuaikan (index.html melakukan loop atas semua kamera).

## Yang tersedia di dashboard

- **Video live** tiap kamera (hasil anotasi kotak deteksi + panel info) via MJPEG stream.
- **Panel statistik** live (update tiap 1.5 detik): total kendaraan unik ter-scan,
  rincian per jenis (Mobil/Motor/Truk/Bus), jumlah di frame saat ini, FPS, status
  kepadatan (LANCAR/PADAT MERAYAP/MACET), dan jam sesi dibuka.
- **Unduh data**: tombol unduh CSV log rinci per kendaraan, CSV ringkasan per 5 menit,
  dan file database SQLite — masing-masing kamera terpisah.

## Perbedaan dengan versi desktop (cv2.imshow) sebelumnya

- Kedua kamera sekarang dijalankan headless di background (bukan window cv2 terpisah),
  jadi bisa jalan di server tanpa monitor/GUI, dan videonya dikirim ke browser lewat
  jaringan.
- Logika "1 kendaraan = 1x hitung, jenis kendaraan dikunci setelah terkonfirmasi" dari
  `fixed2.py` (voting mayoritas + `locked_labels` + tracker `track_buffer` lebih
  panjang) dipakai untuk **kedua** kamera, termasuk kamera Foni KB Sayur yang di file
  `v5.py` sebelumnya belum punya fitur ini.
- **Belum ada**: editor garis-hitung interaktif via klik mouse di browser (di versi
  desktop ini dilakukan dengan tombol `E` + klik). Untuk sekarang, garis hitung
  dikonfigurasi lewat file `zones_config.json` di masing-masing folder `data/<cam_id>/`
  (format sama seperti yang dipakai script desktop — bisa disalin langsung dari sana).
  Editor visual berbasis canvas di browser bisa ditambahkan sebagai langkah berikutnya
  kalau dibutuhkan.
