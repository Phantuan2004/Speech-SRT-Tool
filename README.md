# Speech SRT Tool

Desktop application chuyên nhận diện lời thoại tiếng Anh (English Speech Recognition) từ video (5–10 phút) và xuất file phụ đề `.srt` chuẩn xác, tương thích hoàn toàn với **CapCut**.

---

## 1. Điểm nổi bật & Mục tiêu Kỹ thuật

- **Hệ điều hành mục tiêu**: Windows 10 / 11 (64-bit).
- **Giao diện Desktop**: PySide6 (Qt for Python).
- **Offline First**: Nhận diện giọng nói hoàn toàn offline, không yêu cầu API key hay dịch vụ đám mây trả phí.
- **Tương thích CapCut**: Xuất file `.srt` chuẩn cú pháp với dấu phẩy ngăn cách mili-giây: `HH:MM:SS,mmm --> HH:MM:SS,mmm`.
- **Độ chính xác Timestamp**: Đạt sai lệch trong khoảng **±200 ms trên ít nhất 95%** các mốc được đánh giá trong bộ kiểm thử chuẩn.
- **Hỗ trợ phần cứng thông minh**: Tự động nhận diện phần cứng, fallback an toàn sang CPU (sử dụng int8 quantization) khi không có NVIDIA GPU hỗ trợ CUDA.

---

## 2. Yêu cầu Hệ thống

1. **Hệ điều hành**: Windows 10 hoặc Windows 11 (64-bit).
2. **Python**: Python 3.10 trở lên (khuyên dùng Python 3.11 - 3.13 64-bit).
3. **FFmpeg & FFprobe**:
   - Đã được cài đặt và thêm vào biến môi trường `PATH` hệ thống.
   - Kiểm tra bằng lệnh:
     ```powershell
     ffmpeg -version
     ffprobe -version
     ```
   - *Nếu chưa có FFmpeg*: Tải bản build đầy đủ tại [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) hoặc cài qua `winget install Gyan.FFmpeg`.

---

## 3. Hướng dẫn Cài đặt & Kích hoạt Môi trường trên Windows

### Bước 1: Mở PowerShell tại thư mục dự án

```powershell
cd "c:\Users\Tuan Izi\OneDrive\Máy tính\Code\Speech SRT Tool"
```

### Bước 2: Thiết lập môi trường ảo (`.venv`)

Tạo môi trường ảo cách ly:
```powershell
python -m venv .venv
```

Kích hoạt môi trường ảo:
- Trên **PowerShell**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(Nếu gặp lỗi PowerShell Execution Policy, mở terminal với quyền Administrator và chạy `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*.

- Hoặc trên **Command Prompt (CMD)**:
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

### Bước 3: Cài đặt dependencies (Milestone 1)

Nâng cấp pip và cài đặt các gói cần thiết:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Kiểm tra & Xác thực Môi trường

Chạy script chẩn đoán tự động toàn diện:
```powershell
python scripts/check_environment.py
```

Script sẽ kiểm tra và hiển thị báo cáo chi tiết:
- Phiên bản Python và trạng thái virtual environment.
- Khả năng gọi FFmpeg và FFprobe từ Python subprocess (không dùng `shell=True`).
- Tình trạng Git version control.
- Phân tích phần cứng (NVIDIA GPU / AMD GPU / CPU) và kết luận thiết bị chạy tối ưu.
- Kiểm tra import các package đã cài đặt (`PySide6`, `pytest`).
- Quyền ghi trên các thư mục `output/`, `logs/`, `config/`.

---

## 5. Chạy Kiểm Thử Tự Động (Unit Tests)

Dự án đi kèm bộ kiểm thử đơn vị bao phủ format SRT chuẩn CapCut, kiểm tra timestamp, ngắt câu và xác thực đầu vào:

```powershell
pytest -v
```

Kết quả: **29/29 tests passed** (100% PASS).

---

## 6. Chạy Thử Ứng Dụng (Entrypoint)

```powershell
python app.py
```

---

## 7. Hướng dẫn Sử dụng CLI Pipeline (Speech Recognition & SRT Export)

Công cụ hỗ trợ giao diện dòng lệnh (CLI) tốc độ cao, cho phép nhận diện và xuất phụ đề tự động:

### Lệnh chạy cơ bản:
```powershell
python -m speech_srt.cli --input "duong_dan_video.mp4" --output "output/subtitles.srt" --model small.en --device auto
```

### Các tùy chọn tham số:
| Tham số | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `-i, --input` | *(Bắt buộc)* | Đường dẫn file video hoặc audio nguồn (`.mp4`, `.mkv`, `.mov`, `.wav`, v.v.) |
| `-o, --output` | `output/<tên>.srt` | Đường dẫn file phụ đề `.srt` xuất ra |
| `-m, --model` | `small.en` | Kích thước mô hình Whisper (`tiny.en`, `base.en`, `small.en`, `medium.en`) |
| `-d, --device` | `auto` | Thiết bị xử lý: `auto` (ưu tiên CUDA nếu có, fallback an toàn sang `cpu`), `cpu`, hoặc `cuda` |
| `--compute-type` | `auto` | Định dạng lượng tử hóa (`int8` cho CPU, `float16` cho CUDA) |
| `--beam-size` | `5` | Độ rộng chùm giải mã (Beam search width) |
| `--vad / --no-vad` | `True` | Bật/tắt bộ lọc giọng nói Silero VAD để lọc khoảng lặng |
| `--offline` | `False` | Bật chế độ offline nghiêm ngặt (chỉ dùng model đã có trong cache) |
| `--max-line-length` | `42` | Số ký tự tối đa trên mỗi dòng phụ đề (chuẩn hiển thị CapCut) |

---

## 8. Cấu trúc Thư mục Dự án

```text
Speech SRT Tool/
├── app.py                         # Application entrypoint bootstrap
├── pyproject.toml                 # Package metadata & pytest configuration
├── requirements.txt               # Project dependencies (PySide6, faster-whisper, av<14, pytest)
├── README.md                      # Hướng dẫn cài đặt & vận hành
├── .gitignore                     # Git ignore rules cho Python, media & models
├── AGENTS.md                      # Nguyên tắc kiến trúc & quy chuẩn kỹ thuật
├── config/
│   └── default_settings.json      # Cấu hình mặc định (JSON)
├── src/
│   └── speech_srt/
│       ├── __init__.py            # Package root & metadata (v0.1.0)
│       ├── cli.py                 # CLI dispatcher chạy pipeline nhận diện & xuất SRT
│       ├── config.py              # Typed configuration loader & validator
│       ├── media_info.py          # FFprobe wrapper lấy metadata video/audio
│       ├── audio_extractor.py     # FFmpeg wrapper trích xuất 16kHz mono WAV an toàn
│       ├── asr_engine.py          # Faster-Whisper offline ASR engine (lazy loading)
│       ├── word_aligner.py        # Căn chỉnh timestamp cấp độ từ
│       ├── sentence_segmenter.py  # Cắt câu tự nhiên theo dấu câu, khoảng lặng & CapCut
│       ├── srt_writer.py          # Bộ ghi SRT nguyên tử (atomic) & parser kiểm tra
│       ├── validators.py          # Bộ kiểm tra tính hợp lệ media & subtitle cues
│       ├── pipeline.py            # Pipeline điều phối xử lý toàn diện không phụ thuộc GUI
│       └── ui/                    # Giao diện PySide6
│           ├── __init__.py
│           ├── main_window.py     # Cửa sổ chính
│           ├── settings_dialog.py # Hộp thoại cài đặt
│           ├── subtitle_table.py  # Bảng hiển thị & chỉnh sửa phụ đề
│           └── workers.py         # QThread workers xử lý ngầm không block GUI
├── tests/
│   ├── fixtures/                  # Test sample files
│   ├── test_media_info.py         # Kiểm thử FFprobe media metadata & bắt lỗi no-audio
│   ├── test_audio_extractor.py    # Kiểm thử trích xuất 16kHz WAV & dọn dẹp file tạm
│   ├── test_asr_engine.py         # Kiểm thử ASR lazy loading, device resolution & offline
│   ├── test_sentence_segmenter.py # Kiểm thử ngắt câu theo dấu câu, ngắt dòng & fallback
│   ├── test_srt_writer.py         # Kiểm thử format SRT, parse_srt roundtrip & atomic write
│   ├── test_validators.py         # Kiểm thử validation input, bounds, overlap & duration
│   └── test_pipeline.py           # Kiểm thử pipeline integration
├── scripts/
│   └── check_environment.py       # Script kiểm tra chẩn đoán môi trường
├── output/                        # Thư mục xuất file SRT mặc định
└── logs/                          # Thư mục ghi logs ứng dụng
```

---

## 9. Lộ trình Triển khai (Milestones)

- [x] **Milestone 1**: Project Foundation & Environment Validation *(Hoàn thành)*.
- [x] **Milestone 2**: Media Inspection & Audio Extraction (FFmpeg/FFprobe) *(Hoàn thành)*.
- [x] **Milestone 3**: Offline ASR Engine & Word Alignment (Faster-Whisper) *(Hoàn thành)*.
- [x] **Milestone 4**: Sentence Segmentation & CapCut SRT Generator *(Hoàn thành)*.
- [ ] **Milestone 5**: PySide6 Desktop GUI & Worker Threads *(Kế tiếp)*.
- [ ] **Milestone 6**: Quality Benchmark (±200 ms trên 95% mốc), Error Handling & Windows Packaging.
