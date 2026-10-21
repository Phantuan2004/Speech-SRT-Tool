# AGENTS.md — Speech SRT Tool Guidelines & Architecture Rules

Tài liệu này xác định các nguyên tắc kiến trúc, quy chuẩn mã nguồn, quy trình phát triển theo từng milestone và các quy tắc kỹ thuật nghiêm ngặt dành cho AI Agents và kỹ sư phần mềm phát triển dự án **Speech SRT Tool**.

---

## 1. Mục tiêu và Phạm vi Dự án

- **Tên dự án**: Speech SRT Tool.
- **Mục tiêu**: Desktop application chạy trên Windows, nhận diện giọng nói tiếng Anh (English Speech Recognition) từ video (5–10 phút), xuất file phụ đề `.srt` tương thích chuẩn với **CapCut**.
- **Độ chính xác timestamp**: Sai lệch tuyệt đối trong khoảng **±200 ms trên ít nhất 95%** các mốc được đánh giá trong bộ kiểm thử chuẩn.
- **Phương châm cốt lõi**:
  - **Offline First**: Không phụ thuộc vào Internet sau khi đã tải model, tuyệt đối không sử dụng cloud API có phí.
  - **Độ tin cậy cao**: Tự động nhận diện phần cứng, không bao giờ giả định máy có CUDA. Chạy mượt mà trên CPU với int8 quantization hoặc NVIDIA GPU nếu thực sự hỗ trợ.

---

## 2. Quy tắc Kỹ thuật Nghiêm ngặt (Non-Negotiable Rules)

1. **Quản lý đường dẫn bằng `pathlib.Path`**:
   - Mọi thao tác đường dẫn file/thư mục bắt buộc dùng `pathlib.Path`.
   - Tuyệt đối không hard-code đường dẫn máy cá nhân (ví dụ: `C:\Users\username\...`). Mọi đường dẫn phải dựa trên `Path(__file__)`, thư mục làm việc, hoặc cấu hình từ user.
2. **Không sử dụng `shell=True`**:
   - Khi gọi external tools (FFmpeg, FFprobe), luôn truyền command dưới dạng danh sách token: `['ffmpeg', '-i', str(input_path), ...]`.
   - Luôn kiểm tra returncode và xử lý ngoại lệ `subprocess.CalledProcessError` hoặc `FileNotFoundError`.
3. **Nguyên tắc Timestamp & Tương thích CapCut**:
   - Định dạng timestamp SRT bắt buộc: `HH:MM:SS,mmm --> HH:MM:SS,mmm` (dấu phẩy ngăn cách mili-giây, đúng 3 chữ số mili-giây).
   - Tuyệt đối **không làm tròn thô** hoặc cắt xén timestamp tùy tiện làm trôi lệch đồng bộ (drift).
   - Subtitle block phải đảm bảo: `start_time < end_time`, không có khoảng thời gian âm, không chồng lấn lộn xộn.
4. **Không tải model khi import module**:
   - Các module ASR / aligner phải sử dụng mô hình lazy-loading (chỉ load trọng số model khi gọi phương thức khởi tạo hoặc xử lý cụ thể).
   - Tuyệt đối không load model ở cấp module-level gây đơ máy hoặc tiêu tốn tài nguyên lúc khởi động.
5. **Xác minh phần cứng & Hardware Fallback**:
   - Có GPU không đồng nghĩa với việc CUDA hoạt động. Ví dụ: Card đồ họa AMD Radeon (như RX 580) không hỗ trợ CUDA.
   - Luôn kiểm tra tính khả dụng của CUDA trước khi cấu hình device:
     ```python
     # Luôn fallback an toàn về CPU
     device = "cuda" if is_cuda_actually_available() else "cpu"
     ```
   - Khi chạy trên CPU, ưu tiên `compute_type="int8"` để tối ưu tốc độ và bộ nhớ.
6. **Bảo mật và Quản lý Git**:
   - Tuyệt đối không commit file model weights (hàng trăm MB/GB), file video test dung lượng lớn, hoặc file logs/output vào git.
   - Thư mục `output/` và `logs/` được giữ bằng file `.gitkeep`, nội dung bên trong được ignore.
7. **Phân tách trách nhiệm (Separation of Concerns)**:
   - `app.py` chỉ là file entry-point khởi chạy UI hoặc CLI dispatcher, không chứa bất kỳ logic nghiệp vụ nào.
   - Logic trích xuất audio, nhận diện, căn chỉnh timestamp, xuất SRT nằm riêng biệt trong từng module của `src/speech_srt/`.

---

## 3. Kiến trúc Thư mục và Trách nhiệm Module

```text
speech-srt-tool/
├── app.py                         # Entrypoint duy nhất khởi chạy ứng dụng
├── requirements.txt               # Dependencies phiên bản hóa
├── README.md                      # Hướng dẫn người dùng & cài đặt
├── .gitignore                     # Git ignore rules
├── AGENTS.md                      # Tài liệu quy chuẩn kiến trúc (file này)
├── config/
│   └── default_settings.json      # Cấu hình mặc định (JSON)
├── src/
│   └── speech_srt/
│       ├── __init__.py            # Khai báo package và version
│       ├── config.py              # Đọc/ghi cấu hình với kiểu dữ liệu rõ ràng (dataclass)
│       ├── media_info.py          # Wrapper FFprobe lấy metadata video/audio
│       ├── audio_extractor.py     # Wrapper FFmpeg trích xuất 16kHz mono WAV
│       ├── asr_engine.py          # Interface & implementation Speech-to-Text (offline)
│       ├── word_aligner.py        # Căn chỉnh timestamp cấp độ từ (word-level timing)
│       ├── sentence_segmenter.py  # Cắt câu tự nhiên theo giới hạn ký tự & dòng CapCut
│       ├── srt_writer.py          # Xuất định dạng SRT chuẩn xác từng mili-giây
│       ├── validators.py          # Kiểm tra tính hợp lệ của input/output/timestamp
│       ├── pipeline.py            # Pipeline điều phối tuần tự toàn bộ quy trình
│       └── ui/                    # Giao diện PySide6
│           ├── __init__.py
│           ├── main_window.py     # Cửa sổ chính hiện đại
│           ├── settings_dialog.py # Hộp thoại tùy chỉnh cấu hình
│           ├── subtitle_table.py  # Bảng hiển thị và chỉnh sửa trực quan phụ đề
│           └── workers.py         # QThread worker xử lý ngầm, không block GUI
├── tests/
│   ├── fixtures/                  # Dữ liệu test mẫu
│   ├── test_srt_writer.py         # Kiểm thử format SRT & mili-giây
│   ├── test_validators.py         # Kiểm thử validation logic
│   ├── test_sentence_segmenter.py # Kiểm thử ngắt câu
│   └── test_pipeline.py           # Kiểm thử pipeline integration
├── scripts/
│   └── check_environment.py       # Script chẩn đoán môi trường toàn diện
├── output/                        # Thư mục xuất file .srt mặc định
└── logs/                          # Thư mục ghi log vận hành
```

---

## 4. Coding Conventions & Code Style

- **Python Version**: Python 3.10 trở lên (Môi trường hiện tại: Python 3.13 64-bit trên Windows).
- **Type Annotations**: Tất cả hàm và phương thức công khai phải có đầy đủ type hints (`Path`, `Optional`, `List`, `Dict`, `Tuple`, v.v.).
- **Docstrings**: Sử dụng Google hoặc Sphinx style docstring tóm tắt mục đích, tham số (`Args:`) và giá trị trả về (`Returns:`).
- **Xử lý ngoại lệ**:
  - Bắt lỗi cụ thể, không bao giờ dùng `except: pass` im lặng.
  - Thông báo lỗi phải mang tính định hướng hành động (Actionable error messages) giúp người dùng biết cần sửa gì (ví dụ: "FFmpeg không tìm thấy trong PATH. Hãy tải FFmpeg và thêm vào biến môi trường PATH").
- **Logging**:
  - Sử dụng module chuẩn `logging`.
  - Log ra console và ghi file xoay vòng trong `logs/app.log`.

---

## 5. Quy trình Làm việc theo Từng Milestone

| Milestone | Trọng tâm công việc | Tiêu chí hoàn thành (DoD) |
| :--- | :--- | :--- |
| **Milestone 1** (Hiện tại) | **Project Foundation & Environment Validation** | Tạo cấu trúc, venv, check script, tests cơ sở, config, tài liệu. |
| **Milestone 2** | **Media Inspection & Audio Extraction** | `media_info.py`, `audio_extractor.py`, trích xuất 16kHz mono WAV an toàn. |
| **Milestone 3** | **Offline ASR Engine & Word Alignment** | `asr_engine.py`, tích hợp engine nhẹ/nhanh, word-level timestamps. |
| **Milestone 4** | **Sentence Segmentation & CapCut SRT Writer** | Ngắt câu thông minh, format SRT chuẩn mili-giây, độ trôi < 200ms. |
| **Milestone 5** | **PySide6 Desktop GUI & Worker Threads** | Giao diện kéo thả video, preview phụ đề, progress bar qua QThread. |
| **Milestone 6** | **Quality Benchmark, Error Handling & Packaging** | Đạt test benchmark 95% mốc ±200ms, đóng gói desktop tool. |

---

## 6. Lệnh Vận Hành và Kiểm Thử (Windows PowerShell)

- **Kích hoạt môi trường ảo**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```

- **Chạy script kiểm tra môi trường**:
  ```powershell
  python scripts/check_environment.py
  ```

- **Chạy toàn bộ unit tests**:
  ```powershell
  pytest -v
  ```

- **Chạy file entrypoint**:
  ```powershell
  python app.py
  ```
