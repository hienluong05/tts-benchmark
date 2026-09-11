# TTS benchmark: FastPitch và OmniVoice

Runner này đo latency end-to-end của Text-to-Speech cho từng câu, sau khi model đã
được load vào bộ nhớ. Kết quả phù hợp để so sánh FastPitch + HiFi-GAN với
OmniVoice trên cùng máy, GPU, precision và workload.

> Quan trọng: bản FastPitch + HiFi-GAN reference và API public OmniVoice hiện
> tại đều trả waveform hoàn chỉnh. Hai adapter trong repo **không fake
> streaming** bằng cách cắt waveform đã sinh xong. Vì vậy `TTFA` của chúng gần
> bằng full inference latency, còn `num_chunks=1` là kết quả đúng.

## Các file

| File | Vai trò |
| --- | --- |
| `benchmark_tts.py` | Runner, đo chỉ số và xuất CSV/JSON. |
| `fastpitch_adapter.py` | Load FastPitch + HiFi-GAN TorchScript từ NVIDIA DeepLearningExamples. |
| `omnivoice_adapter.py` | Load `k2-fsa/OmniVoice` bằng Python API. |

## Metric

| Metric | Định nghĩa |
| --- | --- |
| TTFA | Từ ngay trước `adapter.stream(text)` đến chunk audio playable, non-empty đầu tiên. |
| E2E RTF | `total_time / audio_duration`; nhỏ hơn 1 là nhanh hơn realtime. |
| Post-TTFA RTF | `(end_time - first_audio_time) / audio_duration`. Hữu ích khi model có streaming native. |
| First chunk duration | Thời lượng audio trong chunk đầu. |

`summary.csv` tính RTF theo **ratio of totals** trong mỗi model/câu, thay vì lấy
trung bình RTF từng run. Cách này không làm câu ngắn có trọng số quá lớn.

## Yêu cầu

- Python 3.10+.
- PyTorch phù hợp CUDA driver nếu benchmark GPU.
- `numpy`.
- FastPitch chỉ: source NVIDIA DeepLearningExamples và checkpoint FastPitch,
  HiFi-GAN **TorchScript** tương thích.
- OmniVoice chỉ: package `omnivoice`; lần load đầu sẽ tải model khoảng 3.27 GB.

Tạo môi trường riêng, ví dụ Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy
```

Cài PyTorch theo đúng CUDA từ trang PyTorch. Không dùng một lệnh CUDA cố định
nếu chưa kiểm tra driver/GPU của máy.

## Cài và tải OmniVoice

```powershell
python -m pip install omnivoice "huggingface_hub[cli]"
```

### Tải tự động

Không cần tải thủ công. Dùng `model_id: "k2-fsa/OmniVoice"`; adapter gọi
`OmniVoice.from_pretrained()` và Hugging Face sẽ cache model ở máy.

### Tải sẵn vào thư mục project

```powershell
hf download k2-fsa/OmniVoice --local-dir .\models\OmniVoice
```

Sau đó truyền đường dẫn tuyệt đối Windows làm `model_id`. Repo model chính thức:
<https://huggingface.co/k2-fsa/OmniVoice>. API trả list waveform mono `numpy`
24 kHz; adapter chuẩn hóa về `float32` trước khi yield.

### Voice cloning ổn định

Để benchmark cùng speaker giữa nhiều run, dùng một reference audio và transcript
cố định:

```json
{
  "model_id": "G:\\My Drive\\Documents\\TTS\\benchmark\\models\\OmniVoice",
  "device": "cuda:0",
  "dtype": "float16",
  "ref_audio": "G:\\My Drive\\Documents\\TTS\\benchmark\\reference.wav",
  "ref_text": "Nội dung được đọc trong reference.wav.",
  "generate_options": {"num_step": 32, "speed": 1.0}
}
```

Không để auto voice ngẫu nhiên nếu mục tiêu là so sánh hiệu năng lặp lại. Giữ
cố định `num_step`, `speed`, reference prompt và phiên bản model.

## Cài FastPitch + HiFi-GAN

Clone NVIDIA reference source:

```powershell
git clone https://github.com/NVIDIA/DeepLearningExamples.git .\third_party\DeepLearningExamples
```

Tải checkpoint pretrained từ NVIDIA NGC hoặc dùng checkpoint fine-tune của dự
án. FastPitch và vocoder phải tương thích mel configuration, nhất là sample
rate, hop length, số mel bins và text processing. NVIDIA reference inference:
<https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/inference.py>

### Định dạng checkpoint hiện được hỗ trợ

`fastpitch_adapter.py` hiện dùng `torch.jit.load`, vì vậy yêu cầu:

- `fastpitch_checkpoint`: FastPitch **TorchScript**.
- `hifigan_checkpoint`: HiFi-GAN **TorchScript**.

Checkpoint NGC PyTorch `.pt` thông thường không tự động tương thích với adapter
này. Đừng đổi đuôi file hoặc thử load mù. Nếu checkpoint của bạn là `.pt` eager
PyTorch, cần thêm loader dựa trên `models.load_and_setup_model()` của NVIDIA,
hoặc export đúng hai model sang TorchScript từ cùng version source/checkpoint.

### Tiếng Việt

Mặc định adapter dùng `symbol_set="english_basic"` và
`text_cleaners="english_cleaners"`. Nó không phù hợp để benchmark tiếng Việt có
dấu. Để dùng Vietnamese, cung cấp FastPitch đã fine-tune với tokenizer/text
processing tiếng Việt tương ứng và truyền đúng `symbol_set`, `text_cleaners`,
`p_arpabet` của checkpoint. Không so sánh chất lượng hay latency của một input
mà FastPitch không encode đúng.

## Chuẩn bị tập text

Tạo file UTF-8 `texts.txt`, mỗi dòng một câu. Có thể nhóm câu ngắn/trung/dài
bằng thứ tự dòng và đọc summary theo `text_id`.

```text
Xin chào, đây là câu kiểm thử ngắn.
Hệ thống trợ lý giọng nói cần có độ trễ thấp để người dùng có thể hội thoại tự nhiên.
Đây là một đoạn kiểm thử dài hơn nhằm đo độ ổn định của TTFA và real-time factor trên workload gần với sử dụng thực tế.
```

Giữ nguyên text, thứ tự, normalization, punctuation và speed configuration cho
mọi model. Chỉ so sánh model có thể phát âm/encode được cùng workload.

## Chạy benchmark

Ví dụ đầy đủ trên PowerShell. Mỗi `--adapter-options` đứng ngay sau adapter
nó cấu hình; JSON phải dùng dấu nháy đơn bên ngoài để PowerShell không xử lý
dấu nháy kép bên trong.

```powershell
python .\benchmark_tts.py `
  --adapter fastpitch_adapter:create_adapter `
  --adapter-options '{"repo_dir":"G:\\My Drive\\Documents\\TTS\\benchmark\\third_party\\DeepLearningExamples\\PyTorch\\SpeechSynthesis\\FastPitch","fastpitch_checkpoint":"G:\\models\\fastpitch.ts","hifigan_checkpoint":"G:\\models\\hifigan.ts","device":"cuda","sample_rate":22050,"amp":true}' `
  --adapter omnivoice_adapter:create_adapter `
  --adapter-options '{"model_id":"G:\\My Drive\\Documents\\TTS\\benchmark\\models\\OmniVoice","device":"cuda:0","dtype":"float16","ref_audio":"G:\\My Drive\\Documents\\TTS\\benchmark\\reference.wav","ref_text":"Nội dung được đọc trong reference.wav.","generate_options":{"num_step":32,"speed":1.0}}' `
  --texts .\texts.txt `
  --warmup 3 `
  --repeat 10 `
  --out-dir .\results
```

Nếu chỉ chạy một model, chỉ truyền một `--adapter` và một options object.

## Cấu hình adapter

### FastPitch

| Option | Bắt buộc | Mặc định | Ý nghĩa |
| --- | --- | --- | --- |
| `repo_dir` | Có | — | Thư mục `.../PyTorch/SpeechSynthesis/FastPitch`. |
| `fastpitch_checkpoint` | Có | — | TorchScript FastPitch. |
| `hifigan_checkpoint` | Có | — | TorchScript HiFi-GAN tương thích. |
| `device` | Không | `cuda` | Thiết bị PyTorch. |
| `sample_rate` | Không | `22050` | Sample rate output thực tế. |
| `pace` | Không | `1.0` | Tốc độ nói FastPitch. |
| `speaker` | Không | `0` | Speaker ID cho model multi-speaker. |
| `amp` | Không | `false` | AMP trên CUDA. |
| `symbol_set`, `text_cleaners`, `p_arpabet` | Không | English defaults | Phải khớp checkpoint/tokenizer. |

### OmniVoice

| Option | Bắt buộc | Mặc định | Ý nghĩa |
| --- | --- | --- | --- |
| `model_id` | Không | `k2-fsa/OmniVoice` | Hugging Face ID hoặc thư mục local. |
| `device` | Không | `cuda:0` | `device_map` cho OmniVoice. |
| `dtype` | Không | `float16` | Tên dtype của PyTorch, ví dụ `float16`/`bfloat16`. |
| `ref_audio`, `ref_text` | Không | — | Voice cloning prompt. |
| `instruct` | Không | — | Voice design prompt. |
| `generate_options` | Không | `{}` | Ví dụ `num_step`, `speed`, `duration`, `language_id`. |

Không truyền đồng thời `duration` và `speed`: OmniVoice ưu tiên duration.

## Application streaming

Bật `--application-streaming` để runner chia mỗi dòng text thành sentence/phrase
rồi synth **tuần tự**. Audio của phrase đầu được yield ngay khi model synth xong;
do đó đây là streaming ở cấp ứng dụng, không phải native model streaming.

```powershell
python .\benchmark_tts.py `
  --adapter omnivoice_adapter:create_adapter `
  --adapter-options '{"model_id":"G:\\My Drive\\Documents\\TTS\\benchmark\\models\\OmniVoice","device":"cuda:0","dtype":"float16"}' `
  --texts .\texts.txt `
  --application-streaming `
  --max-segment-chars 120 `
  --warmup 3 `
  --repeat 10 `
  --out-dir .\results-segmented
```

Splitter ưu tiên `.`, `!`, `?`, `;`, `:` rồi gộp sentence đến giới hạn
`--max-segment-chars`; câu quá dài mới bị cắt ở whitespace. Dùng cùng giá trị
này cho tất cả model. Mode này thêm các trường sau vào `runs.*`:

- `streaming_mode=segment_streaming`
- `segment_count`: số phrase synth cho input
- `num_chunks`: số audio chunk thực nhận; với hai adapter hiện tại thường bằng
  `segment_count`

TTFA trong mode này là **TTFA của ứng dụng**: request đến audio của phrase đầu.
Nó thường thấp hơn full-text TTFA, nhưng có trade-off về ngữ điệu giữa các
phrase. Report kết quả kèm `Streaming native = No` và
`Application segment streaming = Yes`; không so sánh trực tiếp TTFA mode này
với full-text/non-streaming như thể chúng là cùng workload.
## Output

`--out-dir results` tạo:

| File | Nội dung |
| --- | --- |
| `runs.csv` / `runs.json` | Mỗi lần chạy: TTFA, RTF, duration, chunk count, input text. |
| `summary.csv` / `summary.json` | Mean/p50/p95 TTFA và RTF aggregate theo model + text. |

Ví dụ đọc một dòng summary:

```text
FastPitch+HiFi-GAN, text 2, TTFA p50=420 ms, E2E RTF=0.08
```

Nghĩa là audio đầu chỉ khả dụng sau khoảng 420 ms; tổng thời gian synth bằng
8% thời lượng audio (nhanh hơn realtime khoảng 12.5 lần).

## Quy tắc so sánh công bằng

1. Load model trước benchmark; không tính model download/load vào TTFA.
2. Cùng GPU, driver, torch version, GPU mode, batch size=1 và không có workload
   GPU khác cạnh tranh.
3. Cùng prompt/voice, speed, diffusion steps, sample text và số lần warmup/repeat.
4. Ghi chính xác precision (`fp16`, `bf16`, `fp32`) vào báo cáo.
5. Report từng `text_id` hoặc bucket độ dài; không chỉ báo một trung bình trên
   mọi độ dài text.
6. Report `Streaming native = No` cho hai adapter hiện tại. Không diễn giải
   `first_chunk_duration` là model streaming.

## Troubleshooting

| Triệu chứng | Cách xử lý |
| --- | --- |
| `ModuleNotFoundError: omnivoice` | Activate venv và chạy `python -m pip install omnivoice`. |
| OmniVoice download lỗi | Kiểm tra mạng/Hugging Face access; chạy `hf download ...` để tải trước. |
| `CUDA was requested but is not available` | Cài PyTorch CUDA đúng driver hoặc đặt `device="cpu"` (rất chậm). |
| `torch.jit.load` lỗi FastPitch | Bạn đang dùng eager `.pt`, không phải TorchScript; xem phần checkpoint format. |
| FastPitch phát âm tiếng Việt sai/lỗi encode | Dùng checkpoint/tokenizer Việt hóa và truyền config text matching checkpoint. |
| FastPitch / HiFi-GAN mismatch | Dùng đúng cặp model; kiểm tra sample rate, hop length, mel configuration. |
| TTFA bất thường ở run đầu | Tăng `--warmup`; không đưa cold-start vào summary warmed-run. |

## Tài liệu nguồn

- NVIDIA FastPitch reference: <https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch>
- NVIDIA NGC FastPitch: <https://catalog.ngc.nvidia.com/orgs/nvidia/dle/resources/fastpitch_pyt>
- OmniVoice repository/API: <https://github.com/k2-fsa/OmniVoice>
- OmniVoice model files: <https://huggingface.co/k2-fsa/OmniVoice>
