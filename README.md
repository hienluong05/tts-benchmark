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
| `fastpitch_adapter.py` | Load FastPitch + HiFi-GAN eager `.pt` từ `third_party/FastPitch`; vẫn hỗ trợ TorchScript. |
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
- FastPitch chỉ: source local `third_party/FastPitch` và cặp checkpoint FastPitch/HiFi-GAN eager PyTorch `.pt` tương thích.
- OmniVoice chỉ: package `omnivoice`; lần load đầu sẽ tải model khoảng 3.27 GB.

Tạo môi trường riêng, ví dụ Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "setuptools<70" wheel
```

Cài PyTorch CUDA trước bằng wheel index phù hợp với server. Ví dụ CUDA 12.8:

```powershell
python -m pip install torch==2.8.0 torchaudio==2.8.0 `
  --index-url https://download.pytorch.org/whl/cu128
```

Sau đó cài toàn bộ dependency của benchmark:

```powershell
python -m pip install -r .\requirements.txt
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

Adapter dùng fork đã chỉnh sửa tại `third_party/FastPitch`:

```text
G:\My Drive\Documents\TTS\benchmark\third_party\FastPitch
```

Tải checkpoint pretrained từ NVIDIA NGC hoặc dùng checkpoint fine-tune của dự
án. FastPitch và vocoder phải tương thích mel configuration, nhất là sample
rate, hop length, số mel bins và text processing. Adapter mặc định load eager
PyTorch `.pt` qua đúng `third_party/FastPitch/models.py` và
`models.load_and_setup_model()`, vì vậy các sửa lỗi trong fork này được dùng.

### Định dạng checkpoint hiện được hỗ trợ

Mặc định dùng eager PyTorch:

- `checkpoint_format: "pyt"` (hoặc bỏ qua option này)
- `fastpitch_checkpoint`: FastPitch `.pt` có `state_dict` và config
- `hifigan_checkpoint`: HiFi-GAN `.pt` có `generator` và config
- `hifigan_config`: file config JSON, chỉ cần khi checkpoint HiFi-GAN không
  chứa architecture config

Vẫn có thể dùng TorchScript bằng `checkpoint_format: "ts"`; khi đó cả hai
checkpoint phải là TorchScript. Không đổi đuôi `.pt` thành `.ts`.

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
  --adapter-options '{"repo_dir":"G:\\My Drive\\Documents\\TTS\\benchmark\\third_party\\FastPitch","fastpitch_checkpoint":"G:\\My Drive\\Documents\\TTS\\benchmark\\models\\nvidia_fastpitch_220224.pt","hifigan_checkpoint":"G:\\My Drive\\Documents\\TTS\\benchmark\\models\\hifigan__pyt_ckpt_mode-finetune_ds-ljs22khz_v21.08.0_amp\\hifigan_gen_checkpoint_10000_ft.pt","checkpoint_format":"pyt","device":"cuda","sample_rate":22050,"text_cleaners":["english_cleaners_v2"],"p_arpabet":1.0,"amp":true}' `
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
| `repo_dir` | Có | — | Thư mục `third_party/FastPitch` của fork local. |
| `fastpitch_checkpoint` | Có | — | FastPitch `.pt` eager mặc định; `.ts` khi `checkpoint_format="ts"`. |
| `hifigan_checkpoint` | Có | — | HiFi-GAN `.pt` eager mặc định; `.ts` khi `checkpoint_format="ts"`. |
| `checkpoint_format` | Không | `pyt` | `pyt` cho eager `.pt`, hoặc `ts` cho TorchScript. |
| `hifigan_config` | Không | — | JSON config nếu eager HiFi-GAN checkpoint không chứa config. |
| `device` | Không | `cuda` | Thiết bị PyTorch. |
| `sample_rate` | Không | `22050` | Sample rate output thực tế. |
| `pace` | Không | `1.0` | Tốc độ nói FastPitch. |
| `speaker` | Không | `0` | Speaker ID cho model multi-speaker. |
| `amp` | Không | `false` | AMP trên CUDA. |
| `symbol_set`, `text_cleaners`, `p_arpabet` | Không | `english_basic`, `["english_cleaners_v2"]`, `1.0` | Phải khớp checkpoint/tokenizer. |

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

## Offline deployment lên server

Không commit checkpoint vào Git hoặc Git LFS. Tải model ở máy có mạng, kiểm tra
đủ file, rồi chuyển trực tiếp sang server bằng `scp`, `rsync` hoặc artifact
storage. Thư mục `models/` đã được `.gitignore` bỏ qua.

### OmniVoice

Tải trước ở máy local:

```powershell
python -m pip install "huggingface_hub[cli]"
hf download k2-fsa/OmniVoice --local-dir .\models\OmniVoice
```

Lệnh này tải model, tokenizer và audio tokenizer vào cùng thư mục. Đóng gói
không nén để giảm overhead CPU (checkpoint safetensors/Xet vốn đã lớn):

```powershell
tar -cf omnivoice-model.tar -C .\models OmniVoice
scp .\omnivoice-model.tar user@server:/opt/tts-benchmark/
```

Giải nén trên server:

```bash
cd /opt/tts-benchmark
mkdir -p models
tar -xf omnivoice-model.tar -C models
```

Trỏ adapter đến folder local, không dùng Hugging Face model ID:

```bash
--adapter omnivoice_adapter:create_adapter \
--adapter-options '{"model_id":"/opt/tts-benchmark/models/OmniVoice","device":"cuda:0","dtype":"float16"}'
```

Ép offline để mọi remote request bị fail ngay:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Nếu chạy voice cloning, copy cả `reference.wav` và dùng đường dẫn server cho
`ref_audio`. Có thể kiểm tra transfer bằng checksum trước/sau khi copy:

```powershell
Get-FileHash .\omnivoice-model.tar -Algorithm SHA256
```

```bash
sha256sum /opt/tts-benchmark/omnivoice-model.tar
```

### FastPitch + HiFi-GAN

Chuyển đủ source, không chỉ checkpoint, vì adapter import text processor từ
NVIDIA DeepLearningExamples:

```text
/opt/tts-benchmark/
├── fastpitch_adapter.py
├── models/
│   ├── fastpitch.ts
│   └── hifigan.ts
└── third_party/
    └── DeepLearningExamples/
        └── PyTorch/SpeechSynthesis/FastPitch/
```

Ví dụ copy từ máy local:

```powershell
tar -cf fastpitch-bundle.tar .\models\fastpitch.pt .\models\hifigan.pt .\third_party\FastPitch
scp .\fastpitch-bundle.tar user@server:/opt/tts-benchmark/
```

Trên server, `repo_dir`, `fastpitch_checkpoint` và `hifigan_checkpoint` phải
là các đường dẫn tuyệt đối Linux trong `--adapter-options`.

### Python dependencies trên server

Model files có thể chuyển giữa Windows và Linux, nhưng không copy virtualenv
hoặc PyTorch wheel từ Windows sang Linux. Cài dependencies đúng OS/CPU/CUDA của
server. Nếu server cũng không có Internet, tạo wheelhouse từ một máy **cùng
nền tảng Linux/CUDA** hoặc dùng internal package mirror; sau đó cài bằng
`pip install --no-index --find-links /path/to/wheelhouse ...`.
## Quy trình chạy trên Linux/GPU server

Các lệnh dưới đây phù hợp với cấu trúc project trên server. Chạy từ thư mục gốc
`/media/data3/users/luongdth/tts-benchmark` và thay đường dẫn nếu project nằm ở
vị trí khác.

### 1. Kiểm tra source, model và input

```bash
cd /media/data3/users/luongdth/tts-benchmark
wc -l benchmark_tts.py
python -u benchmark_tts.py --help
find models/OmniVoice -maxdepth 2 -type f | head
ls -lh models/nvidia_fastpitch_220224.pt
ls -lh models/hifigan__pyt_ckpt_mode-finetune_ds-ljs22khz_v21.08.0_amp/hifigan_gen_checkpoint_10000_ft.pt
ls -lh third_party/FastPitch/cmudict/cmudict-0.7b
wc -l texts.txt
```

`texts.txt` là file UTF-8, mỗi dòng một câu. Không dùng `--adapter-options '...'`;
đó chỉ là placeholder và gây `JSONDecodeError`.

### 2. Cài dependency và kiểm tra GPU

```bash
conda activate tts-benchmark
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import omnivoice, soxr; print('OmniVoice dependencies OK')"
```

Nếu package đã cài nhưng `soxr` thiếu, chạy `python -m pip install soxr`.
Không gõ dính thành `pip install omnivoicesoxr`; hai package này có tên riêng.

### 3. Chuẩn bị CMUdict cho FastPitch

Khi `p_arpabet=1.0`, FastPitch bắt buộc có file CMUdict:

```bash
cd third_party/FastPitch
bash scripts/download_cmudict.sh
cd ../..
```

Kiểm tra file phải tồn tại tại:

```text
third_party/FastPitch/cmudict/cmudict-0.7b
```

Nếu server không có Internet, tải file ở máy có mạng rồi copy vào đúng đường dẫn
trên server. Có thể truyền đường dẫn khác qua option `cmudict_path`.

### 4. Chạy smoke test FastPitch

Chạy trước một lần để xác nhận checkpoint, tokenizer và CUDA hoạt động:

```bash
python -u benchmark_tts.py \\
  --adapter fastpitch_adapter:create_adapter \\
  --adapter-options '{"repo_dir":"/media/data3/users/luongdth/tts-benchmark/third_party/FastPitch","fastpitch_checkpoint":"/media/data3/users/luongdth/tts-benchmark/models/nvidia_fastpitch_220224.pt","hifigan_checkpoint":"/media/data3/users/luongdth/tts-benchmark/models/hifigan__pyt_ckpt_mode-finetune_ds-ljs22khz_v21.08.0_amp/hifigan_gen_checkpoint_10000_ft.pt","checkpoint_format":"pyt","cmudict_path":"/media/data3/users/luongdth/tts-benchmark/third_party/FastPitch/cmudict/cmudict-0.7b","device":"cuda","sample_rate":22050,"text_cleaners":["english_cleaners_v2"],"p_arpabet":1.0,"amp":true}' \\
  --texts texts.txt \\
  --warmup 1 \\
  --repeat 1 \\
  --out-dir results-smoke
```

Kiểm tra mã thoát và output:

```bash
echo $?
column -s, -t < results-smoke/summary.csv
```

### 5. Chạy so sánh FastPitch và OmniVoice

Sau khi smoke test thành công, chạy benchmark đầy đủ với cùng `texts.txt`,
`warmup`, `repeat`, GPU và precision:

```bash
python -u benchmark_tts.py \\
  --adapter fastpitch_adapter:create_adapter \\
  --adapter-options '{"repo_dir":"/media/data3/users/luongdth/tts-benchmark/third_party/FastPitch","fastpitch_checkpoint":"/media/data3/users/luongdth/tts-benchmark/models/nvidia_fastpitch_220224.pt","hifigan_checkpoint":"/media/data3/users/luongdth/tts-benchmark/models/hifigan__pyt_ckpt_mode-finetune_ds-ljs22khz_v21.08.0_amp/hifigan_gen_checkpoint_10000_ft.pt","checkpoint_format":"pyt","cmudict_path":"/media/data3/users/luongdth/tts-benchmark/third_party/FastPitch/cmudict/cmudict-0.7b","device":"cuda","sample_rate":22050,"text_cleaners":["english_cleaners_v2"],"p_arpabet":1.0,"amp":true}' \\
  --adapter omnivoice_adapter:create_adapter \\
  --adapter-options '{"model_id":"/media/data3/users/luongdth/tts-benchmark/models/OmniVoice","device":"cuda:0","dtype":"float16","generate_options":{"num_step":32,"speed":1.0}}' \\
  --texts texts.txt \\
  --warmup 3 \\
  --repeat 10 \\
  --out-dir results-full
```

Mỗi adapter phải có đúng một `--adapter-options` ngay sau nó. Nếu chỉ thấy
`FastPitch+HiFi-GAN` trong CSV thì lệnh chỉ chạy FastPitch hoặc OmniVoice đã lỗi
khi import/load. Kiểm tra log cuối cùng và chạy lại sau khi sửa dependency.

### 6. Chạy application streaming

Thêm hai flag sau vào lệnh benchmark đầy đủ:

```bash
--application-streaming \\
--max-segment-chars 120
```

Ví dụ output nằm trong `results-application-streaming/`. Đây là application-level
streaming: runner chia text thành phrase rồi tổng hợp tuần tự. Hai adapter hiện
tại vẫn trả một waveform hoàn chỉnh cho mỗi phrase, nên không phải native model
streaming. Với câu ngắn hơn 120 ký tự, `segments=1` và `chunks=1` là bình thường.

### 7. Đọc và lưu kết quả

```bash
find results-full -maxdepth 1 -type f -ls
column -s, -t < results-full/summary.csv
cut -d, -f1 results-full/runs.csv | sort | uniq
```

`runs.csv` chứa từng run; `summary.csv` chứa aggregate theo model và text.
`ttfa_*_ms` là latency audio đầu tiên, `e2e_rtf_ratio_of_total` là tổng thời gian
chia thời lượng audio. RTF nhỏ hơn 1 nghĩa là nhanh hơn realtime. `repeat=1` chỉ
phù hợp smoke test; báo cáo nên dùng tối thiểu 10 repeats và warmup trước.
## Troubleshooting

| Triệu chứng | Cách xử lý |
| --- | --- |
| `ModuleNotFoundError: omnivoice` | Activate venv và chạy `python -m pip install omnivoice`. |
| `ModuleNotFoundError: soxr` hoặc `Could not import module 'HiggsAudioV2TokenizerModel'` | OmniVoice/Transformers cần thư viện audio `soxr`. Chạy `python -m pip install soxr`, hoặc cài lại toàn bộ `python -m pip install -r requirements.txt`. |
| `ValueError: CMUDict not initialized` | FastPitch đang dùng `p_arpabet > 0` nhưng CMUdict chưa được khởi tạo. Đảm bảo adapter import `from common.text import cmudict` và file `cmudict-0.7b` tồn tại. |
| `CMUDict is required ... missing .../cmudict-0.7b` | Từ thư mục `third_party/FastPitch`, chạy `bash scripts/download_cmudict.sh`, hoặc truyền `cmudict_path` tới file local. |
| `NameError: name 'cmudict' is not defined` | Bản adapter cũ thiếu import. Thêm `from common.text import cmudict` trong khối import FastPitch và chạy lại `python -m py_compile fastpitch_adapter.py`. |
| Lệnh kết thúc với mã `0` nhưng không có thư mục kết quả | Kiểm tra `benchmark_tts.py` không phải file rỗng (`wc -l benchmark_tts.py`), chạy `python -u benchmark_tts.py --help`, và kiểm tra đúng `--out-dir`. |
| `JSONDecodeError` tại `json.loads` | `--adapter-options` phải là JSON đầy đủ; không dùng placeholder `'...'`. Mỗi `--adapter` cần một `--adapter-options` tương ứng. |
| OmniVoice download lỗi | Kiểm tra mạng/Hugging Face access; chạy `hf download ...` để tải trước. |
| `CUDA was requested but is not available` | Cài PyTorch CUDA đúng driver hoặc đặt `device="cpu"` (rất chậm). |
| FastPitch checkpoint load lỗi | Xác nhận `checkpoint_format`, fork `repo_dir`, và cặp checkpoint/config tương thích. |
| FastPitch phát âm tiếng Việt sai/lỗi encode | Dùng checkpoint/tokenizer Việt hóa và truyền config text matching checkpoint. |
| FastPitch / HiFi-GAN mismatch | Dùng đúng cặp model; kiểm tra sample rate, hop length, mel configuration. |
| TTFA bất thường ở run đầu | Tăng `--warmup`; không đưa cold-start vào summary warmed-run. |

## Tài liệu nguồn

- NVIDIA FastPitch reference: <https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch>
- NVIDIA NGC FastPitch: <https://catalog.ngc.nvidia.com/orgs/nvidia/dle/resources/fastpitch_pyt>
- OmniVoice repository/API: <https://github.com/k2-fsa/OmniVoice>
- OmniVoice model files: <https://huggingface.co/k2-fsa/OmniVoice>
