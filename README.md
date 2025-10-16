

# 🗣️ Parallel Kannada Text-to-Speech (TTS) using Parler-TTS

A high-performance **multicore Kannada Text-to-Speech system** built with the **AI4Bharat Indic Parler-TTS model**.  
This project demonstrates **serial**, **threaded**, and **multiprocessing-based parallel generation** of synthetic speech — accelerating TTS workloads using modern CPUs and GPUs.

---

## 🚀 Features

- ✅ Kannada TTS using [`ai4bharat/indic-parler-tts-pretrained`](https://huggingface.co/ai4bharat/indic-parler-tts-pretrained)
- ✅ Converts **numbers and Kannada numerals** into spoken Kannada words
- ✅ Three processing modes:
  - **Serial Mode** — baseline sequential generation
  - **Threaded Mode** — concurrent generation using threads
  - **Multiprocessing Mode** — true parallel generation using multiple cores
- ✅ GPU auto-detection and utilization (if available)
- ✅ Adds natural **silence pauses** between sentences
- ✅ Benchmarks **speedup and parallel efficiency**
- ✅ Modular, extensible, and ready for production scaling

---

## 🧠 Model Used

| Component | Model |
|------------|--------|
| **TTS Model** | [`ai4bharat/indic-parler-tts-pretrained`](https://huggingface.co/ai4bharat/indic-parler-tts-pretrained) |
| **Voice Description** | “A professional female speaker Anu, speaks with clear pronunciation and moderate pacing.” |

---

## 📦 Requirements

Install the following dependencies before running:

```bash
pip install torch transformers soundfile numpy optimum parler-tts
````

If you’re using GPU acceleration, ensure you have a **CUDA-compatible** version of PyTorch installed.

---

## 📁 Project Structure

```
├── tts_parallel_kannada.py    # Main script
├── README.md                  # Project documentation
├── kannada_tts_serial.wav     # Output (Serial mode)
├── kannada_tts_parallel.wav   # Output (Parallel mode)
└── requirements.txt           # Python dependencies
```

---

## ⚙️ Configuration (from `TTSConfig` class)

| Parameter              | Description                                                         |
| ---------------------- | ------------------------------------------------------------------- |
| `MODEL_NAME`           | Pretrained model name from Hugging Face                             |
| `VOICE_DESCRIPTION`    | Descriptive prompt defining speaker style                           |
| `OUTPUT_FILE_SERIAL`   | Output file name for serial mode                                    |
| `OUTPUT_FILE_PARALLEL` | Output file name for parallel mode                                  |
| `SILENCE_PAUSE`        | Silence duration (in seconds) between sentences                     |
| `MAX_WORKERS`          | Maximum number of parallel processes (auto based on CPU cores)      |
| `BATCH_SIZE`           | Number of sentences processed per thread batch                      |
| `GENERATION_CONFIG`    | Parameters controlling speech generation (temperature, top_k, etc.) |

---

## 🧩 How It Works

1. **Text Preprocessing**

   * Replaces digits (both English `0–9` and Kannada numerals `೦–೯`) with Kannada word equivalents.
   * Cleans and splits text into manageable sentence chunks.

2. **Model Loading**

   * Loads the `Parler-TTS` model and tokenizers.
   * Uses `BetterTransformer` from the `optimum` library (if installed) for inference optimization.

3. **Processing Modes**

   * **Serial:** Single-process, single-model inference.
   * **Threading:** Multiple threads using one shared model (light parallelism).
   * **Multiprocessing:** Multiple processes with separate model instances (true parallel execution).

4. **Performance Evaluation**

   * Compares total runtime across modes.
   * Calculates **speedup** and **parallel efficiency** metrics.

---

## ▶️ Usage

### 1️⃣ Run Directly (Default CPU or GPU mode)

```bash
python tts_parallel_kannada.py
```

### 2️⃣ Specify GPUs (if multiple available)

```bash
CUDA_VISIBLE_DEVICES=0,1 python tts_parallel_kannada.py
```

> The script automatically detects CUDA and distributes workloads across available GPUs or CPU cores.

---

## 🗒 Example Input

```text
ಕನ್ನಡ ಭಾಷೆ ಭಾರತದ ಕರ್ನಾಟಕ ರಾಜ್ಯದ ಅಧಿಕೃತ ಭಾಷೆಯಾಗಿದೆ.
ಇದು ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬಕ್ಕೆ ಸೇರಿದೆ ಮತ್ತು ಸುಮಾರು ೫ ಕೋಟಿ ಜನರು ಮಾತನಾಡುವ ಪ್ರಮುಖ ಭಾಷೆಯಾಗಿದೆ.
ಕನ್ನಡವು ಅತ್ಯಂತ ಪ್ರಾಚೀನ ಭಾಷೆಗಳಲ್ಲಿೊಂದಾಗಿದೆ, ಇದರ ಇತಿಹಾಸ ಸುಮಾರು 2000 ವರ್ಷಗಳಷ್ಟು ಹಿಂದಕ್ಕೆ ಹೋಗುತ್ತದೆ.
ಕನ್ನಡ ಸಾಹಿತ್ಯವು ಅಸಂಖ್ಯಾತ ಕವಿಗಳು ಮತ್ತು ಲೇಖಕರಿಂದ ಸಮೃದ್ಧವಾಗಿ ಬೆಳೆದು ಬಂದಿದೆ.
ಕುವೆಂಪು, ಬೇಂದ್ರೆ, ಕಾರಂತರಂತಹ ಮಹಾನ್ ಸಾಹಿತಿಗಳು ಕನ್ನಡಕ್ಕೆ ಅಪಾರ ಕೊಡುಗೆ ನೀಡಿದ್ದಾರೆ.
```

The script will automatically convert Kannada numerals like `೫` → `ಐದು` before generating speech.

---

## 🧾 Example Output

**Console Output:**

```
🎯 MULTICORE TTS PARALLEL PROCESSING DEMONSTRATION
💻 System Information:
   CPU Cores: 12
   GPU Available: True
   GPU Count: 1
   GPU 0: NVIDIA GeForce RTX 3060

📊 Dataset: 5 sentences, 412 characters
🔧 Available workers: 4

🚀 Starting SERIAL processing...
✅ Serial processing completed in 42.5 seconds

🚀 Starting PARALLEL processing...
✅ Parallel processing completed in 18.3 seconds

📊 PERFORMANCE ANALYSIS
⏱ Serial Time:    42.50 seconds
⏱ Parallel Time:  18.30 seconds
🚀 Speedup:        2.32x
📈 Efficiency:     58.0%
```

---

## 🧮 Performance Metrics

| Metric         | Description                           |
| -------------- | ------------------------------------- |
| **Speedup**    | Ratio of serial time to parallel time |
| **Efficiency** | Speedup divided by number of workers  |
| **Workers**    | Number of parallel processes used     |

> Example: 2.3× speedup at 58% efficiency on 4 cores.

---

## 🔬 Technical Insights

| Mode         | Characteristics                                                 |
| ------------ | --------------------------------------------------------------- |
| **Serial**   | Single model instance, single core, minimal overhead            |
| **Threaded** | Shared model, lightweight concurrency, limited CPU utilization  |
| **Parallel** | Multiple processes, separate models, full multicore utilization |

---

## ⚡ Optimization Opportunities

* 🧩 **Model Quantization** for reduced memory footprint
* 🚀 **Dynamic Batching** based on sentence length
* 🔁 **Pipeline Parallelism** to overlap tokenization and inference
* ⚙️ **GPU Memory Pooling** for multi-instance acceleration
* 🧠 **Caching Description Embeddings** to avoid recomputation

---

## 📈 Example Performance Comparison

| Mode     | Time (s) | Speedup | Efficiency |
| -------- | -------- | ------- | ---------- |
| Serial   | 42.5     | 1.0x    | 100%       |
| Parallel | 18.3     | 2.3x    | 58%        |

---


* [AI4Bharat](https://ai4bharat.org/) — for Indic TTS datasets and models
* [Hugging Face](https://huggingface.co/) — for Transformers and Parler-TTS
* [Optimum](https://huggingface.co/docs/optimum/index) — for transformer acceleration tools

