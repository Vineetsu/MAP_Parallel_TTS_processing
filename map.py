import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np
import warnings
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import re
from functools import partial

# Suppress warnings
warnings.filterwarnings("ignore")

# ------------------ Configuration ------------------
class TTSConfig:
    MODEL_NAME = "ai4bharat/indic-parler-tts-pretrained"
    VOICE_DESCRIPTION = "A professional female speaker Anu, speaks with clear pronunciation and moderate pacing"
    OUTPUT_FILE_SERIAL = "kannada_tts_serial.wav"
    OUTPUT_FILE_PARALLEL = "kannada_tts_parallel.wav"
    SILENCE_PAUSE = 0.15  # seconds between sentences
    
    # Parallel processing settings
    MAX_WORKERS = min(4, mp.cpu_count())  # Adjust based on your CPU cores
    BATCH_SIZE = 4
    
    # Generation config
    GENERATION_CONFIG = {
        'do_sample': True,
        'temperature': 0.95,
        'top_k': 50,
        'top_p': 0.95,
        'max_new_tokens': 1024,
        'num_beams': 1,
        'use_cache': True
    }

# ------------------ Text Preprocessing ------------------
number_map = {
    "0": "ಸೊನ್ನೆ", "1": "ಒಂದು", "2": "ಎರಡು", "3": "ಮೂರು", "4": "ನಾಲ್ಕು",
    "5": "ಐದು", "6": "ಆರು", "7": "ಏಳು", "8": "ಎಂಟು", "9": "ಒಂಭತ್ತು",
    "10": "ಹತ್ತು", "50": "ಐವತ್ತು", "100": "ನೂರು", "2000": "ಎರಡು ಸಾವಿರ",
    "2024": "ಎರಡು ಸಾವಿರ ಇಪ್ಪತ್ತ್ನಾಲ್ಕು", "50000000": "ಐದು ಕೋಟಿ"
}

kannada_digit_map = {
    "೦": "ಸೊನ್ನೆ", "೧": "ಒಂದು", "೨": "ಎರಡು", "೩": "ಮೂರು", 
    "೪": "ನಾಲ್ಕು", "೫": "ಐದು", "೬": "ಆರು", "೭": "ಏಳು", "೮": "ಎಂಟು", "೯": "ಒಂಭತ್ತು"
}

def replace_numbers_with_kannada_words(text):
    def replace_match(match):
        number = match.group(0)
        return number_map.get(number, number)
    
    text = re.sub(r'\d+', replace_match, text)
    for digit, word in kannada_digit_map.items():
        text = text.replace(digit, word)
    return text

# ------------------ Model Management ------------------
class TTSEngine:
    def __init__(self, device_id=None):
        self.device = torch.device(f"cuda:{device_id}" if device_id is not None and torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = None
        self.tokenizer = None
        self.description_tokenizer = None
        self.description_inputs = None
        
    def load_model(self):
        """Load model and tokenizers"""
        try:
            print(f"⏳ Loading model for device: {self.device}")
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                TTSConfig.MODEL_NAME,
                torch_dtype=self.torch_dtype
            ).to(self.device)
            self.model.eval()
            
            # Try to apply optimizations
            try:
                from optimum.bettertransformer import BetterTransformer
                self.model = BetterTransformer.transform(self.model)
                print(f"✅ BetterTransformer enabled for {self.device}")
            except ImportError:
                pass
                
        except Exception as e:
            print(f"❌ Error loading model for {self.device}: {e}")
            return False
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(TTSConfig.MODEL_NAME)
            self.description_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
            
            # Pre-tokenize description
            with torch.no_grad():
                self.description_inputs = self.description_tokenizer(
                    TTSConfig.VOICE_DESCRIPTION,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).to(self.device)
                
        except Exception as e:
            print(f"❌ Error loading tokenizers for {self.device}: {e}")
            return False
            
        return True
    
    def process_sentence(self, sentence):
        """Process a single sentence (serial mode)"""
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    sentence,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).to(self.device)

                audio = self.model.generate(
                    input_ids=self.description_inputs.input_ids,
                    attention_mask=self.description_inputs.attention_mask,
                    prompt_input_ids=inputs.input_ids,
                    prompt_attention_mask=inputs.attention_mask,
                    **TTSConfig.GENERATION_CONFIG
                )
                return audio.cpu().numpy().squeeze().astype('float32')
        except Exception as e:
            print(f"⚠ Error processing sentence on {self.device}: {e}")
            return None

# ------------------ Parallel Worker Function ------------------
def parallel_worker(sentence_batch, worker_id):
    """
    Worker function for parallel processing
    Each worker gets its own model instance
    """
    try:
        # Set CUDA device for this worker if available
        if torch.cuda.is_available():
            torch.cuda.set_device(worker_id % torch.cuda.device_count())
        
        # Create engine for this worker
        engine = TTSEngine(worker_id % torch.cuda.device_count() if torch.cuda.is_available() else None)
        if not engine.load_model():
            return []
        
        results = []
        for sentence in sentence_batch:
            audio = engine.process_sentence(sentence)
            if audio is not None:
                results.append(audio)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return results
        
    except Exception as e:
        print(f"❌ Worker {worker_id} failed: {e}")
        return []

# ------------------ Processing Modes ------------------
def serial_processing(sentences, config):
    """Sequential processing - baseline for comparison"""
    print("🚀 Starting SERIAL processing...")
    start_time = time.time()
    
    engine = TTSEngine()
    if not engine.load_model():
        return None
    
    full_audio = np.array([])
    sampling_rate = engine.model.config.sampling_rate
    
    for i, sentence in enumerate(sentences):
        print(f"📝 Processing sentence {i+1}/{len(sentences)}...")
        audio_arr = engine.process_sentence(sentence)
        
        if audio_arr is not None:
            full_audio = np.concatenate((full_audio, audio_arr))
            # Add pause between sentences (except after last one)
            if i < len(sentences) - 1:
                full_audio = np.concatenate((
                    full_audio,
                    np.zeros(int(config.SILENCE_PAUSE * sampling_rate))
                ))
    
    serial_time = time.time() - start_time
    print(f"✅ Serial processing completed in {serial_time:.2f} seconds")
    return full_audio, serial_time

def parallel_processing(sentences, config):
    """True parallel processing using multiprocessing"""
    print("🚀 Starting PARALLEL processing...")
    start_time = time.time()
    
    # Split sentences into batches for workers
    sentence_batches = []
    batch_size = max(1, len(sentences) // config.MAX_WORKERS)
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        sentence_batches.append(batch)
    
    print(f"🔧 Using {len(sentence_batches)} workers for {len(sentences)} sentences")
    
    # Use ProcessPoolExecutor for true parallelism
    full_audio = np.array([])
    sampling_rate = None
    
    try:
        with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            # Map batches to workers
            worker_func = partial(parallel_worker, worker_id=os.getpid())
            results = list(executor.map(worker_func, 
                                      sentence_batches, 
                                      range(len(sentence_batches))))
            
            # Combine results
            for batch_result in results:
                for j, audio_arr in enumerate(batch_result):
                    if audio_arr is not None:
                        full_audio = np.concatenate((full_audio, audio_arr))
                        # Add pause between sentences
                        full_audio = np.concatenate((
                            full_audio,
                            np.zeros(int(config.SILENCE_PAUSE * sampling_rate if sampling_rate else 24000))
                        ))
            
            # Get sampling rate from first successful result
            if results and results[0]:
                sampling_rate = 24000  # Default sampling rate
    
    except Exception as e:
        print(f"❌ Parallel processing failed: {e}")
        # Fallback to threading-based parallelism
        print("🔄 Falling back to threading-based parallelism...")
        return threading_based_parallel(sentences, config)
    
    parallel_time = time.time() - start_time
    print(f"✅ Parallel processing completed in {parallel_time:.2f} seconds")
    return full_audio, parallel_time

def threading_based_parallel(sentences, config):
    """Threading-based parallelism (for comparison)"""
    print("🔧 Using THREADING-based parallelism...")
    start_time = time.time()
    
    engine = TTSEngine()
    if not engine.load_model():
        return None
    
    full_audio = np.array([])
    sampling_rate = engine.model.config.sampling_rate
    
    # Process in batches using threads
    for i in range(0, len(sentences), config.BATCH_SIZE):
        batch = sentences[i:i + config.BATCH_SIZE]
        
        with ThreadPoolExecutor(max_workers=min(len(batch), config.BATCH_SIZE)) as executor:
            batch_results = list(executor.map(engine.process_sentence, batch))
        
        for j, audio_arr in enumerate(batch_results):
            if audio_arr is not None:
                full_audio = np.concatenate((full_audio, audio_arr))
                if j < len(batch_results) - 1 or i + config.BATCH_SIZE < len(sentences):
                    full_audio = np.concatenate((
                        full_audio,
                        np.zeros(int(config.SILENCE_PAUSE * sampling_rate))
                    ))
        
        print(f"✅ Processed batch {i//config.BATCH_SIZE + 1}/{(len(sentences)//config.BATCH_SIZE) + 1}")
    
    threading_time = time.time() - start_time
    print(f"✅ Threading-based processing completed in {threading_time:.2f} seconds")
    return full_audio, threading_time

# ------------------ Main Execution ------------------
def main():
    print("=" * 60)
    print("🎯 MULTICORE TTS PARALLEL PROCESSING DEMONSTRATION")
    print("=" * 60)
    
    # System info
    print(f"💻 System Information:")
    print(f"   CPU Cores: {mp.cpu_count()}")
    print(f"   GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Load and preprocess text
    kannada_text = """
    ಕನ್ನಡ ಭಾಷೆ ಭಾರತದ ಕರ್ನಾಟಕ ರಾಜ್ಯದ ಅಧಿಕೃತ ಭಾಷೆಯಾಗಿದೆ.
    ಇದು ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬಕ್ಕೆ ಸೇರಿದೆ ಮತ್ತು ಸುಮಾರು ೫ ಕೋಟಿ ಜನರು ಮಾತನಾಡುವ ಪ್ರಮುಖ ಭಾಷೆಯಾಗಿದೆ.
    ಕನ್ನಡವು ಅತ್ಯಂತ ಪ್ರಾಚೀನ ಭಾಷೆಗಳಲ್ಲಿೊಂದಾಗಿದೆ, ಇದರ ಇತಿಹಾಸ ಸುಮಾರು 2000 ವರ್ಷಗಳಷ್ಟು ಹಿಂದಕ್ಕೆ ಹೋಗುತ್ತದೆ.
    ಕನ್ನಡ ಸಾಹಿತ್ಯವು ಅಸಂಖ್ಯಾತ ಕವಿಗಳು ಮತ್ತು ಲೇಖಕರಿಂದ ಸಮೃದ್ಧವಾಗಿ ಬೆಳೆದು ಬಂದಿದೆ.
    ಕುವೆಂಪು, ಬೇಂದ್ರೆ, ಕಾರಂತರಂತಹ ಮಹಾನ್ ಸಾಹಿತಿಗಳು ಕನ್ನಡಕ್ಕೆ ಅಪಾರ ಕೊಡುಗೆ ನೀಡಿದ್ದಾರೆ.
    """.strip()
    
    kannada_text = replace_numbers_with_kannada_words(kannada_text)
    sentences = [s.strip() + '.' for s in kannada_text.split('.') if s.strip()]
    
    print(f"📊 Dataset: {len(sentences)} sentences, {len(kannada_text)} characters")
    print(f"🔧 Available workers: {TTSConfig.MAX_WORKERS}")
    
    # Run serial processing
    print("\n" + "="*50)
    serial_audio, serial_time = serial_processing(sentences, TTSConfig)
    if serial_audio is not None:
        sf.write(TTSConfig.OUTPUT_FILE_SERIAL, serial_audio, 24000)
        print(f"💾 Serial output saved to: {TTSConfig.OUTPUT_FILE_SERIAL}")
    
    # Run parallel processing
    print("\n" + "="*50)
    parallel_audio, parallel_time = parallel_processing(sentences, TTSConfig)
    if parallel_audio is not None:
        sf.write(TTSConfig.OUTPUT_FILE_PARALLEL, parallel_audio, 24000)
        print(f"💾 Parallel output saved to: {TTSConfig.OUTPUT_FILE_PARALLEL}")
    
    # Performance analysis
    print("\n" + "="*60)
    print("📊 PERFORMANCE ANALYSIS")
    print("="*60)
    
    if serial_time and parallel_time:
        speedup = serial_time / parallel_time
        efficiency = (speedup / TTSConfig.MAX_WORKERS) * 100
        
        print(f"⏱ Serial Time:    {serial_time:.2f} seconds")
        print(f"⏱ Parallel Time:  {parallel_time:.2f} seconds")
        print(f"🚀 Speedup:        {speedup:.2f}x")
        print(f"📈 Efficiency:     {efficiency:.1f}%")
        print(f"🔧 Workers Used:   {TTSConfig.MAX_WORKERS}")
        
        if speedup > 1:
            print("✅ Parallel processing provided performance improvement!")
        else:
            print("⚠️ Parallel overhead exceeded benefits for this workload")
    
    # Technical insights
    print("\n" + "="*60)
    print("🔬 TECHNICAL INSIGHTS")
    print("="*60)
    print("1. SERIAL PROCESSING:")
    print("   - Single process, single model instance")
    print("   - Simple but utilizes only one CPU core")
    print("   - No parallel overhead")
    
    print("\n2. PARALLEL PROCESSING:")
    print("   - Multiple processes, multiple model instances")
    print("   - Utilizes multiple CPU cores")
    print("   - Higher memory usage (model replication)")
    print("   - Communication overhead between processes")
    
    print("\n3. OPTIMIZATION OPPORTUNITIES:")
    print("   - Model quantization for faster loading")
    print("   - Pipeline parallelism (tokenize/generate overlap)")
    print("   - Dynamic batching based on sentence length")
    print("   - GPU memory pooling for multiple workers")

if __name__ == "__main__":
    # Required for Windows multiprocessing
    if os.name == 'nt':
        mp.freeze_support()
    
    main()