import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
import re

warnings.filterwarnings("ignore")

# ------------------ Configuration ------------------
class TTSConfig:
    MODEL_NAME = "ai4bharat/indic-parler-tts-pretrained"
    VOICE_DESCRIPTION = "A professional female speaker Anu, speaks with clear pronunciation and moderate pacing"
    OUTPUT_FILE = "kannada_tts_cpu.wav"
    SILENCE_PAUSE = 0.15
    BATCH_SIZE = 2  # Smaller batch size for CPU
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

# ------------------ TTS Engine ------------------
class TTSEngine:
    def __init__(self):
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.model = None
        self.tokenizer = None
        self.description_tokenizer = None
        self.description_inputs = None
        
    def load_model(self):
        try:
            print("⏳ Loading model on CPU...")
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                TTSConfig.MODEL_NAME,
                torch_dtype=self.torch_dtype
            ).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(TTSConfig.MODEL_NAME)
            self.description_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
            
            with torch.no_grad():
                self.description_inputs = self.description_tokenizer(
                    TTSConfig.VOICE_DESCRIPTION,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).to(self.device)
        except Exception as e:
            print(f"❌ Error loading tokenizers: {e}")
            return False
        
        return True
    
    def process_sentence(self, sentence):
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
            print(f"⚠ Error processing sentence: {e}")
            return None

# ------------------ Threading-based Parallel Processing ------------------
def threading_based_parallel(sentences):
    print("🔧 Starting THREADING-based CPU processing...")
    start_time = time.time()
    
    engine = TTSEngine()
    if not engine.load_model():
        print("❌ Model could not be loaded. Exiting.")
        return None, 0
    
    full_audio = np.array([])
    sampling_rate = engine.model.config.sampling_rate
    
    for i in range(0, len(sentences), TTSConfig.BATCH_SIZE):
        batch = sentences[i:i + TTSConfig.BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=min(len(batch), TTSConfig.BATCH_SIZE)) as executor:
            batch_results = list(executor.map(engine.process_sentence, batch))
        
        for j, audio_arr in enumerate(batch_results):
            if audio_arr is not None:
                full_audio = np.concatenate((full_audio, audio_arr))
                if j < len(batch_results) - 1 or i + TTSConfig.BATCH_SIZE < len(sentences):
                    full_audio = np.concatenate((
                        full_audio,
                        np.zeros(int(TTSConfig.SILENCE_PAUSE * sampling_rate))
                    ))
        print(f"✅ Processed batch {i//TTSConfig.BATCH_SIZE + 1}/{(len(sentences)//TTSConfig.BATCH_SIZE) + 1}")
    
    total_time = time.time() - start_time
    print(f"✅ CPU threading processing completed in {total_time:.2f} seconds")
    return full_audio, total_time

# ------------------ Main Execution ------------------
def main():
    print("=" * 60)
    print("🎯 CPU-ONLY Kannada TTS Demo")
    print("=" * 60)
    
    # Sample Kannada text (5 sentences)
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
    
    # Run threading-based CPU processing
    audio, total_time = threading_based_parallel(sentences)
    if audio is not None:
        sf.write(TTSConfig.OUTPUT_FILE, audio, 24000)
        print(f"💾 Output saved to: {TTSConfig.OUTPUT_FILE}")
        print(f"⏱ Duration: {len(audio)/24000:.2f} seconds")
        print(f"⏱ Total processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
