# Whisper - OpenAI Speech Recognition

Complete guide to OpenAI's Whisper, a robust automatic speech recognition (ASR) system trained on 680,000 hours of multilingual data.

## Table of Contents
- [Introduction](#introduction)
- [Model Versions](#model-versions)
- [Installation & Setup](#installation--setup)
- [Basic Usage](#basic-usage)
- [Fine-tuning](#fine-tuning)
- [Common Patterns](#common-patterns)
- [Advanced Operations](#advanced-operations)
- [Optimization](#optimization)
- [Deployment](#deployment)
- [Advanced Techniques](#advanced-techniques)
- [Integration](#integration)
- [Best Practices](#best-practices)

## Introduction

Whisper is OpenAI's state-of-the-art automatic speech recognition (ASR) model, released in September 2022. It's trained on 680,000 hours of multilingual and multitask supervised data collected from the web, making it robust to accents, background noise, and technical language.

### Key Features

- **Multilingual**: Supports 99 languages
- **Robust**: Works with noisy audio, accents, technical terms
- **Multitask**: Transcription, translation, language identification, timestamp generation
- **Open Source**: Available under MIT license
- **Multiple Sizes**: From 39M (tiny) to 1550M (large) parameters
- **High Accuracy**: Near-human level performance on clean audio
- **Zero-shot**: Works without fine-tuning

### Architecture

- **Encoder-Decoder Transformer**: Based on sequence-to-sequence architecture
- **Mel Spectrogram Input**: 80-channel log-mel spectrogram
- **Multi-head Attention**: Self and cross-attention mechanisms
- **Positional Encoding**: Sinusoidal positional embeddings
- **Special Tokens**: Task-specific tokens for control
- **Byte Pair Encoding**: Multilingual tokenizer

### Supported Languages

```python
# Major languages supported (99 total)
languages = [
    "English", "Chinese", "Spanish", "French", "German", "Japanese",
    "Portuguese", "Russian", "Korean", "Arabic", "Hindi", "Italian",
    "Dutch", "Polish", "Turkish", "Vietnamese", "Indonesian", "Thai",
    "Hebrew", "Greek", "Czech", "Romanian", "Swedish", "Hungarian"
    # ... and 75 more
]
```

### Tasks

1. **Transcription**: Audio → text in same language
2. **Translation**: Audio → English text
3. **Language Detection**: Identify spoken language
4. **Timestamp Generation**: Word-level or segment-level timing
5. **Voice Activity Detection**: Detect speech regions

## Model Versions

### Overview

| Model | Parameters | VRAM (FP16) | Relative Speed | English WER |
|-------|------------|-------------|----------------|-------------|
| tiny | 39M | 1GB | 32x | 7.5% |
| base | 74M | 1GB | 16x | 5.5% |
| small | 244M | 2GB | 6x | 3.5% |
| medium | 769M | 5GB | 2x | 2.8% |
| large | 1550M | 10GB | 1x | 2.3% |
| large-v2 | 1550M | 10GB | 1x | 2.1% |
| large-v3 | 1550M | 10GB | 1x | 1.8% |

### Tiny Model

**Best for: Real-time applications, edge devices, quick prototyping**

```python
import whisper

model = whisper.load_model("tiny")
result = model.transcribe("audio.mp3")
print(result["text"])
```

**Characteristics:**
- Fastest inference
- Lowest memory usage
- Good for English
- Lower accuracy on noisy audio
- 32x faster than large

### Base Model

**Best for: Balanced speed and accuracy**

```python
model = whisper.load_model("base")
result = model.transcribe("audio.mp3", language="en")
```

**Characteristics:**
- Fast inference
- Decent accuracy
- Good multilingual support
- 16x faster than large
- Low resource requirements

### Small Model

**Best for: Production applications with reasonable accuracy**

```python
model = whisper.load_model("small")
result = model.transcribe(
    "audio.mp3",
    language="en",
    task="transcribe"
)
```

**Characteristics:**
- Balanced speed/accuracy
- Good multilingual performance
- Handles accents well
- 6x faster than large
- Popular choice for APIs

### Medium Model

**Best for: High accuracy without extreme resources**

```python
model = whisper.load_model("medium")
result = model.transcribe(
    "audio.mp3",
    language="es",
    verbose=True
)
```

**Characteristics:**
- High accuracy
- Good for difficult audio
- Better punctuation
- 2x faster than large
- Good multilingual performance

### Large Models (v1, v2, v3)

**Best for: Maximum accuracy, research, difficult audio**

```python
# Large-v3 (recommended)
model = whisper.load_model("large-v3")

# Large-v2
model = whisper.load_model("large-v2")

# Large (original)
model = whisper.load_model("large")

result = model.transcribe(
    "audio.mp3",
    task="transcribe",
    language="en"
)
```

**Characteristics:**
- Highest accuracy
- Best multilingual support
- Handles noise, accents, dialects
- Slower inference
- Large-v3 is most recent and accurate

**Version Differences:**
- **v1**: Original release
- **v2**: Improved for difficult audio, better timestamps
- **v3**: Best overall, improved low-resource languages

### Model Selection Guide

```python
def select_model(use_case):
    models = {
        "realtime": "tiny",           # Live transcription
        "mobile": "tiny",              # Mobile apps
        "chatbot": "base",             # Voice assistants
        "subtitles": "small",          # Video subtitles
        "meetings": "medium",          # Meeting transcription
        "medical": "large-v3",         # Medical dictation
        "legal": "large-v3",           # Legal transcription
        "research": "large-v3",        # Academic research
        "multilingual": "large-v3",    # Multiple languages
        "low_latency": "tiny",         # < 1s response
        "high_accuracy": "large-v3",   # Best quality
    }
    return models.get(use_case, "small")

# Usage
model_name = select_model("meetings")
model = whisper.load_model(model_name)
```

## Installation & Setup

### Method 1: Official OpenAI Whisper

```bash
pip install -U openai-whisper

# Install ffmpeg (required for audio processing)
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows (use chocolatey)
choco install ffmpeg
```

Basic usage:
```python
import whisper

# Load model
model = whisper.load_model("base")

# Transcribe
result = model.transcribe("audio.mp3")
print(result["text"])
```

### Method 2: Hugging Face Transformers

```bash
pip install transformers torch accelerate
```

```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

# Load model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Transcribe
import librosa

audio, sr = librosa.load("audio.mp3", sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
inputs = inputs.to(device)

generated_ids = model.generate(inputs["input_features"])
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(transcription)
```

### Method 3: faster-whisper (Recommended for Production)

**CTranslate2-based implementation, 4x faster with same accuracy**

```bash
pip install faster-whisper
```

```python
from faster_whisper import WhisperModel

# Load model (runs on GPU by default)
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# Or CPU
# model = WhisperModel("large-v3", device="cpu", compute_type="int8")

# Transcribe
segments, info = model.transcribe("audio.mp3", language="en")

print(f"Detected language: {info.language} (probability: {info.language_probability})")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

**Benefits:**
- 4x faster than openai-whisper
- Lower memory usage
- Same accuracy
- Better batching
- Production-ready

### Method 4: whisperX (State-of-the-art alignment)

**Adds word-level timestamps and speaker diarization**

```bash
pip install whisperx
```

```python
import whisperx

device = "cuda"
batch_size = 16
compute_type = "float16"

# Load model
model = whisperx.load_model("large-v3", device, compute_type=compute_type)

# Transcribe
audio = whisperx.load_audio("audio.mp3")
result = model.transcribe(audio, batch_size=batch_size)

# Align (word-level timestamps)
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"],
    device=device
)
result = whisperx.align(
    result["segments"],
    model_a,
    metadata,
    audio,
    device
)

# Diarization (speaker identification)
diarize_model = whisperx.DiarizationPipeline(
    use_auth_token="YOUR_HF_TOKEN",
    device=device
)
diarize_segments = diarize_model(audio)
result = whisperx.assign_word_speakers(diarize_segments, result)

# Print with speakers
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] Speaker {segment.get('speaker', 'Unknown')}: {segment['text']}")
```

### Method 5: OpenAI API (Cloud)

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# Transcribe
with open("audio.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )
    print(transcript)

# With timestamps
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="verbose_json",
    timestamp_granularities=["word", "segment"]
)

# Translation (to English)
translation = client.audio.translations.create(
    model="whisper-1",
    file=audio_file
)
```

### Method 6: Command Line

```bash
# Install
pip install openai-whisper

# Transcribe
whisper audio.mp3 --model medium --language en

# With options
whisper audio.mp3 \
  --model large-v3 \
  --language en \
  --task transcribe \
  --output_format srt \
  --output_dir ./transcripts

# Multiple files
whisper *.mp3 --model small --language auto
```

## Basic Usage

### Simple Transcription

```python
import whisper

# Load model
model = whisper.load_model("base")

# Transcribe
result = model.transcribe("audio.mp3")

# Get text
print(result["text"])

# Get segments with timestamps
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")
```

### With Language Specification

```python
# Specify language (faster than auto-detection)
result = model.transcribe("audio.mp3", language="en")

# Spanish
result = model.transcribe("audio.mp3", language="es")

# Japanese
result = model.transcribe("audio.mp3", language="ja")
```

### Translation to English

```python
# Translate any language to English
result = model.transcribe("audio_spanish.mp3", task="translate")
print(result["text"])  # Output in English
```

### From Different Audio Sources

```python
import whisper

model = whisper.load_model("base")

# From file
result = model.transcribe("audio.mp3")

# From URL
import urllib.request
url = "https://example.com/audio.mp3"
urllib.request.urlretrieve(url, "temp.mp3")
result = model.transcribe("temp.mp3")

# From numpy array
import numpy as np
audio_array = np.load("audio.npy")
result = model.transcribe(audio_array)

# From microphone (real-time)
import sounddevice as sd

duration = 5  # seconds
sample_rate = 16000
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()
result = model.transcribe(audio.flatten())
```

### Detailed Output

```python
result = model.transcribe("audio.mp3", verbose=True)

# Access detailed information
print(f"Language: {result['language']}")
print(f"Text: {result['text']}")

# Segments with more info
for segment in result['segments']:
    print(f"ID: {segment['id']}")
    print(f"Start: {segment['start']:.2f}s")
    print(f"End: {segment['end']:.2f}s")
    print(f"Text: {segment['text']}")
    print(f"Tokens: {segment['tokens']}")
    print(f"Temperature: {segment['temperature']}")
    print(f"Avg Logprob: {segment['avg_logprob']}")
    print(f"Compression Ratio: {segment['compression_ratio']}")
    print(f"No Speech Prob: {segment['no_speech_prob']}")
    print("---")
```

### Output Formats

```python
# Plain text
result = model.transcribe("audio.mp3")
text = result["text"]

# SRT (SubRip)
def to_srt(result):
    srt = ""
    for i, segment in enumerate(result["segments"], start=1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()
        srt += f"{i}\n{start} --> {end}\n{text}\n\n"
    return srt

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

srt_output = to_srt(result)
with open("subtitles.srt", "w") as f:
    f.write(srt_output)

# VTT (WebVTT)
def to_vtt(result):
    vtt = "WEBVTT\n\n"
    for segment in result["segments"]:
        start = format_vtt_timestamp(segment["start"])
        end = format_vtt_timestamp(segment["end"])
        text = segment["text"].strip()
        vtt += f"{start} --> {end}\n{text}\n\n"
    return vtt

# JSON
import json
with open("transcript.json", "w") as f:
    json.dump(result, f, indent=2)
```

## Fine-tuning

### When to Fine-tune

✅ **Good use cases:**
- Domain-specific vocabulary (medical, legal, technical)
- Specific accents or dialects
- Low-resource languages
- Custom output format
- Improved accuracy for your use case

❌ **Not needed for:**
- General transcription
- Standard languages/accents
- When base model works well

### Prepare Dataset

```python
# Dataset format: audio files + transcripts
# Directory structure:
# data/
#   train/
#     audio1.mp3
#     audio1.txt
#     audio2.mp3
#     audio2.txt
#   test/
#     audio1.mp3
#     audio1.txt

from datasets import Dataset, Audio
import os

def load_data(data_dir):
    audio_files = []
    transcripts = []

    for filename in os.listdir(data_dir):
        if filename.endswith('.mp3'):
            audio_path = os.path.join(data_dir, filename)
            txt_path = audio_path.replace('.mp3', '.txt')

            if os.path.exists(txt_path):
                audio_files.append(audio_path)
                with open(txt_path, 'r') as f:
                    transcripts.append(f.read().strip())

    return Dataset.from_dict({
        "audio": audio_files,
        "transcription": transcripts
    }).cast_column("audio", Audio(sampling_rate=16000))

train_dataset = load_data("data/train")
test_dataset = load_data("data/test")
```

### Fine-tune with Hugging Face

```bash
pip install transformers datasets accelerate evaluate jiwer
```

```python
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

# Load model and processor
model_id = "openai/whisper-small"
model = WhisperForConditionalGeneration.from_pretrained(model_id)
processor = WhisperProcessor.from_pretrained(model_id)

# Prepare data
def prepare_dataset(batch):
    audio = batch["audio"]

    # Compute input features
    batch["input_features"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Encode transcription
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids

    return batch

train_dataset = train_dataset.map(
    prepare_dataset,
    remove_columns=train_dataset.column_names
)

# Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=50,
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    fp16=True,
    predict_with_generate=True,
    generation_max_length=225,
)

# Metrics
import evaluate
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# Train
trainer.train()

# Save
model.save_pretrained("./whisper-finetuned-final")
processor.save_pretrained("./whisper-finetuned-final")
```

### LoRA Fine-tuning (Memory Efficient)

```bash
pip install peft
```

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# LoRA configuration
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

# Prepare model
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 2.6M / total: 244M (~1%)

# Train as before
trainer = Seq2SeqTrainer(...)
trainer.train()

# Merge and save
model = model.merge_and_unload()
model.save_pretrained("./whisper-lora-merged")
```

### Use Fine-tuned Model

```python
from transformers import pipeline

# Load fine-tuned model
pipe = pipeline(
    "automatic-speech-recognition",
    model="./whisper-finetuned-final",
    device="cuda:0"
)

# Transcribe
result = pipe("audio.mp3")
print(result["text"])
```

## Common Patterns

### Pattern 1: Language Detection

```python
import whisper

model = whisper.load_model("base")

# Detect language
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)

mel = whisper.log_mel_spectrogram(audio).to(model.device)
_, probs = model.detect_language(mel)

detected_language = max(probs, key=probs.get)
print(f"Detected language: {detected_language} (confidence: {probs[detected_language]:.2%})")

# All probabilities
for lang, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{lang}: {prob:.2%}")
```

### Pattern 2: Batch Processing

```python
import whisper
import os
from pathlib import Path

model = whisper.load_model("small")

def transcribe_directory(input_dir, output_dir, language="en"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    audio_files = list(Path(input_dir).glob("*.mp3")) + \
                  list(Path(input_dir).glob("*.wav"))

    for audio_file in audio_files:
        print(f"Transcribing: {audio_file.name}")

        result = model.transcribe(
            str(audio_file),
            language=language,
            verbose=False
        )

        # Save transcript
        output_file = Path(output_dir) / f"{audio_file.stem}.txt"
        with open(output_file, "w") as f:
            f.write(result["text"])

        # Save SRT
        srt_file = Path(output_dir) / f"{audio_file.stem}.srt"
        with open(srt_file, "w") as f:
            f.write(to_srt(result))

        print(f"✓ Saved: {output_file.name}")

# Usage
transcribe_directory("./audio_files", "./transcripts", language="en")
```

### Pattern 3: Real-time Streaming

```python
import whisper
import pyaudio
import numpy as np
import queue
import threading

model = whisper.load_model("base")

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5  # Process every 5 seconds

audio_queue = queue.Queue()

def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (in_data, pyaudio.paContinue)

def transcribe_stream():
    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=audio_callback
    )

    stream.start_stream()

    audio_buffer = []

    try:
        while stream.is_active():
            if not audio_queue.empty():
                data = audio_queue.get()
                audio_buffer.append(np.frombuffer(data, dtype=np.int16))

                # Process when buffer reaches target length
                if len(audio_buffer) >= (RATE * RECORD_SECONDS) // CHUNK:
                    audio_data = np.concatenate(audio_buffer).astype(np.float32) / 32768.0

                    result = model.transcribe(audio_data)
                    print(f"Transcription: {result['text']}")

                    audio_buffer = []

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# Run
transcribe_stream()
```

### Pattern 4: Video Subtitles

```python
import whisper
import subprocess
import os

def generate_subtitles(video_file, output_srt, model_size="small", language="en"):
    # Extract audio from video
    audio_file = "temp_audio.mp3"
    subprocess.run([
        "ffmpeg", "-i", video_file,
        "-vn", "-acodec", "mp3",
        "-y", audio_file
    ], check=True)

    # Transcribe
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_file, language=language)

    # Generate SRT
    with open(output_srt, "w") as f:
        f.write(to_srt(result))

    # Clean up
    os.remove(audio_file)

    print(f"✓ Subtitles saved to: {output_srt}")

# Burn subtitles into video
def burn_subtitles(video_file, srt_file, output_file):
    subprocess.run([
        "ffmpeg", "-i", video_file,
        "-vf", f"subtitles={srt_file}",
        "-c:a", "copy",
        "-y", output_file
    ], check=True)

    print(f"✓ Video with subtitles: {output_file}")

# Usage
generate_subtitles("video.mp4", "subtitles.srt", model_size="medium", language="en")
burn_subtitles("video.mp4", "subtitles.srt", "video_with_subs.mp4")
```

### Pattern 5: Timestamp-based Search

```python
def search_in_transcript(audio_file, search_terms, model_size="base"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_file)

    matches = []

    for segment in result["segments"]:
        text = segment["text"].lower()

        for term in search_terms:
            if term.lower() in text:
                matches.append({
                    "term": term,
                    "timestamp": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"]
                })

    return matches

# Usage
results = search_in_transcript(
    "meeting.mp3",
    ["budget", "deadline", "milestone"]
)

for match in results:
    print(f"[{match['timestamp']:.1f}s] Found '{match['term']}': {match['text']}")
```

### Pattern 6: Meeting Transcription

```python
import whisper
from datetime import datetime, timedelta

def transcribe_meeting(audio_file, meeting_name=None):
    model = whisper.load_model("medium")

    print("Transcribing meeting...")
    result = model.transcribe(audio_file, language="en")

    # Generate formatted transcript
    meeting_name = meeting_name or "Meeting"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    transcript = f"# {meeting_name}\n"
    transcript += f"Date: {timestamp}\n"
    transcript += f"Duration: {result['segments'][-1]['end']:.0f} seconds\n\n"
    transcript += "## Transcript\n\n"

    for segment in result["segments"]:
        time_str = str(timedelta(seconds=int(segment["start"])))
        transcript += f"**[{time_str}]** {segment['text']}\n\n"

    # Save
    output_file = f"{meeting_name}_{datetime.now().strftime('%Y%m%d')}.md"
    with open(output_file, "w") as f:
        f.write(transcript)

    print(f"✓ Meeting transcript saved: {output_file}")
    return result

# Usage
transcribe_meeting("team_meeting.mp3", "Weekly Team Sync")
```

## Advanced Operations

### Voice Activity Detection (VAD)

```python
from faster_whisper import WhisperModel
import torch

model = WhisperModel("base", device="cuda")

# Transcribe with VAD
segments, info = model.transcribe(
    "audio.mp3",
    vad_filter=True,
    vad_parameters=dict(
        threshold=0.5,
        min_speech_duration_ms=250,
        max_speech_duration_s=float('inf'),
        min_silence_duration_ms=2000,
        window_size_samples=1024,
        speech_pad_ms=400
    )
)

for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
```

### Initial Prompt Engineering

```python
# Use initial_prompt to guide transcription style
result = model.transcribe(
    "audio.mp3",
    initial_prompt="This is a technical discussion about machine learning, neural networks, and artificial intelligence."
)

# For proper nouns and terminology
result = model.transcribe(
    "medical.mp3",
    initial_prompt="Medical terminology: MRI, CT scan, diagnosis, prognosis, pharmacology"
)

# For formatting
result = model.transcribe(
    "interview.mp3",
    initial_prompt="Q: Question text\nA: Answer text"
)

# For specific style
result = model.transcribe(
    "presentation.mp3",
    initial_prompt="Professional presentation with proper punctuation and capitalization."
)
```

### Conditioning on Previous Text

```python
# Process long audio in chunks with context
def transcribe_with_context(audio_file, chunk_duration=30, overlap=5):
    model = whisper.load_model("medium")
    audio = whisper.load_audio(audio_file)
    sample_rate = 16000

    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap * sample_rate

    transcripts = []
    previous_text = ""

    for i in range(0, len(audio), chunk_samples - overlap_samples):
        chunk = audio[i:i + chunk_samples]

        # Use previous text as context
        result = model.transcribe(
            chunk,
            initial_prompt=previous_text[-200:] if previous_text else None
        )

        transcripts.append(result["text"])
        previous_text = result["text"]

    return " ".join(transcripts)

# Usage
full_transcript = transcribe_with_context("long_audio.mp3")
```

### Temperature Fallback

```python
# Use multiple temperatures for difficult audio
result = model.transcribe(
    "difficult_audio.mp3",
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
)

# Check which temperature was used
for segment in result["segments"]:
    print(f"Temperature used: {segment['temperature']}")
    print(f"Text: {segment['text']}")
```

### Beam Search Tuning

```python
# Adjust beam search for better accuracy
result = model.transcribe(
    "audio.mp3",
    beam_size=10,          # Default: 5
    best_of=5,             # Default: 5
    patience=2.0,          # Default: 1.0
)

# For faster inference with acceptable quality
result = model.transcribe(
    "audio.mp3",
    beam_size=1,  # Greedy decoding
)
```

### Compression Ratio Filtering

```python
# Filter out hallucinations using compression ratio
def transcribe_with_filtering(audio_file, compression_threshold=2.4):
    model = whisper.load_model("medium")
    result = model.transcribe(audio_file)

    filtered_segments = []

    for segment in result["segments"]:
        if segment["compression_ratio"] < compression_threshold:
            filtered_segments.append(segment)
        else:
            print(f"Filtered segment (compression: {segment['compression_ratio']:.2f}): {segment['text']}")

    return filtered_segments

segments = transcribe_with_filtering("audio.mp3")
```

### No Speech Probability Filtering

```python
# Remove segments without speech
def filter_no_speech(result, threshold=0.6):
    return [
        segment for segment in result["segments"]
        if segment["no_speech_prob"] < threshold
    ]

result = model.transcribe("audio.mp3")
speech_segments = filter_no_speech(result, threshold=0.5)
```

### Word-level Timestamps

```python
# Using faster-whisper for word-level timestamps
from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cuda")

segments, info = model.transcribe(
    "audio.mp3",
    word_timestamps=True
)

for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")

    for word in segment.words:
        print(f"  [{word.start:.2f}s - {word.end:.2f}s] {word.word}")
```

## Optimization

### Speed Optimization

```python
# 1. Use faster-whisper (4x faster)
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cuda", compute_type="float16")
segments, _ = model.transcribe("audio.mp3")

# 2. Use smaller model
model = whisper.load_model("tiny")  # 32x faster than large

# 3. Reduce beam size
result = model.transcribe("audio.mp3", beam_size=1)

# 4. Skip language detection
result = model.transcribe("audio.mp3", language="en")

# 5. Lower temperature
result = model.transcribe("audio.mp3", temperature=0.0)

# 6. Batch processing with faster-whisper
model = WhisperModel("base")
audio_files = ["audio1.mp3", "audio2.mp3", "audio3.mp3"]

for audio_file in audio_files:
    segments, _ = model.transcribe(audio_file)
    for segment in segments:
        print(segment.text)
```

### Memory Optimization

```python
import torch
import gc

# 1. Use smaller model
model = whisper.load_model("small")

# 2. Process in chunks
def transcribe_large_file(audio_file, chunk_duration=30):
    model = whisper.load_model("base")
    audio = whisper.load_audio(audio_file)
    sample_rate = 16000

    transcripts = []
    chunk_samples = chunk_duration * sample_rate

    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        result = model.transcribe(chunk)
        transcripts.append(result["text"])

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

    return " ".join(transcripts)

# 3. Use int8 quantization (CPU)
from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cpu", compute_type="int8")

# 4. Enable gradient checkpointing (for training)
model.gradient_checkpointing_enable()
```

### Quality vs Speed Trade-offs

```python
import time

def benchmark_models(audio_file):
    models = ["tiny", "base", "small", "medium", "large-v3"]
    results = []

    for model_name in models:
        print(f"Testing {model_name}...")
        model = whisper.load_model(model_name)

        start = time.time()
        result = model.transcribe(audio_file)
        duration = time.time() - start

        results.append({
            "model": model_name,
            "duration": duration,
            "text": result["text"]
        })

        del model
        torch.cuda.empty_cache()

    return results

# Analyze results
results = benchmark_models("test_audio.mp3")
for r in results:
    print(f"{r['model']}: {r['duration']:.2f}s")
```

### Batched Inference

```python
from faster_whisper import WhisperModel
import concurrent.futures

model = WhisperModel("base", device="cuda")

def transcribe_file(audio_file):
    segments, _ = model.transcribe(audio_file)
    return " ".join([segment.text for segment in segments])

# Parallel processing
audio_files = ["audio1.mp3", "audio2.mp3", "audio3.mp3"]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(transcribe_file, audio_files))

for audio_file, result in zip(audio_files, results):
    print(f"{audio_file}: {result[:100]}...")
```

### GPU Optimization

```python
import torch

# 1. Use float16 (half precision)
model = whisper.load_model("medium").half().cuda()

# 2. Enable TensorFloat32 (Ampere GPUs)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 3. Use faster-whisper with optimal settings
from faster_whisper import WhisperModel

model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16",
    num_workers=4
)

# 4. Pin memory for faster data transfer
# (Handled automatically by faster-whisper)
```

## Deployment

### REST API with FastAPI

```python
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import whisper
import tempfile
import os

app = FastAPI()

# Load model at startup
model = whisper.load_model("base")

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form("en"),
    task: str = Form("transcribe")
):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        # Transcribe
        result = model.transcribe(
            temp_path,
            language=language,
            task=task
        )

        return JSONResponse({
            "text": result["text"],
            "language": result["language"],
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"]
                }
                for seg in result["segments"]
            ]
        })

    finally:
        # Clean up
        os.unlink(temp_path)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run: uvicorn app:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Install Python and ffmpeg
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application
COPY app.py .

# Download model at build time (optional)
RUN python3 -c "import whisper; whisper.load_model('base')"

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t whisper-api .
docker run -p 8000:8000 --gpus all whisper-api
```

### Worker Queue with Celery

```python
# tasks.py
from celery import Celery
import whisper

app = Celery('tasks', broker='redis://localhost:6379/0')

model = whisper.load_model("base")

@app.task
def transcribe_task(audio_path, language="en"):
    result = model.transcribe(audio_path, language=language)
    return {
        "text": result["text"],
        "segments": result["segments"]
    }

# client.py
from tasks import transcribe_task

# Submit task
task = transcribe_task.delay("audio.mp3", language="en")

# Get result
result = task.get(timeout=300)
print(result["text"])
```

### Serverless (AWS Lambda)

```python
# lambda_function.py
import json
import boto3
import whisper
import tempfile

s3 = boto3.client('s3')
model = whisper.load_model("tiny")  # Use tiny for lambda

def lambda_handler(event, context):
    # Get audio from S3
    bucket = event['bucket']
    key = event['key']

    with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_file:
        s3.download_fileobj(bucket, key, temp_file)
        temp_file.flush()

        # Transcribe
        result = model.transcribe(temp_file.name)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'text': result['text']
            })
        }
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: whisper-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: whisper-api
  template:
    metadata:
      labels:
        app: whisper-api
    spec:
      containers:
      - name: whisper
        image: whisper-api:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "8Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: whisper-service
spec:
  selector:
    app: whisper-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Advanced Techniques

### Speaker Diarization

```python
# Using WhisperX with pyannote
import whisperx

device = "cuda"
audio_file = "meeting.mp3"

# 1. Transcribe
model = whisperx.load_model("large-v3", device, compute_type="float16")
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=16)

# 2. Align
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"],
    device=device
)
result = whisperx.align(result["segments"], model_a, metadata, audio, device)

# 3. Diarize
from pyannote.audio import Pipeline

diarize_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token="YOUR_HF_TOKEN"
)
diarize_segments = diarize_pipeline(audio_file)

# 4. Assign speakers
result = whisperx.assign_word_speakers(diarize_segments, result)

# 5. Format output
for segment in result["segments"]:
    speaker = segment.get("speaker", "UNKNOWN")
    print(f"[{segment['start']:.1f}s - {segment['end']:.1f}s] {speaker}: {segment['text']}")
```

### Multi-language Detection and Switching

```python
def transcribe_multilingual(audio_file):
    model = whisper.load_model("large-v3")

    # Initial language detection
    audio = whisper.load_audio(audio_file)
    audio_segment = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio_segment).to(model.device)

    _, probs = model.detect_language(mel)
    primary_language = max(probs, key=probs.get)

    print(f"Primary language: {primary_language}")

    # Transcribe with language switching detection
    result = model.transcribe(
        audio_file,
        task="transcribe",
        verbose=True
    )

    # Detect language per segment
    segments_with_language = []

    for segment in result["segments"]:
        segment_audio = audio[int(segment["start"] * 16000):int(segment["end"] * 16000)]
        segment_audio = whisper.pad_or_trim(segment_audio)
        mel = whisper.log_mel_spectrogram(segment_audio).to(model.device)

        _, seg_probs = model.detect_language(mel)
        seg_language = max(seg_probs, key=seg_probs.get)

        segments_with_language.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "language": seg_language,
            "confidence": seg_probs[seg_language]
        })

    return segments_with_language

# Usage
segments = transcribe_multilingual("multilingual_audio.mp3")
for seg in segments:
    print(f"[{seg['language']}] {seg['text']}")
```

### Custom Vocabulary and Spelling

```python
def transcribe_with_vocabulary(audio_file, vocabulary):
    """
    Use initial_prompt to guide recognition of specific terms
    """
    model = whisper.load_model("medium")

    # Create prompt with vocabulary
    vocab_prompt = "Vocabulary: " + ", ".join(vocabulary) + "."

    result = model.transcribe(
        audio_file,
        initial_prompt=vocab_prompt
    )

    return result

# Usage
custom_vocab = [
    "TensorFlow", "PyTorch", "CUDA", "GPU",
    "Kubernetes", "Docker", "CI/CD",
    "API", "REST", "GraphQL"
]

result = transcribe_with_vocabulary("tech_talk.mp3", custom_vocab)
```

### Noise Reduction Preprocessing

```python
import noisereduce as nr
import librosa
import soundfile as sf
import whisper

def transcribe_with_noise_reduction(audio_file):
    # Load audio
    audio, sr = librosa.load(audio_file, sr=16000)

    # Reduce noise
    reduced_noise = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)

    # Save temporarily
    temp_file = "temp_cleaned.wav"
    sf.write(temp_file, reduced_noise, sr)

    # Transcribe
    model = whisper.load_model("base")
    result = model.transcribe(temp_file)

    # Clean up
    import os
    os.remove(temp_file)

    return result

# Usage
result = transcribe_with_noise_reduction("noisy_audio.mp3")
```

### Audio Normalization

```python
from pydub import AudioSegment
from pydub.effects import normalize
import whisper

def transcribe_with_normalization(audio_file):
    # Load and normalize
    audio = AudioSegment.from_file(audio_file)
    normalized_audio = normalize(audio)

    # Export
    temp_file = "temp_normalized.mp3"
    normalized_audio.export(temp_file, format="mp3")

    # Transcribe
    model = whisper.load_model("base")
    result = model.transcribe(temp_file)

    # Clean up
    import os
    os.remove(temp_file)

    return result
```

### Transcript Post-processing

```python
import re

def post_process_transcript(text):
    """
    Clean up and format transcript
    """
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Capitalize sentences
    text = '. '.join(sentence.capitalize() for sentence in text.split('. '))

    # Fix common errors
    replacements = {
        ' i ': ' I ',
        "im ": "I'm ",
        "ive ": "I've ",
        "youre ": "you're ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove filler words (optional)
    fillers = ['um', 'uh', 'er', 'ah']
    for filler in fillers:
        text = re.sub(rf'\b{filler}\b', '', text, flags=re.IGNORECASE)

    # Clean up spacing
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Usage
result = model.transcribe("audio.mp3")
clean_text = post_process_transcript(result["text"])
```

### Confidence Scoring

```python
def transcribe_with_confidence(audio_file):
    model = whisper.load_model("medium")
    result = model.transcribe(audio_file, verbose=False)

    segments_with_confidence = []

    for segment in result["segments"]:
        # Average log probability as confidence
        avg_logprob = segment["avg_logprob"]
        confidence = np.exp(avg_logprob)  # Convert to probability

        segments_with_confidence.append({
            "text": segment["text"],
            "start": segment["start"],
            "end": segment["end"],
            "confidence": confidence,
            "no_speech_prob": segment["no_speech_prob"]
        })

    return segments_with_confidence

# Usage
segments = transcribe_with_confidence("audio.mp3")
for seg in segments:
    if seg["confidence"] > 0.8:
        print(f"HIGH CONF: {seg['text']}")
    else:
        print(f"LOW CONF: {seg['text']} (review needed)")
```

## Integration

### With LangChain

```python
from langchain.document_loaders import WhisperAudioLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI

# Transcribe audio
loader = WhisperAudioLoader("meeting.mp3")
documents = loader.load()

# Summarize transcript
llm = OpenAI(temperature=0)
chain = load_summarize_chain(llm, chain_type="map_reduce")
summary = chain.run(documents)

print(summary)
```

### With Streamlit

```python
import streamlit as st
import whisper
import tempfile

st.title("Whisper Transcription App")

# Upload audio
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Model selection
    model_size = st.selectbox("Model size", ["tiny", "base", "small", "medium", "large-v3"])

    # Language selection
    language = st.selectbox("Language", ["auto", "en", "es", "fr", "de", "ja", "zh"])

    if st.button("Transcribe"):
        with st.spinner("Loading model..."):
            model = whisper.load_model(model_size)

        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        with st.spinner("Transcribing..."):
            result = model.transcribe(
                temp_path,
                language=None if language == "auto" else language
            )

        # Display results
        st.subheader("Transcript")
        st.write(result["text"])

        st.subheader("Segments")
        for segment in result["segments"]:
            st.write(f"**[{segment['start']:.1f}s - {segment['end']:.1f}s]** {segment['text']}")

        # Download button
        st.download_button(
            "Download Transcript",
            result["text"],
            file_name="transcript.txt"
        )
```

### With Flask

```python
from flask import Flask, request, jsonify, render_template
import whisper
import os

app = Flask(__name__)
model = whisper.load_model("base")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    language = request.form.get("language", "en")

    # Save file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        # Transcribe
        result = model.transcribe(filepath, language=language)

        return jsonify({
            "text": result["text"],
            "segments": result["segments"]
        })

    finally:
        # Clean up
        os.remove(filepath)

if __name__ == "__main__":
    app.run(debug=True)
```

### With Discord Bot

```python
import discord
from discord.ext import commands
import whisper
import os

bot = commands.Bot(command_prefix="!")
model = whisper.load_model("base")

@bot.command()
async def transcribe(ctx):
    """Transcribe an attached audio file"""
    if not ctx.message.attachments:
        await ctx.send("Please attach an audio file!")
        return

    attachment = ctx.message.attachments[0]

    # Download file
    filepath = f"temp_{attachment.filename}"
    await attachment.save(filepath)

    await ctx.send("Transcribing...")

    try:
        # Transcribe
        result = model.transcribe(filepath)

        # Send result (split if too long)
        text = result["text"]
        if len(text) > 2000:
            for i in range(0, len(text), 2000):
                await ctx.send(text[i:i+2000])
        else:
            await ctx.send(text)

    finally:
        os.remove(filepath)

bot.run("YOUR_BOT_TOKEN")
```

### With Telegram Bot

```python
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import whisper
import os

model = whisper.load_model("base")

async def transcribe_audio(update: Update, context):
    """Handle voice messages"""
    if update.message.voice:
        file = await update.message.voice.get_file()
    elif update.message.audio:
        file = await update.message.audio.get_file()
    else:
        await update.message.reply_text("Please send an audio file or voice message!")
        return

    # Download
    filepath = "temp_audio.ogg"
    await file.download_to_drive(filepath)

    await update.message.reply_text("Transcribing...")

    try:
        # Transcribe
        result = model.transcribe(filepath)
        await update.message.reply_text(result["text"])

    finally:
        os.remove(filepath)

# Create application
app = Application.builder().token("YOUR_BOT_TOKEN").build()

# Add handlers
app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, transcribe_audio))

# Run
app.run_polling()
```

## Best Practices

### 1. Model Selection

```python
# Production decision tree
def choose_model(requirements):
    if requirements["latency"] == "realtime":
        return "tiny"
    elif requirements["accuracy"] == "high" and requirements["resources"] == "available":
        return "large-v3"
    elif requirements["accuracy"] == "medium":
        return "small" if requirements["latency"] == "fast" else "medium"
    else:
        return "base"

# Example
model_name = choose_model({
    "latency": "moderate",
    "accuracy": "high",
    "resources": "available"
})
```

### 2. Error Handling

```python
def safe_transcribe(audio_file, model_size="base", max_retries=3):
    """Robust transcription with error handling"""
    import logging

    for attempt in range(max_retries):
        try:
            model = whisper.load_model(model_size)
            result = model.transcribe(audio_file)
            return result

        except FileNotFoundError:
            logging.error(f"Audio file not found: {audio_file}")
            raise

        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.warning(f"OOM on attempt {attempt + 1}, trying smaller model")
                model_size = {
                    "large-v3": "medium",
                    "medium": "small",
                    "small": "base",
                    "base": "tiny"
                }.get(model_size, "tiny")
                continue
            raise

        except Exception as e:
            logging.error(f"Transcription failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise

    return None
```

### 3. Audio Preprocessing

```python
def preprocess_audio(audio_file, output_file="processed.wav"):
    """Prepare audio for optimal transcription"""
    from pydub import AudioSegment

    # Load audio
    audio = AudioSegment.from_file(audio_file)

    # Convert to mono
    audio = audio.set_channels(1)

    # Set sample rate to 16kHz
    audio = audio.set_frame_rate(16000)

    # Normalize volume
    from pydub.effects import normalize
    audio = normalize(audio)

    # Remove silence from start/end
    from pydub.silence import detect_leading_silence

    start_trim = detect_leading_silence(audio)
    end_trim = detect_leading_silence(audio.reverse())

    duration = len(audio)
    audio = audio[start_trim:duration-end_trim]

    # Export
    audio.export(output_file, format="wav")

    return output_file
```

### 4. Monitoring and Logging

```python
import logging
import time
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_transcription(func):
    """Decorator to monitor transcription performance"""
    @wraps(func)
    def wrapper(audio_file, *args, **kwargs):
        logger.info(f"Starting transcription: {audio_file}")
        start_time = time.time()

        try:
            result = func(audio_file, *args, **kwargs)

            duration = time.time() - start_time
            text_length = len(result.get("text", ""))

            logger.info(f"Transcription completed in {duration:.2f}s")
            logger.info(f"Generated {text_length} characters")

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    return wrapper

@monitor_transcription
def transcribe(audio_file, model_size="base"):
    model = whisper.load_model(model_size)
    return model.transcribe(audio_file)
```

### 5. Caching

```python
import hashlib
import json
import os

CACHE_DIR = ".whisper_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_file_hash(filepath):
    """Get MD5 hash of file"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def transcribe_with_cache(audio_file, model_size="base"):
    """Transcribe with result caching"""
    # Generate cache key
    file_hash = get_file_hash(audio_file)
    cache_key = f"{file_hash}_{model_size}"
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")

    # Check cache
    if os.path.exists(cache_file):
        print("Loading from cache...")
        with open(cache_file, 'r') as f:
            return json.load(f)

    # Transcribe
    print("Transcribing...")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_file)

    # Save to cache
    with open(cache_file, 'w') as f:
        json.dump(result, f)

    return result
```

### 6. Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def transcribe_single(args):
    """Transcribe single file (for parallel processing)"""
    audio_file, model_size = args
    import whisper

    model = whisper.load_model(model_size)
    result = model.transcribe(audio_file)

    return {
        "file": audio_file,
        "text": result["text"]
    }

def transcribe_parallel(audio_files, model_size="base", max_workers=None):
    """Transcribe multiple files in parallel"""
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    args = [(f, model_size) for f in audio_files]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(transcribe_single, args))

    return results

# Usage
audio_files = ["audio1.mp3", "audio2.mp3", "audio3.mp3"]
results = transcribe_parallel(audio_files, model_size="base")
```

### 7. Quality Assurance

```python
def validate_transcription(result, min_confidence=0.7):
    """Validate transcription quality"""
    issues = []

    # Check for hallucination indicators
    for segment in result["segments"]:
        # High compression ratio = possible hallucination
        if segment["compression_ratio"] > 2.4:
            issues.append(f"High compression at {segment['start']:.1f}s")

        # High no_speech_prob = not actually speech
        if segment["no_speech_prob"] > 0.6:
            issues.append(f"Low speech probability at {segment['start']:.1f}s")

        # Low confidence
        confidence = np.exp(segment["avg_logprob"])
        if confidence < min_confidence:
            issues.append(f"Low confidence at {segment['start']:.1f}s: {confidence:.2%}")

    if issues:
        print("⚠️  Quality issues detected:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Quality check passed")

    return len(issues) == 0
```

## Resources

### Official
- [Whisper GitHub](https://github.com/openai/whisper)
- [OpenAI Blog Post](https://openai.com/blog/whisper/)
- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [Model Card](https://github.com/openai/whisper/blob/main/model-card.md)

### Alternative Implementations
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - CTranslate2 implementation (4x faster)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - C++ implementation
- [whisperX](https://github.com/m-bain/whisperX) - Word-level timestamps and diarization
- [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) - Optimized with Flash Attention

### Tools
- [Hugging Face Space](https://huggingface.co/spaces/openai/whisper)
- [Replicate API](https://replicate.com/openai/whisper)
- [Whisper Web](https://whisper.ggerganov.com/) - Browser-based

### Fine-tuning
- [Hugging Face Tutorial](https://huggingface.co/blog/fine-tune-whisper)
- [Fine-tuning notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/pytorch/speech_recognition.ipynb)

### Community
- [Whisper Discussions](https://github.com/openai/whisper/discussions)
- r/OpenAI
- Hugging Face Forums

### Benchmarks
- [WER Benchmarks](https://github.com/openai/whisper#available-models-and-languages)
- [Language Performance](https://github.com/openai/whisper/blob/main/language-breakdown.svg)

## Conclusion

Whisper represents a breakthrough in automatic speech recognition, offering:

- **Robustness**: Works across accents, noise, and technical language
- **Multilingual**: 99 languages with strong performance
- **Flexibility**: Multiple model sizes for different requirements
- **Open Source**: MIT license enables wide adoption

### Key Takeaways

1. **Start with the right model**:
   - Tiny/Base: Prototyping, real-time
   - Small/Medium: Production balance
   - Large-v3: Maximum accuracy

2. **Optimize for production**:
   - Use faster-whisper for 4x speedup
   - Implement caching and batching
   - Add error handling and monitoring

3. **Fine-tune when needed**:
   - Domain-specific vocabulary
   - Specialized accents
   - Improved accuracy

4. **Leverage advanced features**:
   - Word-level timestamps
   - Speaker diarization
   - Language detection

5. **Consider trade-offs**:
   - Speed vs accuracy
   - Memory vs quality
   - Cost vs performance

Whisper has democratized speech recognition, making state-of-the-art ASR accessible to everyone. Whether building a voice assistant, transcription service, or accessibility tool, Whisper provides the foundation for robust speech-to-text applications.
