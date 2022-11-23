import os
import json

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.functional import word_error_rate
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from utils import get_device
from spectral_gate import spectral_model
from autoencoder import AudioDataset, FRAMERATE


BATCH_SIZE = 16
FRAMERATE = int(16000)
ENGLISH_TRANSCRIBE_TOKEN_IDX = 50258


def transcribe_audio(model, processor, audio, device="cuda"):
    """Generates takes from raw wave data"""

    input_features = processor(list(audio.numpy()), sampling_rate=FRAMERATE, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        logits = model.generate(input_features, decoder_input_ids=torch.tensor([[ENGLISH_TRANSCRIBE_TOKEN_IDX] for _ in range(len(audio))]).to(device), max_length=64)
    texts = processor.batch_decode(logits)
    return [
        sentence.split("<|notimestamps|>")[1].replace("<|endoftext|>", "").strip()
    for sentence in texts]


def evaluate_models(clean_samples, noisy_samples, models, asr_model, asr_processor, device="cuda"):
    dataset = AudioDataset(clean_samples, noisy_samples)
    dataset.files = dataset.files[:49]
    eval_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    transcriptions = {}

    output_dir = "/".join(clean_samples.split("/")[:-1])

    for model_name in models:
        model_val_path = os.path.join(output_dir, f"test_{model_name}")
        if not os.path.exists(model_val_path):
            os.mkdir(model_val_path)

    for data in tqdm(eval_loader):
        original = data['original'].to(device)
        original_asr = transcribe_audio(asr_model, asr_processor, original, device)
        noisy = data['processed'].to(device)
        noisy_asr = transcribe_audio(asr_model, asr_processor, noisy, device)
        batch_transcriptions = {
            "original": original_asr,
            "noisy": noisy_asr
        }
        for model_name, model in models.items():
            with torch.no_grad():
                denoised = model(noisy)
                denoised_asr = transcribe_audio(asr_model, asr_processor, denoised, device)
                batch_transcriptions[model_name] = denoised_asr
            for idx in range(len(denoised)):
                torchaudio.save(os.path.join(
                    output_dir, f"test_{model_name}" , f"{data['filename'][idx]}"), denoised[idx].reshape(1, -1), FRAMERATE)
        for idx in range(len(data['filename'])):
            transcriptions[data['filename'][idx]] = {
                label: batch_transcriptions[label][idx] for label in batch_transcriptions
            }
    return transcriptions


def add_wer(results):
    for file in results:
        for model in list(results[file]):
            if model == "original":
                continue
            results[file][f"{model}_wer"] = float(word_error_rate(results[file][model], results[file]["original"]).numpy())


def get_summary_stats(results):
    summary_stats = {}
    for model in list(results.values())[0]:
        if not model.endswith("_wer"):
            continue
        wers = [results[file][model] for file in results]
        summary_stats[model] = {
            "avg": round(np.average(wers).astype(float), 3),
            "std": round(np.std(wers).astype(float), 3)
        }
    return summary_stats


def run_eval(clear_path, noisy_path, output_file, models, asr_model_size="tiny"):
    device = get_device()
    asr_model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{asr_model_size}").to(device)
    asr_processor = WhisperProcessor.from_pretrained(f"openai/whisper-{asr_model_size}")
    result = evaluate_models(clear_path, noisy_path, models, asr_model, asr_processor, device)
    add_wer(result)
    stats = get_summary_stats(result)
    with open(output_file, "w+") as f:
        json.dump({
            "summary": stats,
            "raw": result
        }, f, indent=4)


if __name__ == "__main__":
    run_eval("./data/test_clear", "./data/test_noisy", "./data/test_results.json", {
        "spectral": spectral_model
    })