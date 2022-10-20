from curses import raw
import random
import glob
from turtle import position
from tqdm import tqdm
from pydub import AudioSegment
from pydub.silence import split_on_silence

from superiear.utils import ensure_dir_exists

MS = 1000

@ensure_dir_exists
def split_sentences(input_dir, output_dir, skip_intro=60, duration=7):
  for raw_filename in tqdm(glob.glob(f"{input_dir}/*.wav")):
    track = AudioSegment.from_file(raw_filename)

    if skip_intro > 0:
      track = track[skip_intro * MS:]

    # Ensure we split tracks into self-contained chunks
    segments = split_on_silence(track, min_silence_len=550, silence_thresh=-40)

    # Make all chunks the same length
    segments = [chunk[:duration * MS] for chunk in segments if len(chunk) >= duration * MS]

    for i, segment in enumerate(segments):
      raw_clean_filename = raw_filename.split("/")[-1].replace(".wav", "")
      filename = f"{output_dir}/{raw_clean_filename}_{i}.wav"
      segment.export(filename, format="wav")

@ensure_dir_exists
def insert_noise(input_dir, output_dir, noise_dir):
  noises = [AudioSegment.from_file(fname) for fname in glob.glob(f"{noise_dir}/*.wav")]

  random.seed(0)

  for processed_filename in tqdm(glob.glob(f"{input_dir}/*.wav")):
    track = AudioSegment.from_file(processed_filename)

    noise_type = random.randint(0, len(noises) - 1)
    
    # Start noise at random point in first half of the track
    start = random.random() * len(track) / 2
    noisy_track = track.overlay(noises[noise_type], position=start)

    raw_clean_filename = processed_filename.split("/")[-1]
    filename = f"{output_dir}/{raw_clean_filename}"
    noisy_track.export(filename, format="wav")
