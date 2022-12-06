import random
import glob
import os
from tqdm import tqdm
from pydub import AudioSegment
from pydub.silence import split_on_silence

from superiear.utils import ensure_dir_exists

random.seed(0)
MS = 1000
FRAMERATE = int(16000)


@ensure_dir_exists
def split_sentences(input_dir, output_dir, skip_intro=60, duration=7):
    for raw_filename in tqdm(glob.glob(f"{input_dir}/*.wav")):
        print(raw_filename)
        if ".temp.wav" in raw_filename:
            continue
        track = AudioSegment.from_file(raw_filename)

        if skip_intro > 0:
            track = track[skip_intro * MS:]

        # Ensure we split tracks into self-contained chunks
        segments = split_on_silence(
            track, min_silence_len=550, silence_thresh=-40, seek_step=100)

        # Make all chunks the same length
        segments = [chunk[:duration * MS]
                    for chunk in segments if len(chunk) >= duration * MS]

        for i, segment in enumerate(segments):
            raw_clean_filename = raw_filename.split(
                "/")[-1].replace(".wav", "")
            filename = f"{output_dir}/{raw_clean_filename}_{i}.wav"
            segment = segment.set_frame_rate(FRAMERATE)
            segment.export(filename, format="wav")


@ensure_dir_exists
def insert_noise(input_dir, output_dir, noise_dir):
    noises = [AudioSegment.from_file(fname)
              for fname in glob.glob(f"{noise_dir}/*.wav")]

    for processed_filename in tqdm(glob.glob(f"{input_dir}/*.wav")):
        track = AudioSegment.from_file(processed_filename)

        noise_type = random.randint(0, len(noises) - 1)

        # Start noise at random point in first half of the track
        start = random.random() * len(track) / 2
        noisy_track = track.overlay(noises[noise_type], position=start)

        raw_clean_filename = processed_filename.split("/")[-1]
        filename = f"{output_dir}/{raw_clean_filename}"
        noisy_track = noisy_track.set_frame_rate(FRAMERATE)
        noisy_track.export(filename, format="wav")


@ensure_dir_exists
def split_test_samples(clear_dir, noise_dir, frac_test=0.1):
    files = [f.split("/")[-1] for f in glob.glob(f"{clear_dir}/*.wav")]
    test_files = random.sample(files, int(len(files) * frac_test))
    path = "/".join(clear_dir.split("/")[:-1])
    for dir in [f"{path}/test_clear", f"{path}/test_noisy"]:
        if not os.path.exists(dir):
            os.mkdir(dir)
    for f in tqdm(test_files):
        os.rename(f"{clear_dir}/{f}", f"{path}/test_clear/{f}")
        os.rename(f"{noise_dir}/{f}", f"{path}/test_noisy/{f}")
