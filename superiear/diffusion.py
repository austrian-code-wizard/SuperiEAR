from diffwave.inference import predict as diffwave_predict
from diffwave.preprocess import main as diffwave_transform
import os
from diffwave.__main__ import main as diffwave_main
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm
import torchaudio



def use_diffusion_api(model_dir, eval_set):
    spectrogram = eval_set # get your hands on a spectrogram in [N,C,W] format
    audio, sample_rate = diffwave_predict(spectrogram, model_dir, fast_sampling=True)
    torchaudio.save("audiofile.wave", audio, sample_rate)

def train_model(model_dir, data_dir):
    # pre process
    filenames = glob(f'{data_dir}/**/*.wav', recursive=True)
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(diffwave_transform, filenames), desc='Preprocessing', total=len(filenames)))

    # main (would be easier to do with terminal commands, but assume using os with
    # remote desktops may be difficult?



if __name__ == '__main__':
    types = ["pretrained", "train"]
    data_directory = "\data" #set this to the correct directory
    type = types[0] #set what type of model we are going to do
    if (type == types[0]):
        model_dir = '/pretrained_model'  # download pretrained model and save directory
        eval_set = 'something' #figure out how to load a audio file to evaluate
        use_diffusion_api(model_dir, eval_set)

    if (type == types[1]):
        train_model("directory", data_directory)
