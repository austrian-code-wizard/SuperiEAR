from superiear.scraping import download_audio, read_urls
from superiear.processing import split_sentences, insert_noise

data_urls = read_urls("data_urls.json")
download_audio(data_urls, "./data/raw")

noise_urls = read_urls("noise_urls.json")
download_audio(noise_urls, "./data/noises")

split_sentences("./data/raw", "./data/processed")
insert_noise("./data/processed", "./data/final", "./data/noises/")
