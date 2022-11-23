from superiear.scraping import download_audio, read_urls
from superiear.processing import split_sentences, insert_noise, split_test_samples

data_urls = read_urls("data_urls.json")
download_audio(data_urls, "./data/raw")

noise_urls = read_urls("noise_urls.json")
download_audio(noise_urls, "./data/noises")

split_sentences("./data/raw", "./data/clear_samples")
insert_noise("./data/clear_samples", "./data/noisy_samples", "./data/noises/")
split_test_samples("./data/clear_samples", "./data/noisy_samples")