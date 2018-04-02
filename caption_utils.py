import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import image
import matplotlib.pyplot as plt
import itertools
from collections import Counter
import json


exclude_words = ('x')


def load_split_lists():
    def read_file(fname):
        with open(fname, "r") as f:
            contents = f.read()
        return contents.split("\n")[:-1]

    train_fname = "Flickr_8k.trainImages.txt"
    dev_fname = "Flickr_8k.devImages.txt"
    test_fname = "Flickr_8k.testImages.txt"

    train_fns_list = read_file("data/Flickr8k_text/" + train_fname)
    dev_fns_list = read_file("data/Flickr8k_text/" + dev_fname)
    test_fns_list = read_file("data/Flickr8k_text/" + test_fname)
    
    return train_fns_list, dev_fns_list, test_fns_list


def get_caption_split():
    train_fns_list, dev_fns_list, test_fns_list = load_split_lists()
    
    train_captions = {img_name: [] for img_name in train_fns_list}
    dev_captions = {img_name: [] for img_name in dev_fns_list}
    test_captions = {img_name: [] for img_name in test_fns_list}    
    
    with open("data/Flickr8k_text/" + "Flickr8k.token.txt") as f:
        flickr8k_token = f.readlines()
    
    for line in flickr8k_token:
        line_split = line.split("\t")
        img_file, _ = line_split[0].split("#")
        
        caption = line_split[1].strip()        
        
        if img_file in train_captions:
            train_captions[img_file] = train_captions[img_file] + [caption]
        elif img_file in dev_captions:
            dev_captions[img_file] = dev_captions[img_file] + [caption]
        elif img_file in test_captions:
            test_captions[img_file] = test_captions[img_file] + [caption]
        else:
            #print(img_file, "not in train, dev, test split")
            pass
    return train_captions, dev_captions, test_captions


def create_vocab(train_captions_raw, min_word_freq=5, exclude_numbers=True):
    word_count = get_word_count(train_captions_raw)
    raw_vocab = sorted(word_count.keys())
    if exclude_numbers:
        vocab = raw_vocab[raw_vocab.index('a'):]  
    
    new_vocab = []
    for word, val in word_count.items():  
        if word not in vocab or word in exclude_words:
            continue
        if val >= min_word_freq:
            new_vocab.append(word)
            
    return ['<pad>', '<bos>', '<eos>', '<unk>'] + sorted(new_vocab)


def get_word_count(train_captions_raw):
    """ returns dict of word counts {word: count,...} """
    captions_list = itertools.chain.from_iterable(train_captions_raw.values())
    captions_tokens = map(text_to_word_sequence, captions_list)
    all_words = itertools.chain.from_iterable(captions_tokens)
    return Counter(all_words)
    

def vocab_to_index(vocab):
    token2idx = {token: i for i, token in enumerate(vocab)}
    idx2token = {i: token for i, token in enumerate(vocab)}
    
    assert(len(idx2token) == len(idx2token))
    for token, idx in token2idx.items():
        assert(idx2token[idx] == token, "token2idx and idx2token not equivalent")
    
    return token2idx, idx2token


def process_captions(captions_data, token2idx):
    def caption2idx(caption):
        return [ [token2idx["<bos>"]] + list(map(lambda x: token2idx.get(x, token2idx['<unk>']), 
                         text_to_word_sequence(cap))) + [token2idx["<eos>"]] for cap in caption]
    
    for data in captions_data:
        for img, cap in data.items():
            data[img] = caption2idx(cap)
            
    return captions_data


def get_max_length(train_captions, dev_captions, test_captions):
    return max(max([len(x) for x in itertools.chain.from_iterable(train_captions.values())]), 
               max([len(x) for x in itertools.chain.from_iterable(test_captions.values())]), 
               max([len(x) for x in itertools.chain.from_iterable(dev_captions.values())]))
    

def intseq_to_caption(idx2token, caption):
    """ 
    token2idx: dict
    caption list(int) representing a caption
    """
    return ' '.join(map(lambda x: idx2token[x], caption))


def visualize_training_example(img_fname, captions):
    img = image.load_img("data/Flicker8k_Dataset/" + img_fname, target_size=(224, 224))
    plt.imshow(img)
    print(captions)
    plt.title("\n".join(captions))
    plt.show()
    
    
if __name__ == "__main__":
    train_fns_list, dev_fns_list, test_fns_list = load_split_lists()
    train_captions_raw, dev_captions_raw, test_captions_raw = get_caption_split()
    vocab = create_vocab(train_captions_raw)
    token2idx, idx2token = vocab_to_index(vocab)     
    captions_data = (train_captions_raw.copy(), dev_captions_raw.copy(), test_captions_raw.copy())
    train_captions, dev_captions, test_captions = process_captions(captions_data, token2idx)
    max_length = get_max_length(train_captions, dev_captions, test_captions)
    
    
    img_fname = train_fns_list[int(input("Image num: "))]    
    visualize_training_example(img_fname, train_captions_raw[img_fname])
