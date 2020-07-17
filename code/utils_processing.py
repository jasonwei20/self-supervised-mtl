import utils_common
import numpy as np
from sklearn.utils import shuffle
import pathlib
import eda
import utils_bert
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_augmented_data_swaps(train_txt_path, n_aug=5):

    output_pkl_path = train_txt_path.parent.joinpath(f"train_aug_swap_data.pkl")

    if not output_pkl_path.exists():

        print(f"creating {output_pkl_path}")

        lines = open(train_txt_path, 'r').readlines()
        sentence_to_augmented_sentences = {}

        for line in lines:
            parts = line[:-1].split('\t')
            sentence = parts[1]
            augmented_sentences = eda.get_swap_sentences(sentence, n_aug, alpha=0.3)
            sentence_to_augmented_sentences[sentence] = augmented_sentences

        utils_common.save_pickle(output_pkl_path, sentence_to_augmented_sentences)
    
    return utils_common.load_pickle(output_pkl_path)

def get_sentence_to_label(train_txt_path):
    lines = open(train_txt_path).readlines()
    sentence_to_label = {}
    for line in lines:
        parts = line[:-1].split('\t')
        label = int(parts[0])
        sentence = parts[1]
        sentence_to_label[sentence] = label
    return sentence_to_label

def get_split_train_x_y(train_txt_path, train_subset, seed_num, setup):

    sentence_to_label = get_sentence_to_label(train_txt_path)
    sentence_to_augmented_sentences = get_augmented_data_swaps(train_txt_path)
    string_to_embedding = utils_bert.get_split_train_embedding_dict(sentence_to_augmented_sentences, train_txt_path)

    sentences = list(sentence_to_augmented_sentences.keys())
    labels = []
    for sentence in sentences:
        label = sentence_to_label[sentence]
        labels.append(label)

    train_sentences, _, train_labels, _ = train_test_split(sentences, labels, train_size=train_subset, random_state = seed_num, stratify = labels)

    train_x = []
    train_x_swap = []
    for train_sentence in train_sentences:
        train_sentence_swap = sentence_to_augmented_sentences[train_sentence][0]
        embedding = string_to_embedding[train_sentence]
        embedding_swap = string_to_embedding[train_sentence_swap]
        train_x.append(embedding)
        if setup in ['vanilla']:
            train_x_swap.append(embedding)
        else:
            train_x_swap.append(embedding_swap)
        
    train_x_np = np.asarray(train_x + train_x_swap)

    for _ in [train_x_swap]:
        train_labels += train_labels
    train_y_np = np.asarray(train_labels)

    num_classes_aux = len([train_x, train_x_swap])
    train_labels_aux = []
    for y_aux in range(num_classes_aux):
        for _ in range(len(train_sentences)):
            train_labels_aux.append(y_aux)
    train_y_aux_np = np.asarray(train_labels_aux)

    return train_x_np, train_y_np, train_y_aux_np, num_classes_aux

def get_x_y(txt_path, embedding_path):
    lines = open(txt_path).readlines()
    string_to_embedding = utils_common.load_pickle(embedding_path)

    x = np.zeros((len(lines), 768))
    y = np.zeros((len(lines), ))

    for i, line in enumerate(lines):
        parts = line[:-1].split('\t')
        label = int(parts[0])
        string = parts[1]
        assert string in string_to_embedding
        embedding = string_to_embedding[string]
        x[i, :] = embedding
        y[i] = label
    
    x, y = shuffle(x, y, random_state = 0)
    return x, y

def find_num_classes(lines):

    highest = 0
    for line in lines:
        parts = line[:-1].split('\t')
        label = int(parts[0])
        highest = max(highest, label)
    return highest + 1

def get_x_y_mlp(txt_path, embedding_path):
    lines = open(txt_path).readlines()
    string_to_embedding = utils_common.load_pickle(embedding_path)

    num_classes = find_num_classes(lines)
    x = np.zeros((len(lines), 768))
    y = np.zeros((len(lines), num_classes))

    for i, line in enumerate(lines):
        parts = line[:-1].split('\t')
        label = int(parts[0])
        string = parts[1]
        assert string in string_to_embedding
        embedding = string_to_embedding[string]
        x[i, :] = embedding
        y[i, label] = 1
    
    x, y = shuffle(x, y, random_state = 0)
    return x, y

def augment_swap(source_txt_path, target_txt_path, n_aug, alpha):
    
    writer = open(target_txt_path, 'w')
    lines = open(source_txt_path, 'r').readlines()
    for line in lines:
        parts = line[:-1].split('\t')
        label = int(parts[0])
        string = parts[1]
        augmented_sentences = eda.get_swap_sentences(string, n_aug=n_aug, alpha=alpha)

        for augmented_sentence in augmented_sentences:
            output_line = '\t'.join([str(label), augmented_sentence])
            writer.write(output_line + '\n')
    
    print(f"output file at {target_txt_path}")

def augment_insert(source_txt_path, target_txt_path, n_aug, alpha):
    
    writer = open(target_txt_path, 'w')
    lines = open(source_txt_path, 'r').readlines()
    for line in lines:
        parts = line[:-1].split('\t')
        label = int(parts[0])
        string = parts[1]
        augmented_sentences = eda.get_insert_sentences(string, n_aug=n_aug, alpha=alpha)

        for augmented_sentence in augmented_sentences:
            output_line = '\t'.join([str(label), augmented_sentence])
            writer.write(output_line + '\n')
    
    print(f"output file at {target_txt_path}")

def augment_delete(source_txt_path, target_txt_path, n_aug=10, alpha=0.1):
    
    writer = open(target_txt_path, 'w')
    lines = open(source_txt_path, 'r').readlines()
    
    for line in lines:
        parts = line[:-1].split('\t')
        label = int(parts[0])
        string = parts[1]
        augmented_sentences = eda.get_delete_sentences(string, n_aug=n_aug)

        for augmented_sentence in augmented_sentences:
            output_line = '\t'.join([str(label), augmented_sentence])
            writer.write(output_line + '\n')
    
    print(f"output file at {target_txt_path}")

def augment_eda(source_txt_path, target_txt_path):
    
    writer = open(target_txt_path, 'w')
    lines = open(source_txt_path, 'r').readlines()
    for line in lines:
        parts = line[:-1].split('\t')
        label = int(parts[0])
        string = parts[1]
        augmented_sentences = eda.eda(string)

        for augmented_sentence in augmented_sentences:
            output_line = '\t'.join([str(label), augmented_sentence])
            writer.write(output_line + '\n')
    
    print(f"output file at {target_txt_path}")