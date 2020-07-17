import utils_common
import numpy as np
from sklearn.utils import shuffle
import pathlib
import eda
import utils_bert
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_augmented_data(train_txt_path, augmentation, alpha, n_aug=1):

    output_pkl_path = train_txt_path.parent.joinpath(f"train_aug_{augmentation}_data_{alpha}.pkl")

    if not output_pkl_path.exists():

        print(f"creating {output_pkl_path}")

        lines = open(train_txt_path, 'r').readlines()
        sentence_to_augmented_sentences = {}

        for line in lines:
            parts = line[:-1].split('\t')
            sentence = parts[1]
            if augmentation == 'swap':
                augmented_sentences = eda.get_swap_sentences(sentence, n_aug, alpha)
            elif augmentation == 'insert':
                augmented_sentences = eda.get_insert_sentences(sentence, n_aug, alpha)
            elif augmentation == 'delete':
                augmented_sentences = eda.get_insert_sentences(sentence, n_aug, alpha)
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

def get_split_train_x_y(train_txt_path, train_subset, seed_num, setup, alpha):

    setup_to_augmentations = {  'swap': ['swap'], 'delete': ['delete'], 'insert': ['insert'],
                                'swap-mtl': ['swap'], 'delete-mtl': ['delete'], 'insert-mtl': ['insert'], 
                                'three_aug': ['delete', 'insert', 'swap'],
                                'three_aug-mtl': ['delete', 'insert', 'swap'],
                                'vanilla': []}
    augmentations = setup_to_augmentations[setup]

    big_dict_aug_sentences = {}
    big_dict_embeddings = {}
    for augmentation in augmentations:
        sentence_to_augmented_sentences = get_augmented_data(train_txt_path, augmentation, alpha)
        string_to_embedding = utils_bert.get_split_train_embedding_dict(sentence_to_augmented_sentences, train_txt_path, augmentation, alpha)
        big_dict_aug_sentences[augmentation] = sentence_to_augmented_sentences
        big_dict_embeddings[augmentation] = string_to_embedding

    sentence_to_label = get_sentence_to_label(train_txt_path)
    sentences = list(sentence_to_label.keys())
    labels = []
    for sentence in sentences:
        label = sentence_to_label[sentence]
        labels.append(label)
    original_sentence_to_embedding = utils_common.load_pickle(train_txt_path.parent.joinpath(f"train_embeddings.pkl"))

    train_sentences, _, train_labels, _ = train_test_split(sentences, labels, train_size=train_subset, random_state = seed_num, stratify = labels)

    # get train_x_np
    train_x = []
    aug_train_x_dict = {augmentation: [] for augmentation in augmentations}

    for train_sentence in train_sentences:

        embedding = original_sentence_to_embedding[train_sentence]
        train_x.append(embedding)

        for augmentation in augmentations:
            sentence_to_augmented_sentences = big_dict_aug_sentences[augmentation]
            string_to_embedding = big_dict_embeddings[augmentation]
            train_sentence_swap = sentence_to_augmented_sentences[train_sentence][0]
            embedding_swap = string_to_embedding[train_sentence_swap]
            aug_train_x_dict[augmentation].append(embedding_swap)
    
    for augmentation in augmentations:
        train_x += aug_train_x_dict[augmentation]

    train_x_np = np.asarray(train_x)

    # get train_y_np
    train_labels_dup = list(train_labels)
    for _ in augmentations:
        train_labels_dup += train_labels
    train_y_np = np.asarray(train_labels_dup)

    #get train_y_aux
    num_classes_aux = len([train_x] + augmentations)
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