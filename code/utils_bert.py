import utils_common
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import gc
from sklearn.metrics.pairwise import cosine_similarity

model_class = BertModel
tokenizer_class = BertTokenizer
pretrained_weights = 'bert-base-uncased'
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

def get_split_train_embedding_dict(sentence_to_augmented_sentences, train_txt_path, augmentation, alpha):
    
    embeddings_dict_path = train_txt_path.parent.joinpath(f"train_aug_{augmentation}_embeddings_{alpha}.pkl")
    
    if not embeddings_dict_path.exists():

        print(f"creating {embeddings_dict_path}")
        
        string_to_embedding = {}

        for sentence, augmented_sentences in tqdm(sentence_to_augmented_sentences.items()):
            embedding = get_embedding(sentence, tokenizer, model)
            string_to_embedding[sentence] = embedding
            for augmented_sentence in augmented_sentences:
                aug_embedding = get_embedding(augmented_sentence, tokenizer, model)
                string_to_embedding[augmented_sentence] = aug_embedding
    
        utils_common.save_pickle(embeddings_dict_path, string_to_embedding)
    
    return utils_common.load_pickle(embeddings_dict_path)

# Encode text
def get_embedding(input_string, tokenizer, model):
    input_ids = torch.tensor([tokenizer.encode(input_string, add_special_tokens = True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0].numpy()  # Models outputs are now tuples
        # last_hidden_states = last_hidden_states[:, 0, :]
        last_hidden_states = np.mean(last_hidden_states, axis = 1)
        last_hidden_states = last_hidden_states.flatten()
        return last_hidden_states
    