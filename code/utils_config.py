from pathlib import Path

def get_txt_paths(folder_path):

    def get_train_txt_path(folder_path):
        return folder_path.joinpath("train.txt")

    def get_test_txt_path(folder_path):
        return folder_path.joinpath("test.txt")

    def get_train_embedding_path(folder_path):
        return folder_path.joinpath("train_embeddings.pkl")

    def get_test_embedding_path(folder_path):
        return folder_path.joinpath("test_embeddings.pkl")

    return (get_train_txt_path(folder_path),
            get_train_embedding_path(folder_path),
            get_test_txt_path(folder_path),
            get_test_embedding_path(folder_path))

def make_exp_folder(data_folder, new_folder_name):
    new_folder = data_folder.joinpath(new_folder_name)
    new_folder.mkdir(exist_ok=True)
    return new_folder

