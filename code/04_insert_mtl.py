import config, utils_config, utils_mlp_02
from pathlib import Path

train_size_to_minibatch_size = {20:20, 50:20, 100:66, 200:100, 500:200, None:200}

if __name__ == "__main__":

    for dataset_name in ['sst2', 'subj']:

        data_folder = config.data_folders[dataset_name]
        num_classes = config.num_classes_dict[dataset_name]
        train_txt_path, train_embedding_path, test_txt_path, test_embedding_path = utils_config.get_txt_paths(data_folder)
        print(f"\n-------{dataset_name.upper()}-------")

        for setup in ['insert', 'insert-mtl', 'vanilla']:

            print(f"\n{setup}:")

            for train_size in [20]:#, 50, 100, 500]:#, None]:

                mean_val_acc, stdev_acc = utils_mlp_02.train_mlp_multiple(   
                    setup,
                    train_txt_path,
                    train_embedding_path,
                    test_txt_path,
                    test_embedding_path,
                    num_classes,
                    dataset_name,
                    train_size,
                    alpha = 0.2,
                    num_seeds = 10,
                    minibatch_size = train_size_to_minibatch_size[train_size],
                    )

                print(f"{train_size},{mean_val_acc:.3f},{stdev_acc:.3f}")