from pathlib import Path

data_folders = {"subj": Path("data/subj"),
                "sst2": Path("data/sst2"),
                "imdb": Path("data/imdb"),
                "trec": Path("data/trec"),
                }

num_classes_dict = {"subj": 2,
                    "sst2": 2,
                    "imdb": 2,
                    "trec": 6,
                    }