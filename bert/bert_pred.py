import random

import conlleval
import hydra
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, BertConfig

from tools.ne_extracter import extract
from utils.boi_convert import convert
from utils.datamodule import BertCRF
from utils.maxMatchTokenizer import MaxMatchTokenizer
from utils.utils import get_texts_and_labels, path_to_data, val_to_key


@hydra.main(config_path="../config", config_name="conll2003", version_base="1.1")
def main(cfg):
    length = cfg.length
    path = cfg.path
    test = cfg.test
    leak = cfg.leak
    pred(length, path, test, leak)


def pred(length, path, test, leak):
    batch_size = 1
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

    p = 0
    non_train_p = 0.000000001
    split_first = True

    device = "cuda"
    if test == "2003":
        dataset = load_dataset("conll2003")
        test_dataset = dataset["test"]
        encoding = "cp932"

    elif test == "2023":
        test_dataset = path_to_data("C:/Users/chenr/Desktop/python/subword_regularization/test_datasets/conll2023.txt")
        encoding = "utf-8"

    elif test == "crossweigh":
        test_dataset = path_to_data("C:/Users/chenr/Desktop/python/subword_regularization/test_datasets/conllcw.txt")
        encoding = "utf-8"

    trained_tokens = "C:/Users/chenr/Desktop/python/subword_regularization/ne_data_train.txt"
    print("Dataset Loaded")
    ner_dict = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-MISC": 7,
        "I-MISC": 8,
        "PAD": 9,
    }
    pos_dict = {
        '"': 0,
        "''": 1,
        "#": 2,
        "$": 3,
        "(": 4,
        ")": 5,
        ",": 6,
        ".": 7,
        ":": 8,
        "``": 9,
        "CC": 10,
        "CD": 11,
        "DT": 12,
        "EX": 13,
        "FW": 14,
        "IN": 15,
        "JJ": 16,
        "JJR": 17,
        "JJS": 18,
        "LS": 19,
        "MD": 20,
        "NN": 21,
        "NNP": 22,
        "NNPS": 23,
        "NNS": 24,
        "NN|SYM": 25,
        "PDT": 26,
        "POS": 27,
        "PRP": 28,
        "PRP$": 29,
        "RB": 30,
        "RBR": 31,
        "RBS": 32,
        "RP": 33,
        "SYM": 34,
        "TO": 35,
        "UH": 36,
        "VB": 37,
        "VBD": 38,
        "VBG": 39,
        "VBN": 40,
        "VBP": 41,
        "VBZ": 42,
        "WDT": 43,
        "WP": 44,
        "WP$": 45,
        "WRB": 46,
    }
    test_data = get_texts_and_labels(test_dataset)

    trained_tokens, _ = extract("train", all_token_extract=True)
    trained_tokens = [train_row.split(", ")[0] for train_row in trained_tokens.split("\n") if len(train_row) != 0]

    mmt = MaxMatchTokenizer(ner_dict=ner_dict, p=0, padding=length, trained_tokens=trained_tokens)
    bert_tokeninzer = AutoTokenizer.from_pretrained("bert-base-cased")
    mmt.loadBertTokenizer(bertTokenizer=bert_tokeninzer)

    test_data = mmt.dataset_encode(
        test_data,
        p=p,
        subword_label="PAD",
        non_train_p=non_train_p,
        split_first=split_first,
    )

    device = "cuda"
    config = BertConfig.from_pretrained("bert-base-cased", num_labels=len(ner_dict))
    model = AutoModelForTokenClassification.from_config(config)
    model = BertCRF(model, batch_size, len(ner_dict), length).to(device)
    model.load_state_dict(torch.load(path))

    print("Model Loaded")

    inputs, attention_mask, labels, out_tokens, out_poses, out_ners = (
        test_data["input_ids"],
        test_data["attention_mask"],
        test_data["subword_labels"],
        test_dataset["tokens"],
        test_dataset["pos_tags"],
        test_dataset["ner_tags"],
    )

    if leak in ["non_leak", "non_leak_divided", "only_leak"]:
        trained_tokens, _ = extract("train")
        trained_tokens = [train_row.split(", ")[0] for train_row in trained_tokens.split("\n") if len(train_row) != 0]

    output = []
    model.eval()
    with torch.no_grad():
        for input, sub_input, label, out_token, out_pos, out_ner in zip(tqdm(inputs), attention_mask, labels, out_tokens, out_poses, out_ners):
            if leak == "non_leak":
                leak_check = [t not in trained_tokens for t in out_token]

            elif leak == "non_leak_divided":
                leak_check = [t not in trained_tokens and (len(mmt.tokenizeWord(t)) != 1) for t in out_token]

            elif leak == "only_leak":
                leak_check = [t in trained_tokens for t in out_token]

            input, sub_input, label = (
                input.to(device),
                sub_input.to(device),
                label.tolist(),
            )
            input = input.unsqueeze(0)
            sub_input = sub_input.unsqueeze(0)

            pred = model.decode(input, sub_input).squeeze().to("cpu").tolist()

            pred = [val_to_key(prd, ner_dict) for (prd, lbl) in zip(pred, label) if lbl != ner_dict["PAD"]]
            pred = [c if c != "PAD" else "O" for c in pred]

            if len(pred) != len(out_ner):
                print(" ".join(out_token))
                pred = pred + ["O"] * (len(out_ner) - len(pred))

            out_pos = [val_to_key(o_p, pos_dict) for o_p in out_pos]
            out_ner = [val_to_key(o_n, ner_dict) for o_n in out_ner]

            out_check = (
                [(n == "O") or l_k for l_k, n in zip(leak_check, out_ner)]
                if leak in ["non_leak", "non_leak_divided", "only_leak"]
                else [True] * len(pred)
            )
            out = [" ".join([t, p, c, pred]) for t, p, c, pred, o_c in zip(out_token, out_pos, out_ner, pred, out_check) if o_c]
            if len(out) != 0:
                out = "\n".join(out)
                output.append(out)
                output.append("\n\n")
    output = "".join(output).split("\n")
    path = "./output.txt"
    convert(output, path, encoding=encoding)

    with open(path, encoding=encoding) as f:
        file_iter = f.readlines()
    conlleval.evaluate_conll_file(file_iter)


if __name__ == "__main__":
    main()
