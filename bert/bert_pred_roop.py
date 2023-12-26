import os
import random
import sys
from logging import getLogger

import hydra
import numpy as np
import torch
from round_table import round_table
from tqdm import tqdm
from tqdm.contrib import tzip
from transformers import AutoModelForTokenClassification, AutoTokenizer, BertConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.maxMatchTokenizer import MaxMatchTokenizer
from utils.utils import path_to_data, val_to_key

logger = getLogger(__name__)

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

model_dict = {
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


@hydra.main(config_path="../config", config_name="conll2003", version_base="1.1")
def main(cfg):
    os.environ["TRANSFORMERS_CACHE"] = "D:\\huggingface\\cashe"
    length = cfg.length
    test = cfg.test
    model_name = cfg.model_name
    loop = cfg.loop
    p = cfg.pred_p
    vote = cfg.vote
    local_model = cfg.local_model
    pre_sentence_padding = cfg.pre_sentence_padding
    post_sentence_padding = cfg.post_sentence_padding
    add_sep_between_sentences = cfg.add_sep_between_sentences
    loop_pred(
        length,
        model_name,
        test,
        loop=loop,
        p=p,
        vote=vote,
        local_model=local_model,
        pre_sentence_padding=pre_sentence_padding,
        post_sentence_padding=post_sentence_padding,
        add_sep_between_sentences=add_sep_between_sentences,
    )


def loop_pred(
    length,
    model_name,
    test,
    loop=100,
    p=0.1,
    vote="majority",
    local_model=False,
    pre_sentence_padding=False,
    post_sentence_padding=False,
    add_sep_between_sentences=False,
):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

    if model_name[:10] == "dslim/bert":
        global model_dict
        model_dict = {
            "O": 0,
            "B-MISC": 1,
            "I-MISC": 2,
            "B-PER": 3,
            "I-PER": 4,
            "B-ORG": 5,
            "I-ORG": 6,
            "B-LOC": 7,
            "I-LOC": 8,
            "PAD": 9,
        }

    device = "cuda"
    if test == "2003":
        test_dataset = path_to_data("C:/Users/chenr/Desktop/python/subword_regularization/test_datasets/eng.testb")
        encoding = "utf-8"

    elif test == "valid":
        test_dataset = path_to_data("C:/Users/chenr/Desktop/python/subword_regularization/test_datasets/eng.testa")
        encoding = "utf-8"

    elif test == "2023":
        test_dataset = path_to_data("C:/Users/chenr/Desktop/python/subword_regularization/test_datasets/conllpp.txt")
        encoding = "utf-8"
    elif test == "crossweigh":
        test_dataset = path_to_data("C:/Users/chenr/Desktop/python/subword_regularization/test_datasets/conllcw.txt")
        encoding = "utf-8"

    mmt = MaxMatchTokenizer(ner_dict=ner_dict, p=p, padding=length)
    bert_tokeninzer = AutoTokenizer.from_pretrained(model_name)
    mmt.loadBertTokenizer(bertTokenizer=bert_tokeninzer)

    if local_model:
        config = BertConfig.from_pretrained(model_name, num_labels=len(ner_dict))
        model = AutoModelForTokenClassification.from_config(config)
        model.load_state_dict(torch.load(local_model))
        model = model.to(device)
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)

    model.eval()
    outputs = []
    for i in tqdm(range(loop)):
        if i == loop - 1:
            p = 0
        output = pred(
            mmt,
            test_dataset,
            model,
            device,
            p=p,
            pre_sentence_padding=pre_sentence_padding,
            post_sentence_padding=post_sentence_padding,
            add_sep_between_sentences=add_sep_between_sentences,
        )
        outputs.append(output)

    joined_out = []
    for index in range(len(outputs[0])):
        if len(outputs[0][index]) < 5:
            joined_out.append("")
        else:
            pred_ners = []
            for output in outputs:
                token, pos, ner, pred_ner = output[index].split(" ")
                if len(pred_ners) == 0:
                    pred_ners.append(token)
                    pred_ners.append(pos)
                    pred_ners.append(ner)
                pred_ners.append(pred_ner)
            pred_ners = " ".join(pred_ners)
            joined_out.append(pred_ners)

    joined_out = "\n".join(joined_out)
    path = "./many_preds.txt"
    with open(path, "w", encoding=encoding) as f_out:
        f_out.write(joined_out)

    with open(path, encoding=encoding) as f:
        file_iter = f.readlines()
    round_table(file_iter, encoding, vote)


def pred(
    mmt,
    test_dataset,
    model,
    device="cuda",
    p=0,
    pre_sentence_padding=False,
    post_sentence_padding=False,
    add_sep_between_sentences=False,
):
    encoded_dataset = mmt.dataset_encode(
        test_dataset,
        p=p,
        subword_label="PAD",
        pre_sentence_padding=pre_sentence_padding,
        post_sentence_padding=post_sentence_padding,
        add_sep_between_sentences=add_sep_between_sentences,
    )

    inputs, attention_mask, type_ids, labels, out_tokens, out_ners = (
        encoded_dataset["input_ids"],
        encoded_dataset["attention_mask"],
        encoded_dataset["token_type_ids"],
        encoded_dataset["predict_labels"],
        encoded_dataset["tokens"],
        encoded_dataset["labels"],
    )

    output = []
    with torch.no_grad():
        for input, mask, type_id, label, out_token, out_ner in tzip(
            inputs, attention_mask, type_ids, labels, out_tokens, out_ners, leave=False
        ):
            input, mask, type_id, label = (
                input.to(device),
                mask.to(device),
                type_id.to(device),
                label.tolist(),
            )
            input = input.unsqueeze(0)
            mask = mask.unsqueeze(0)
            type_id = type_id.unsqueeze(0)

            pred = model(input, mask, type_id).logits.squeeze().argmax(-1).to("cpu").tolist()

            pred = [val_to_key(prd, model_dict) for (prd, lbl) in zip(pred, label) if lbl != ner_dict["PAD"]]
            pred = [c if c != "PAD" else "O" for c in pred]
            if len(pred) != len(out_ner):
                logger.warning("Bad Prediction!!!!")
                pred = pred + ["O"] * (len(out_ner) - len(pred))

            out_pos = ["POS" for _ in out_ner]
            out_ner = [val_to_key(o_n, ner_dict) for o_n in out_ner]
            out = [" ".join([t, p, c, pred]) for t, p, c, pred in zip(out_token, out_pos, out_ner, pred)]
            out = "\n".join(out)
            output.append(out)
            output.append("\n\n")
    output = "".join(output).split("\n")

    return output


if __name__ == "__main__":
    main()
