import copy
import random

import nltk
import torch
from nltk.corpus import stopwords
from transformers import AutoTokenizer


class MaxMatchTokenizer:
    def __init__(
        self,
        vocab=None,
        midPref="##",
        headPref="",
        p=0.0,
        padding=False,
        ner_dict=None,
        stop_word=False,
        trained_tokens=[],
    ):
        self.midPref = midPref
        self.headPref = headPref
        self.doNaivePreproc = False
        self.trained_tokens = trained_tokens
        if vocab:
            self.__build(vocab)

        self.p = p
        self.padding = padding
        self.ner_dict = ner_dict
        self.subword_dict = {
            0: 0,
            1: 2,
            2: 2,
            3: 4,
            4: 4,
            5: 6,
            6: 6,
            7: 8,
            8: 8,
            9: 9,
        }  # Chage from subword B-xxx label to I-xxx label
        if stop_word:
            nltk.download("stopwords")
            self.stop_words = stopwords.words("english")
        self.use_stop_word = stop_word

    def __build(self, vocab):
        self.unkToken = "[UNK]"
        self.vocab = set(vocab)
        self.vocab.add(self.unkToken)
        self.vocabSize = len(self.vocab)

        self.maxLength = max(len(w) for w in self.vocab)

        self.word2id = {}
        self.id2word = {}
        for w in sorted(self.vocab):
            self.word2id[w] = len(self.word2id)
            self.id2word[self.word2id[w]] = w

    # This function corresponds to Algorithm 1 in the paper.
    def tokenizeWord(self, word, p=False, split_first=False):
        p = p if p else self.p
        if self.use_stop_word:
            if word in self.stop_words:
                return [word]

        subwords = []
        i = 0
        wordLength = len(word)
        while i < wordLength:
            subword = None
            for j in range(1, min(self.maxLength + 1, wordLength - i + 1)):
                w = word[i : i + j]

                if 0 == i:
                    w = self.headPref + w
                if 0 < i:
                    w = self.midPref + w

                if w in self.vocab:
                    # random for subword regularization
                    if j == 1 or p < random.random():
                        # drop acception with p
                        subword = w

            if subword is None:
                # return unk if including unk
                return [self.unkToken]
            else:
                if split_first and len(subwords) == 0 and subword == word:
                    for k in range(1, len(subword)):
                        if subword[: (len(subword) - k)] in self.vocab:
                            subword = subword[: (len(subword) - k)]
                            break

                i += len(subword) - len(self.midPref) if 0 < i else len(subword) - len(self.headPref)
                subwords.append(subword)
        return subwords

    def tokenize(self, text, p=False, non_train_p=False, split_first=False):
        p = p if p else self.p
        if type(text) is list:
            return [self.tokenize(line, p, non_train_p, split_first=split_first) for line in text]
        if self.doNaivePreproc:
            text = self.naivePreproc(text)
        if non_train_p:
            subwords = []
            for i, word in enumerate(text.split()):
                if word in self.trained_tokens:
                    subword = self.tokenizeWord(word, p)
                else:
                    subword = self.tokenizeWord(word, non_train_p, split_first)

                for sw in subword:
                    subwords.append((i, sw))

        else:
            subwords = [
                (i, subword)
                for i, word in enumerate(text.split())
                for subword in self.tokenizeWord(word, p, split_first)
            ]
        word_ids = [sw[0] for sw in subwords]
        subwords = [sw[1] for sw in subwords]
        return subwords, word_ids

    def encode(self, text, p=False, non_train_p=False, split_first=False):
        p = p if p else self.p
        if type(text) is list:
            subwords = [self.clsTokenId] + [
                self.word2id[w]
                for line in text
                for w in self.tokenize(line, p, non_train_p, split_first)[0] + [self.sepToken]
            ]
            word_ids = [None] + [self.tokenize(line, p, non_train_p, split_first)[1] for line in text + [None]]

            return subwords, word_ids
        subwords, word_ids = self.tokenize(text, p, non_train_p, split_first)
        subwords = [self.clsTokenId] + [self.word2id[w] for w in subwords] + [self.sepTokenId]
        word_ids = [None] + word_ids + [None]
        if self.padding:
            if len(subwords) >= self.padding:
                subwords = subwords[: self.padding]
                word_ids = word_ids[: self.padding]
                attention_mask = [1] * self.padding
            else:
                attention_len = len(subwords)
                pad_len = self.padding - len(subwords)
                subwords += [self.padTokenId] * pad_len
                word_ids += [None] * pad_len
                attention_mask = [1] * attention_len + [0] * pad_len

        else:
            attention_mask = [1] * len(subwords)

        return subwords, word_ids, attention_mask

    def loadVocab(self, path):
        words = [line.strip() for line in open(path)]
        self.vocab = set()
        self.word2id = {}
        self.id2word = {}
        for i, w in enumerate(words):
            self.vocab.add(w)
            self.word2id[w] = i
            self.id2word[i] = w
        self.vocabSize = len(self.vocab)
        self.maxLength = max(len(w) for w in self.vocab)

        self.unkToken = "[UNK]"
        self.unkTokenId = self.word2id[self.unkToken]
        self.clsToken = "[CLS]"
        self.clsTokenId = self.word2id[self.clsToken]
        self.sepToken = "[SEP]"
        self.sepTokenId = self.word2id[self.sepToken]

    def loadBertTokenizer(self, bertTokenizer, doNaivePreproc=False):
        if doNaivePreproc:
            self.doNaivePreproc = doNaivePreproc
            self.bertTokenizer = bertTokenizer

        self.midPref = "##"
        self.vocab = set()
        self.word2id = {}
        self.id2word = {}

        for w, i in bertTokenizer.vocab.items():
            self.vocab.add(w)
            self.word2id[w] = i
            self.id2word[i] = w
        self.vocabSize = len(self.vocab)
        self.maxLength = max(len(w) for w in self.vocab)

        self.unkToken = bertTokenizer.unk_token
        self.unkTokenId = bertTokenizer.unk_token_id
        self.clsToken = bertTokenizer.cls_token
        self.clsTokenId = bertTokenizer.cls_token_id
        self.sepToken = bertTokenizer.sep_token
        self.sepTokenId = bertTokenizer.sep_token_id
        self.bosToken = bertTokenizer.bos_token
        self.bosTokenId = bertTokenizer.bos_token_id
        self.eosToken = bertTokenizer.eos_token
        self.eosTokenId = bertTokenizer.eos_token_id
        self.padToken = bertTokenizer.pad_token
        self.padTokenId = bertTokenizer.pad_token_id

        if self.use_stop_word:
            self.stop_words = [s_word for s_word in self.stop_words if s_word in self.vocab]

    def naivePreproc(self, text):
        return " ".join(self.bertTokenizer.tokenize(text)).replace(" " + self.midPref, "")

    def get_label(self, word_ids, label, subword_label):
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(self.ner_dict["PAD"])
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                if subword_label == "I":
                    label_ids.append(self.subword_dict[label[word_idx]])
                elif subword_label == "B":
                    label_ids.append(label[word_idx])
                elif subword_label == "PAD":
                    label_ids.append(self.ner_dict["PAD"])
                else:
                    print("subword_label must be 'I', 'B' or 'PAD'.")
                    exit(1)

            previous_word_idx = word_idx
        return label_ids

    def dataset_encode(
        self,
        data,
        p=False,
        return_tensor=True,
        subword_label="I",
        non_train_p=False,
        split_first=False,
    ):
        p = p if p else self.p

        def max_with_none(lst):
            lst = [ls for ls in lst if ls is not None]
            return max(lst)

        def min_with_none(lst):
            lst = [ls for ls in lst if ls is not None]
            return min(lst)

        row_tokens = []
        row_labels = []
        input_ids = []
        attention_mask = []
        subword_labels = []
        predict_labels = []
        for document in data:
            text = document["tokens"]
            labels = document["labels"]
            max_length = self.padding - 2 if self.padding else len(text)

            for i, j in document["doc_index"]:
                subwords, word_ids = self.tokenize(" ".join(text[i:j]), p)
                row_tokens.append(text[i:j])
                row_labels.append(labels[i:j])
                masked_ids = copy.deepcopy(word_ids)

                while len(subwords) < max_length and j < len(text):
                    if j in [d[0] for d in document["doc_index"]]:
                        subwords = subwords + [self.sepToken]
                        word_ids = word_ids + [None]
                        masked_ids = masked_ids + [None]
                    ex_subwords = self.tokenizeWord(text[j])
                    subwords = subwords + ex_subwords
                    word_ids = word_ids + [max_with_none(word_ids) + 1] * len(ex_subwords)
                    masked_ids = masked_ids + [None] * len(ex_subwords)
                    j += 1
                    if len(subwords) < max_length:
                        subwords = subwords[:max_length]
                        word_ids = word_ids[:max_length]
                        masked_ids = masked_ids[:max_length]

                while len(subwords) < max_length and i > 0:
                    if i in [d[1] for d in document["doc_index"]]:
                        subwords = [self.sepToken] + subwords
                        word_ids = [None] + word_ids
                        masked_ids = [None] + masked_ids
                    i -= 1
                    ex_subwords = self.tokenizeWord(text[i])
                    subwords = ex_subwords + subwords
                    word_ids = [min_with_none(word_ids) - 1] * len(ex_subwords) + word_ids
                    masked_ids = [None] * len(ex_subwords) + masked_ids
                    if len(subwords) < max_length:
                        subwords = subwords[-max_length:]
                        word_ids = word_ids[:max_length]
                        masked_ids = masked_ids[:max_length]

                subwords = [self.clsTokenId] + [self.word2id[w] for w in subwords] + [self.sepTokenId]
                word_ids = [None] + word_ids + [None]
                masked_ids = [None] + masked_ids + [None]

                if self.padding:
                    if len(subwords) >= self.padding:
                        subwords = subwords[: self.padding]
                        word_ids = word_ids[: self.padding]
                        masked_ids = masked_ids[: self.padding]
                        mask = [1] * self.padding
                    else:
                        attention_len = len(subwords)
                        pad_len = self.padding - len(subwords)
                        subwords += [self.padTokenId] * pad_len
                        word_ids += [None] * pad_len
                        masked_ids += [None] * pad_len
                        mask = [1] * attention_len + [0] * pad_len
                else:
                    mask = [1] * len(subwords)

                input_ids.append(subwords)
                attention_mask.append(mask)

                label = labels[i:j]
                word_ids = [w_i - min_with_none(word_ids) if w_i is not None else None for w_i in word_ids]
                label_ids = self.get_label(word_ids, label, subword_label)
                subword_labels.append(label_ids)

                masked_label = row_labels[-1]
                masked_label_ids = self.get_label(masked_ids, masked_label, "PAD")
                predict_labels.append(masked_label_ids)

        if return_tensor:
            data = {
                "input_ids": torch.tensor(input_ids, dtype=torch.int),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.int),
                "subword_labels": torch.tensor(subword_labels, dtype=torch.int),
                "predict_labels": torch.tensor(predict_labels, dtype=torch.int),
                "tokens": row_tokens,
                "labels": row_labels,
            }
        else:
            data = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "subword_labels": subword_labels,
                "predict_labels": predict_labels,
                "tokens": row_tokens,
                "labels": row_labels,
            }
        return data


if __name__ == "__main__":
    sent = "The English data is a collection 's of news wire articles from the Reuters Corpus ."
    sent2 = "Converts a single index or a sequence of indices in a token or a sequence of tokens ."
    print(sent)

    mmt = MaxMatchTokenizer(p=0.3, padding=40)
    bert_tokeninzer = AutoTokenizer.from_pretrained("bert-base-cased")
    mmt.loadBertTokenizer(bertTokenizer=bert_tokeninzer, doNaivePreproc=True)

    tokens = mmt.tokenize([sent, sent2])
    # tokens = [list(chain.from_iterable(tkns)) for tkns in tokens]
    ids = [mmt.encode(tkns) for tkns in [sent, sent2]]
    # , truncation=True return_tensors="pt", max_length=80, padding="max_length")
    print(ids[0][0])
    print(ids[0][1])
    print(len(ids[0][1]))
    # print(ids.word_ids())
    print(bert_tokeninzer.decode(ids[0][0]))
