#!/usr/bin/env python3

import os
import argparse
import torch.nn as nn
import tqdm, torch
from model_postag import RNN_postag
from collections import defaultdict
from use_conllulib import CoNLLUReader, Util
from torch.utils.data import TensorDataset, DataLoader
from train_postag import read_corpus
from conllu import parse_incr
from conllu.serializer import serialize
from conllu.models import TokenList


def get_args():
    parser = argparse.ArgumentParser(
        description="POS tagging inference with trained RNN model"
    )

    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Fichier CoNLL-U à annoter (dev / test)",
    )

    parser.add_argument(
        "-m",
        "--model_file",
        type=str,
        default="model.pt",
        help="Fichier du modèle entraîné (.pt)",
    )

    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="pred.conllu",
        help="Fichier de sortie CoNLL-U annoté",
    )

    parser.add_argument(
        "-l", "--max_len", type=int, default=40, help="Longueur maximale des séquences"
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (utilisé pour le padding)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Chargement du modèle
    load_dict = torch.load(args.model_file, weights_only=False)

    wordvocab = load_dict["wordvocab"]
    tagvocab = load_dict["tagvocab"]
    hp = load_dict["hyperparams"]

    num_embeddings = len(wordvocab)
    output_size = len(tagvocab)

    model = RNN_postag(
        embedding_dim=hp["embedding_dim"],
        hidden_size=hp["hidden_size"],
        num_embeddings=num_embeddings,
        output_size=output_size,
    )
    model.load_state_dict(load_dict["model_params"])
    model.eval()

    # Lecture du corpus (sans batching)
    words, _, _, _ = read_corpus(
        filename=args.input_file,
        wordvocab=wordvocab,
        tagvocab=tagvocab,
        max_len=args.max_len,
        batch_size=args.batch_size,
        train_mode=False,
        batch_mode=False,
    )

    revtagvocab = Util.rev_vocab(tagvocab)
    revwordvocab = Util.rev_vocab(wordvocab)

    # Métadonnées or
    gold_metadata = []
    for sent in parse_incr(open(args.input_file, encoding="UTF-8")):
        gold_metadata.append(sent.metadata)

    sentences = []

    for i, sent in enumerate(words):
        with torch.no_grad():
            logits = model(torch.LongTensor([sent]))[0]

        forms = [revwordvocab[w] for w in sent]
        upos = [revtagvocab[l.argmax().item()] for l in logits]

        conllu_format = [
            {
                "id": idx,
                "form": w,
                "lemma": "_",
                "upos": t,
                "xpos": "_",
                "feats": "_",
                "head": "_",
                "deprel": "_",
                "deps": "_",
                "misc": "_",
            }
            for idx, (w, t) in enumerate(zip(forms, upos), start=1)
        ]

        sentences.append(TokenList(conllu_format, gold_metadata[i]))

    # Écriture du fichier de sortie
    os.makedirs("pred", exist_ok=True)
    model_name = os.path.splitext(os.path.basename(args.model_file))[0]

    with open(
        f"pred/{args.output_file}_{model_name}_l{args.max_len}_bs{args.batch_size}",
        "w",
        encoding="utf-8",
    ) as f:
        f.writelines([s.serialize() + "\n" for s in sentences])
