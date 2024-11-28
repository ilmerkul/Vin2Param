import pandas as pd
import numpy as np


class Vin2ParamTokenizer:
    def __init__(self, sos_token, eos_token):
        self.sos_token = sos_token
        self.eos_token = eos_token

        self.vocab = None
        self.token2id = None
        self.id2token = None

        self.brand2id = None
        self.model2id = None
        self.color2id = None

        self.id2brand = None
        self.id2model = None
        self.id2color = None

    def fit_vin(self, vins: pd.DataFrame):
        self.vocab = set()
        for vin in vins["VIN"].apply(lambda x: list(x)).values:
            for x in vin:
                self.vocab.add(x)

        self.token2id = {token: i + 2 for i, token in enumerate(self.vocab)}
        self.token2id[self.sos_token] = 1
        self.token2id[self.eos_token] = 0
        self.vocab.add(self.sos_token)
        self.vocab.add(self.eos_token)

        self.id2token = {id: token for token, id in self.token2id.items()}

    def tokenize_vin(self, vins: pd.DataFrame):
        assert self.vocab is not None

        vins["VIN"] = vins["VIN"].apply(lambda x:
                                        [1] + list(map(lambda y:
                                                       self.token2id[y],
                                                       list(x))) + [0])

    def fit_label(self, labels: pd.DataFrame):
        self.brand2id = {brand: i for i, brand in
                         enumerate(labels["CarBrand"].unique())}
        self.model2id = {model: i for i, model in
                         enumerate(labels["CarModel"].unique())}
        self.color2id = {color: i for i, color in
                         enumerate(labels["Color"].unique())}

        self.id2brand = {id: brand for brand, id in self.brand2id.items()}
        self.id2model = {id: model for model, id in self.model2id.items()}
        self.id2color = {id: color for color, id in self.color2id.items()}

    def tokenize_label(self, labels: pd.DataFrame):
        assert self.brand2id is not None and \
               self.model2id is not None and \
               self.color2id is not None

        labels["CarBrand"] = labels["CarBrand"].apply(
            lambda x: self.brand2id[x]).values
        labels["CarModel"] = labels["CarModel"].apply(
            lambda x: self.model2id[x]).values
        labels["Color"] = labels["Color"].apply(
            lambda x: self.color2id[x]).values
