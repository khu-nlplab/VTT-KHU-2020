from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import tokenization
from Level_Prediction_model import SequenceClassification, BertConfig
from utility import InputExample, InputFeatures, convert_examples_to_features

class LevelClassificationModel():
    def __init__(self):
        #data load
        super(LevelClassificationModel, self).__init__()
        bert_config_file = "../uncased_L-12_H-768_A-12/bert_config.json"
        vocab_file = "../uncased_L-12_H-768_A-12/vocab.txt"
        dropout_prob = 0.2
        memory_model_path = "model/Memory_level_model.bin"
        logical_model_path = "model/Logical_level_model.bin"
        config = BertConfig.from_json_file(bert_config_file)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

        #model definition
        self.memory_level_model = SequenceClassification(config, dropout_prob, num_labels=2)
        self.logical_level_model = SequenceClassification(config, dropout_prob, num_labels=4)

        #model load
        self.memory_level_model.load_state_dict(torch.load(memory_model_path, map_location='cpu'))
        self.logical_level_model.load_state_dict(torch.load(logical_model_path, map_location='cpu'))

    def data_loader(self, question, description=None, answer=None, utterance=None):
        examples = []
        example = InputExample(guid='DramaQA input examples: ', question=question, des=description, ans=answer,
                               utter=utterance)
        examples.append(example)
        memory_label_list = ["2", "3"]
        logic_label_list = ["1", "2", "3", "4"]

        memory_input_feature = convert_examples_to_features(examples, memory_label_list, max_seq_length=64,
                                                            tokenizer=self.tokenizer)
        logical_input_feature = convert_examples_to_features(examples, logic_label_list, max_seq_length=64,
                                                             tokenizer=self.tokenizer)
        input_data = []

        input_q_ids = torch.tensor([f.question_ids for f in memory_input_feature], dtype=torch.long)
        input_q_masks = torch.tensor([f.question_mask for f in memory_input_feature], dtype=torch.long)
        input_q_seg = torch.tensor([f.question_segment_ids for f in memory_input_feature], dtype=torch.long)
        input_data.append([input_q_ids, input_q_masks, input_q_seg])
        try:
            input_d_ids = torch.tensor([f.description_ids for f in memory_input_feature], dtype=torch.long)
            input_d_masks = torch.tensor([f.description_mask for f in memory_input_feature], dtype=torch.long)
            input_d_seg = torch.tensor([f.description_segment_ids for f in memory_input_feature], dtype=torch.long)
            input_data.append([input_d_ids, input_d_masks, input_d_seg])
        except :
            pass

        try:
            input_a_ids = torch.tensor([f.answer_ids for f in memory_input_feature], dtype=torch.long)
            input_a_masks = torch.tensor([f.answer_mask for f in memory_input_feature], dtype=torch.long)
            input_a_seg = torch.tensor([f.answer_segment_ids for f in memory_input_feature], dtype=torch.long)
            input_data.append([input_a_ids, input_a_masks, input_a_seg])
        except :
            pass
        try :
            input_u_ids = torch.tensor([f.utter_ids for f in memory_input_feature], dtype=torch.long)
            input_u_masks = torch.tensor([f.utter_mask for f in memory_input_feature], dtype=torch.long)
            input_u_seg = torch.tensor([f.utter_seg_ids for f in memory_input_feature], dtype=torch.long)
            input_data.append([input_u_ids, input_u_masks, input_u_seg])
        except :
            pass

        return input_data, memory_label_list, logic_label_list

    def convert_id_to_label(self, label_, label_list):
        label_map = {}
        for (i, labels) in enumerate(label_list):
            label_map[i] = labels
        label = label_
        return label_map[label]

    def predict(self, question, utterance, candidate_answers):

        input_data, memory_label_map, logical_label_map = self.data_loader(question, answer=candidate_answers, utterance=utterance)

        memory_logits = self.memory_level_model(q_vec=input_data[0][0], q_mask=input_data[0][1], q_segment=input_data[0][2],
                                          d_vec=None, d_mask=None, d_segment=None,
                                          a_vec=input_data[1][0], a_mask=input_data[1][1], a_segment=input_data[1][2],
                                          u_vec=input_data[2][0], u_mask=input_data[2][1], u_segment=input_data[2][2])

        logical_logits = self.logical_level_model(q_vec=input_data[0][0], q_mask=input_data[0][1], q_segment=input_data[0][2],
                                            d_vec=None, d_mask=None, d_segment=None,
                                            a_vec=input_data[1][0], a_mask=input_data[1][1], a_segment=input_data[1][2],
                                            u_vec=input_data[2][0], u_mask=input_data[2][1], u_segment=input_data[2][2])

        memory_logits = memory_logits.to('cpu').detach().numpy()
        logical_logits = logical_logits.to('cpu').detach().numpy()

        memory_level = np.argmax(memory_logits, axis=1)
        logical_level = np.argmax(logical_logits, axis=1)

        memory_level = self.convert_id_to_label(memory_level[0], memory_label_map)
        logical_level = self.convert_id_to_label(logical_level[0], logical_label_map)

        return memory_level, logical_level