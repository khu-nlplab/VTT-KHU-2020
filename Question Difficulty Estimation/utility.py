import csv
import logging
import tokenization
import os


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, question, des=None, utter=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.question = question
        self.des = des
        self.utter = utter
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, que_ids, que_mask, que_seg_ids, des_ids=None, des_mask=None, des_seg_ids=None, utter_ids=None, utter_mask=None, utter_seg_ids=None, label_id=None):
        self.question_ids = que_ids
        self.question_mask = que_mask
        self.question_segment_ids = que_seg_ids

        self.description_ids = des_ids
        self.description_mask = des_mask
        self.description_segment_ids = des_seg_ids

        self.utter_ids = utter_ids
        self.utter_mask = utter_mask
        self.utter_seg_ids = utter_seg_ids

        self.label_id = label_id

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):

        question = tokenizer.tokenize(example.question)

        question_tokens = []
        question_segment_ids = []
        question_tokens.append("[CLS]")
        question_segment_ids.append(0)

        for token in question:
            question_tokens.append(token)
            question_segment_ids.append(0)

        question_tokens.append("[SEP]")
        question_segment_ids.append(0)

        question_ids = tokenizer.convert_tokens_to_ids(question_tokens)
        question_mask = [1] * len(question_ids)

        if len(question_ids) < max_seq_length or len(question_ids) == max_seq_length:
            while len(question_ids) < max_seq_length:
                question_ids.append(0)
                question_mask.append(0)
                question_segment_ids.append(0)
        else:
            while len(question_ids) > max_seq_length:
                question_ids.pop()
                question_mask.pop()
                question_segment_ids.pop()

        assert len(question_ids) == max_seq_length
        assert len(question_mask) == max_seq_length
        assert len(question_segment_ids) == max_seq_length

        description = tokenizer.tokenize(example.des)

        description_tokens = []
        description_segment_ids = []
        description_tokens.append("[CLS]")
        description_segment_ids.append(0)

        for token in description:
            description_tokens.append(token)
            description_segment_ids.append(0)

        description_tokens.append("[SEP]")
        description_segment_ids.append(0)

        description_ids = tokenizer.convert_tokens_to_ids(description_tokens)
        description_mask = [1] * len(description_ids)

        if len(description_ids) < max_seq_length or len(description_ids) == max_seq_length:
            while len(description_ids) < max_seq_length:
                description_ids.append(0)
                description_mask.append(0)
                description_segment_ids.append(0)
        else:
            while len(description_ids) > max_seq_length:
                description_ids.pop()
                description_mask.pop()
                description_segment_ids.pop()

        assert len(description_ids) == max_seq_length
        assert len(description_mask) == max_seq_length
        assert len(description_segment_ids) == max_seq_length


        utterance = tokenizer.tokenize(example.utter)

        utterance_tokens = []
        utterance_segment_ids = []
        utterance_tokens.append("[CLS]")
        utterance_segment_ids.append(0)

        for token in utterance:
            utterance_tokens.append(token)
            utterance_segment_ids.append(0)

        utterance_tokens.append("[SEP]")
        utterance_segment_ids.append(0)

        utterance_ids = tokenizer.convert_tokens_to_ids(utterance_tokens)
        utterance_mask = [1] * len(utterance_ids)

        if len(utterance_ids) < max_seq_length or len(utterance_ids) == max_seq_length:
            while len(utterance_ids) < max_seq_length:
                utterance_ids.append(0)
                utterance_mask.append(0)
                utterance_segment_ids.append(0)
        else:
            while len(utterance_ids) > max_seq_length:
                utterance_ids.pop()
                utterance_mask.pop()
                utterance_segment_ids.pop()

        assert len(utterance_ids) == max_seq_length
        assert len(utterance_mask) == max_seq_length
        assert len(utterance_segment_ids) == max_seq_length

        label_id = None

        features.append(
            InputFeatures(
                que_ids=question_ids,
                que_mask=question_mask,
                que_seg_ids=question_segment_ids,
                des_ids=description_ids,
                des_mask=description_mask,
                des_seg_ids=description_segment_ids,
                utter_ids=utterance_ids,
                utter_mask=utterance_mask,
                utter_seg_ids=utterance_segment_ids,
                label_id=label_id)
        )

    return features