import csv
import logging


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

import tokenization
import os
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, question, des=None, ans=None, utter=None, label=None):
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
        self.ans = ans
        self.utter = utter
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, que_ids, que_mask, que_seg_ids, des_ids=None, des_mask=None, des_seg_ids=None, ans_ids=None,
                 ans_mask=None, ans_seg_ids=None, utter_ids=None, utter_mask=None, utter_seg_ids=None, label_id=None):
        self.question_ids = que_ids
        self.question_mask = que_mask
        self.question_segment_ids = que_seg_ids

        self.description_ids = des_ids
        self.description_mask = des_mask
        self.description_segment_ids = des_seg_ids

        self.answer_ids = ans_ids
        self.answer_mask = ans_mask
        self.answer_segment_ids = ans_seg_ids

        self.utter_ids = utter_ids
        self.utter_mask = utter_mask
        self.utter_seg_ids = utter_seg_ids

        self.label_id = label_id
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf8') as f:
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                if len(line) < 2:
                    pass
                else:
                    lines.append(line)

            return lines

class FriendsQAProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    def get_test_examples(self, data_dir):

        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")


    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            que = tokenization.convert_to_unicode(line[0])
            des = tokenization.convert_to_unicode(line[1])
            tmp=''
            for sent in eval(line[2]):
                tmp += sent + '[SEP]'
            utter = tokenization.convert_to_unicode(tmp)
            ans = tokenization.convert_to_unicode(line[3])

            m_label = tokenization.convert_to_unicode(line[-1])

            examples.append(
                InputExample(guid=guid, question=que, des=des, ans=ans, utter=utter, label=m_label))
        return examples

class DramaQAProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):

        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")


    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            que = tokenization.convert_to_unicode(line[1])
            des = tokenization.convert_to_unicode(lines[i-1][2])
            ans = tokenization.convert_to_unicode(lines[i-1][3])
            utter = tokenization.convert_to_unicode(line[4].replace('[UTR]', ''))

            l_label = tokenization.convert_to_unicode(line[-1])

            examples.append(
                InputExample(guid=guid, question=que, des=des, ans=ans, utter=utter, label=l_label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

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

        if example.des is not None:
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

        else:
            description_ids = None
            description_mask = None
            description_segment_ids = None

        if example.ans is not None:
            answer = tokenizer.tokenize(example.ans)

            answer_tokens = []
            answer_segment_ids = []
            answer_tokens.append("[CLS]")
            answer_segment_ids.append(0)

            for token in answer:
                answer_tokens.append(token)
                answer_segment_ids.append(0)

            answer_tokens.append("[SEP]")
            answer_segment_ids.append(0)

            answer_ids = tokenizer.convert_tokens_to_ids(answer_tokens)
            answer_mask = [1] * len(answer_ids)

            if len(answer_ids) < max_seq_length or len(answer_ids) == max_seq_length:
                while len(answer_ids) < max_seq_length:
                    answer_ids.append(0)
                    answer_mask.append(0)
                    answer_segment_ids.append(0)
            else:
                while len(answer_ids) > max_seq_length:
                    answer_ids.pop()
                    answer_mask.pop()
                    answer_segment_ids.pop()

            assert len(answer_ids) == max_seq_length
            assert len(answer_mask) == max_seq_length
            assert len(answer_segment_ids) == max_seq_length

        else:
            answer_ids = None
            answer_mask = None
            answer_segment_ids = None

        if example.utter is not None:
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

        else:
            utterance_ids = None
            utterance_mask = None
            utterance_segment_ids = None
        if example.label is not None:
            label_id = label_map[example.label]
        else:
            label_id = None

        features.append(
            InputFeatures(
                que_ids=question_ids,
                que_mask=question_mask,
                que_seg_ids=question_segment_ids,
                des_ids=description_ids,
                des_mask=description_mask,
                des_seg_ids=description_segment_ids,
                ans_ids=answer_ids,
                ans_mask=answer_mask,
                ans_seg_ids=answer_segment_ids,
                utter_ids=utterance_ids,
                utter_mask=utterance_mask,
                utter_seg_ids=utterance_segment_ids,
                label_id=label_id)
        )
    return features