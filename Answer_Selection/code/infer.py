import json

import torch
from torch.nn import functional as F
from ignite.engine.engine import Engine, State, Events
from metric.stat_metric import StatMetric

from ckpt import get_model_ckpt
from model import get_model
from loss import get_loss
from logger import log_results_cmd

from utils import prepare_batch
from metric import get_metrics
from dataloader.dataset_multichoice import get_iterator, get_cache_data, preprocess_input


def get_evaluator(args, model, loss_fn):
    def _inference(evaluator, batch):
        model.eval()
        with torch.no_grad():
            qids = batch["new_qid"]
            net_inputs, _ = prepare_batch(args, batch, model.vocab)
            y_pred = model(**net_inputs)
            y_soft_output = y_pred #F.softmax(y_pred, dim=-1)
            #y_soft_output = F.softmax(y_pred, dim=-1)

            for qid, soft_output in zip(qids, y_soft_output):

                if soft_output.dim() > 1:
                    pred_tuple = soft_output.unbind(dim=0)
                    final_pred = soft_output.sum(dim=0)
                    engine.answers[qid] = {'correct_idx' : final_pred.argmax().item(),
                                         'soft_output' : final_pred.tolist()}

                    for idx, pred in enumerate(pred_tuple):
                        engine.answers[qid].update({'model_{}_output'.format(idx+1) : pred.tolist()})

                else:
                    engine.answers[qid] = {'correct_idx': soft_output.argmax().item(),
                                           'soft_output': soft_output.tolist()}

            return

    engine = Engine(_inference)
    engine.answers = {}

    return engine


def evaluate_once(evaluator, iterator):
    evaluator.run(iterator)
    return evaluator.answers


def prepare_model(args):
    args, dt = get_model_ckpt(args)
    if dt is None:
        print("check the ckpt file : {} ".args.ckpt_name)

    vocab = dt['vocab']
    model = get_model(args, vocab)
    model.load_embedding(vocab)
    model.load_state_dict(dt['model'])

    args, vocab, image, text = get_cache_data(args, vocab)

    cache = {"vocab" : vocab,
             "model" : model,
             "image" : image,
             "text" : text}

    return args, cache

def infer(args, cache=None):

    if args.input is None:
        args, model, iters, vocab, ckpt_available = get_model_ckpt(args)
        if ckpt_available:
            print("loaded checkpoint {}".format(args.ckpt_name))

    else:
        model = cache["model"]
        vocab = cache["vocab"]
        image = cache["image"]
        text = cache["text"]
        candidate_answers = args.input['answers']
        iters = preprocess_input(args, vocab, text, image)

    loss_fn = get_loss(args, vocab)
    evaluator = get_evaluator(args, model, loss_fn)

    answers = evaluate_once(evaluator, iterator=iters['infer'])
    keys = sorted(list(answers.keys()))
    answers = [{"qid": key,
                **answers[key]} for key in keys]
    if args.print_output and args.input is not None:
        for answer in answers:
            print("correct_idx : {}\nanswer : {}".format(answer['correct_idx'],
                                                     candidate_answers[answer['correct_idx']]))
    path = str(args.data_path.parent / '{}_answers.json'.format(args.model_name))
    with open(path, 'w') as f:
        json.dump(answers, f, indent=4)
    # print("saved outcome at {}".format(path))

