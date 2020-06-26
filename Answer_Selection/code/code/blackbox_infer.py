import json

import torch
from ignite.engine.engine import Engine, State, Events
from ignite.contrib.handlers import ProgressBar
from metric.stat_metric import StatMetric

from ckpt import get_model_ckpt
from model import get_model
from loss import get_loss
from logger import log_results_cmd

from utils import prepare_batch
from metric import get_metrics


def get_evaluator(args, model, loss_fn):
    def _inference(evaluator, batch):
        model.eval()
        with torch.no_grad():
            new_qids = batch["new_qid"]
            real_ans = batch["correct_idx"]
            net_inputs, _ = prepare_batch(args, batch, model.vocab)

            y_pred = model(**net_inputs)
            y_pred = y_pred.argmax(dim=-1)  # + 1  # 0~4 -> 1~5

            for new_qid, ans, r_ans in zip(new_qids, y_pred, real_ans):
                engine.answers[new_qid] = ans.item()
                if ans.item() == r_ans:
                    engine.count += 1
            return

    engine = Engine(_inference)
    engine.answers = {}
    engine.count = 0
    return engine


def evaluate_once(evaluator, iterator):
    '''
    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_iter_results(engine):
        print("iteration : ", engine.state.iteration)
        print(engine.)
    '''
    # ProgressBar(persist=False).attach(evaluator)
    evaluator.run(iterator)
    # print(len(evaluator.answers))
    # print((evaluator.count) )
    # print(evaluator.count / len(evaluator.answers))
    return evaluator.answers


def blackbox_infer(args):

    args, model, iters, vocab, ckpt_available = get_model_ckpt(args)
    if ckpt_available:
        print("loaded checkpoint {}".format(args.ckpt_name))
    loss_fn = get_loss(args, vocab)

    evaluator = get_evaluator(args, model, loss_fn)

    answers = evaluate_once(evaluator, iterator=iters['train'])
    keys = sorted(list(answers.keys()))
    answers = [{"correct_idx": answers[key], "new_qid": key} for key in keys]
    path = str(args.data_path.parent / '{}_train_answers.json'.format(args.model_name))
    with open(path, 'w') as f:
        json.dump(answers, f, indent=4)

    print("saved outcome at {}".format(path))
