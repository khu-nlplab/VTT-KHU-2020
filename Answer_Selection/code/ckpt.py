import os

import torch, pickle
from pathlib import Path

from dataloader.dataset_multichoice import get_iterator as get_iterator_multichoice
from model import get_model
from utils import get_dirname_from_args

except_key_list = ["path", "input", "print_output"]

def get_ckpt_path(args, epoch, loss):
    ckpt_name = get_dirname_from_args(args)
    ckpt_path = args.ckpt_path / ckpt_name

    args.ckpt_path.mkdir(exist_ok=True)
    ckpt_path.mkdir(exist_ok=True)
    loss = '{:.4f}'.format(loss)
    ckpt_path = ckpt_path / 'loss_{}_epoch_{}.pickle'.format(loss, epoch)

    return ckpt_path

def save_ckpt(args, epoch, loss, model, vocab):
    print('saving epoch {}'.format(epoch))
    dt = {
        'args': args,
        'epoch': epoch,
        'loss': loss,
        'model': model.state_dict(),
        'vocab': vocab,
    }
    ckpt_path = get_ckpt_path(args, epoch, loss)
    print("Saving checkpoint {}".format(ckpt_path))
    torch.save(dt, str(ckpt_path))

def get_model_ckpt(args):
    ckpt_available = args.ckpt_name is not None
    vocab = None

    dt = None

    if ckpt_available:
        name = '{}'.format(args.ckpt_name)
        name = '{}*'.format(name) if not name.endswith('*') else name
        ckpt_paths = sorted(args.ckpt_path.glob(name), reverse=False)
        assert len(ckpt_paths) > 0, "no ckpt candidate for {}".format(args.ckpt_path / args.ckpt_name)
        ckpt_path = ckpt_paths[0]  # monkey patch for choosing the best ckpt
        print("loading from {}".format(ckpt_path))
        dt = torch.load(ckpt_path)
        args = update_args(args, dt['args'])
        vocab = dt['vocab']

    if args.input is not None and dt is not None:
        return args, dt

    iters, vocab = get_iterator_multichoice(args, vocab)
    model = get_model(args, vocab)
    model.load_embedding(vocab)

    if ckpt_available:
        model.load_state_dict(dt['model'])

    return args, model, iters, vocab, ckpt_available

def update_args(o_args, n_args):

    for key, value in n_args.items():

        do_update = True
        for except_key in except_key_list:
            if except_key in key:
                do_update = False

        if do_update:
            o_args.update({key: value})

    return o_args