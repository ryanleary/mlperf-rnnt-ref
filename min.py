import argparse
import itertools
import os
import time
import toml
import torch
import apex
from apex import amp
import random
import numpy as np
import math
from dataset import AudioToTextDataLayer
from helpers import monitor_asr_train_progress, process_evaluation_batch, process_evaluation_epoch, Optimization, add_blank_label, AmpOptimizations, model_multi_gpu, print_dict, print_once
from model_rnnt import AudioPreprocessing, RNNT
from decoders import RNNTGreedyDecoder
from loss import RNNTLoss
from optimizers import Novograd, AdamW
from tb_logger import DummyLogger, TensorBoardLogger


def repro(greedy_decoder, t_audio_signal_t, t_a_sig_length_t):
    t_predictions_t = greedy_decoder.decode(t_audio_signal_t, t_a_sig_length_t)

def lr_policy(initial_lr, step, N):
    """
    learning rate decay
    Args:
        initial_lr: base learning rate
        step: current iteration number
        N: total number of iterations over which learning rate is decayed
    """
    min_lr = 0.00001
    res = initial_lr * ((N - step) / N) ** 2
    return max(res, min_lr)


def setup():
    parser = argparse.ArgumentParser(description='RNNT Training Reference')
    parser.add_argument("--local_rank", default=None, type=int)
    parser.add_argument("--batch_size", default=16, type=int, help='data batch size')
    parser.add_argument("--eval_batch_size", default=1, type=int, help='eval data batch size')
    parser.add_argument("--num_epochs", default=10, type=int, help='number of training epochs. if number of steps if specified will overwrite this')
    parser.add_argument("--num_steps", default=None, type=int, help='if specified overwrites num_epochs and will only train for this number of iterations')
    parser.add_argument("--save_freq", dest="save_frequency", default=300, type=int, help='number of epochs until saving checkpoint. will save at the end of training too.')
    parser.add_argument("--eval_freq", dest="eval_frequency", default=200, type=int, help='number of iterations until doing evaluation on full dataset')
    parser.add_argument("--train_freq", dest="train_frequency", default=25, type=int, help='number of iterations until printing training statistics on the past iteration')
    parser.add_argument("--lr", default=1e-3, type=float, help='learning rate')
    parser.add_argument("--weight_decay", default=1e-3, type=float, help='weight decay rate')
    parser.add_argument("--train_manifest", type=str, required=True, help='relative path given dataset folder of training manifest file')
    parser.add_argument("--model_toml", type=str, required=True, help='relative path given dataset folder of model configuration file')
    parser.add_argument("--val_manifest", type=str, required=True, help='relative path given dataset folder of evaluation manifest file')
    parser.add_argument("--max_duration", type=float, help='maximum duration of audio samples for training and evaluation')
    parser.add_argument("--pad_to_max", action="store_true", default=False, help="pad sequence to max_duration")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help='number of accumulation steps')
    parser.add_argument("--optimizer", dest="optimizer_kind", default="novograd", type=str, help='optimizer')
    parser.add_argument("--dataset_dir", dest="dataset_dir", required=True, type=str, help='root dir of dataset')
    parser.add_argument("--lr_decay", action="store_true", default=False, help='use learning rate decay')
    parser.add_argument("--cudnn", action="store_true", default=False, help="enable cudnn benchmark")
    parser.add_argument("--fp16", action="store_true", default=False, help="use mixed precision training")
    parser.add_argument("--output_dir", type=str, required=True, help='saves results in this directory')
    parser.add_argument("--ckpt", default=None, type=str, help="if specified continues training from given checkpoint. Otherwise starts from beginning")
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--tb_path", default=None, type=str, help='where to store tensorboard data')
    parser.add_argument("--histogram", default=False, action='store_true', help='whether to log param and grad histograms')
    args=parser.parse_args('--batch_size=8 --eval_batch_size=2 --num_epochs=100 --output_dir=/results --model_toml=configs/rnnt.toml --lr=0.011 --seed=6 --optimizer=adam --dataset_dir=/datasets/LibriSpeech --val_manifest=/datasets/LibriSpeech/librispeech-dev-clean-wav.json --train_manifest=/datasets/LibriSpeech/librispeech-train-clean-100-wav.json,/datasets/LibriSpeech/librispeech-train-clean-360-wav.json,/datasets/LibriSpeech/librispeech-train-other-500-wav.json --weight_decay=0.001 --save_freq=10 --eval_freq=1000 --train_freq=25 --gradient_accumulation_steps=1 --fp16 --cudnn --tb_path /home/samgd/logs/rnnt/repro/full_tr_spec_drop_spd/LR0.011_BS8_adam_ACC1_a1112ec/18:56:411575655001'.split(' '))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    assert(torch.cuda.is_available())
    torch.backends.cudnn.benchmark = args.cudnn

    args.local_rank = os.environ.get('LOCAL_RANK', args.local_rank)
    # set up distributed training
    if args.local_rank is not None:
        args.local_rank = int(args.local_rank)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    multi_gpu = torch.distributed.is_initialized()
    if multi_gpu:
        print_once("DISTRIBUTED TRAINING with {} gpus".format(torch.distributed.get_world_size()))

    # define amp optimiation level
    if args.fp16:
        optim_level = Optimization.mxprO1
    else:
        optim_level = Optimization.mxprO0

    model_definition = toml.load(args.model_toml)
    dataset_vocab = model_definition['labels']['labels']
    ctc_vocab = add_blank_label(dataset_vocab)

    train_manifest = args.train_manifest
    val_manifest = args.val_manifest
    featurizer_config = model_definition['input']
    featurizer_config_eval = model_definition['input_eval']
    featurizer_config["optimization_level"] = optim_level
    featurizer_config_eval["optimization_level"] = optim_level

    sampler_type = featurizer_config.get("sampler", 'default')
    perturb_config = model_definition.get('perturb', None)
    if args.pad_to_max:
        assert(args.max_duration > 0)
        featurizer_config['max_duration'] = args.max_duration
        featurizer_config_eval['max_duration'] = args.max_duration
        featurizer_config['pad_to'] = "max"
        featurizer_config_eval['pad_to'] = "max"
    print_once('model_config')
    print_dict(model_definition)

    if args.gradient_accumulation_steps < 1:
        raise ValueError('Invalid gradient accumulation steps parameter {}'.format(args.gradient_accumulation_steps))
    if args.batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError('gradient accumulation step {} is not divisible by batch size {}'.format(args.gradient_accumulation_steps, args.batch_size))


    data_layer = AudioToTextDataLayer(
                                    dataset_dir=args.dataset_dir,
                                    featurizer_config=featurizer_config,
                                    perturb_config=perturb_config,
                                    manifest_filepath=train_manifest,
                                    labels=dataset_vocab,
                                    batch_size=args.batch_size // args.gradient_accumulation_steps,
                                    multi_gpu=multi_gpu,
                                    pad_to_max=args.pad_to_max,
                                    sampler=sampler_type)

    data_layer_eval = AudioToTextDataLayer(
                                    dataset_dir=args.dataset_dir,
                                    featurizer_config=featurizer_config_eval,
                                    manifest_filepath=val_manifest,
                                    labels=dataset_vocab,
                                    batch_size=args.eval_batch_size,
                                    multi_gpu=multi_gpu,
                                    pad_to_max=args.pad_to_max
                                    )

    model = RNNT(
        feature_config=featurizer_config,
        rnnt=model_definition['rnnt'],
        num_classes=len(ctc_vocab)
    )

    if args.ckpt is not None:
        print_once("loading model from {}".format(args.ckpt))
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        args.start_epoch = checkpoint['epoch']
    else:
        args.start_epoch = 0

    loss_fn = RNNTLoss(blank=len(ctc_vocab) - 1)

    N = len(data_layer)
    if sampler_type == 'default':
        args.step_per_epoch = math.ceil(N / (args.batch_size * (1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size())))
    elif sampler_type == 'bucket':
        args.step_per_epoch = int(len(data_layer.sampler) / args.batch_size )

    print_once('-----------------')
    print_once('Have {0} examples to train on.'.format(N))
    print_once('Have {0} steps / (gpu * epoch).'.format(args.step_per_epoch))
    print_once('-----------------')

    fn_lr_policy = lambda s: lr_policy(args.lr, s, args.num_epochs * args.step_per_epoch)


    model.cuda()


    if args.optimizer_kind == "novograd":
        optimizer = Novograd(model.parameters(),
                        lr=args.lr,
                        weight_decay=args.weight_decay)
    elif args.optimizer_kind == "adam":
        optimizer = AdamW(model.parameters(),
                        lr=args.lr,
                        weight_decay=args.weight_decay)
    else:
        raise ValueError("invalid optimizer choice: {}".format(args.optimizer_kind))

    if optim_level in AmpOptimizations:
        model, optimizer = amp.initialize(
            min_loss_scale=0.125,
            models=model,
            optimizers=optimizer,
            opt_level=AmpOptimizations[optim_level]
        )

    if args.ckpt is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    model = model_multi_gpu(model, multi_gpu)
    print_once(model)
    print_once("# parameters: {}".format(sum(p.numel() for p in model.parameters())))
    greedy_decoder = RNNTGreedyDecoder(len(ctc_vocab) - 1, model.module if multi_gpu else model)

    if args.tb_path and args.local_rank == 0:
        logger = TensorBoardLogger(args.tb_path, model.module if multi_gpu else model, args.histogram)
    else:
        logger = DummyLogger()

    print_once("Starting .....")
    start_time = time.time()

    train_dataloader = data_layer.data_iterator
    epoch = args.start_epoch
    step = epoch * args.step_per_epoch

    while True:
        if multi_gpu:
            data_layer.sampler.set_epoch(epoch)
        print_once("Starting epoch {0}, step {1}".format(epoch, step))
        last_epoch_start = time.time()
        batch_counter = 0
        average_loss = 0
        for data in train_dataloader:
            tensors = []
            for d in data:
                if isinstance(d, torch.Tensor):
                    tensors.append(d.cuda())
                else:
                    tensors.append(d)

            if batch_counter == 0:

                if fn_lr_policy is not None:
                    adjusted_lr = fn_lr_policy(step)
                    for param_group in optimizer.param_groups:
                            param_group['lr'] = adjusted_lr
                optimizer.zero_grad()
                last_iter_start = time.time()

            t_audio_signal_t, t_a_sig_length_t, t_transcript_t, t_transcript_len_t = tensors
            model.train()

            t_log_probs_t, (x_len, y_len) = model(
                ((t_audio_signal_t, t_transcript_t), (t_a_sig_length_t, t_transcript_len_t)),
            )

            t_loss_t = loss_fn(
                (t_log_probs_t, x_len), (t_transcript_t, y_len)
            )
            logger.log_scalar('loss', t_loss_t.item(), step)
            del t_log_probs_t
            if args.gradient_accumulation_steps > 1:
                t_loss_t = t_loss_t / args.gradient_accumulation_steps

            if optim_level in AmpOptimizations:
                with amp.scale_loss(t_loss_t, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                t_loss_t.backward()
            batch_counter += 1
            average_loss += t_loss_t.item()

            if batch_counter % args.gradient_accumulation_steps == 0:
                optimizer.step()
                return greedy_decoder, t_audio_signal_t, t_a_sig_length_t


def main():
    repro(*setup())


if __name__ =="__main__":
    main()
