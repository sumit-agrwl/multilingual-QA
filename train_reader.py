# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options

import src.util
import src.evaluation
import src.data
import src.model
from tqdm import tqdm
import json


def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.seed)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=0,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    answers, best_answers = [], []
    while step < opt.total_steps:
        epoch += 1
        for i, batch in enumerate(tqdm(train_dataloader)):
            step += 1
            (idx, labels, _, context_ids, context_mask) = batch

            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda()
            )[0]

            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if step % opt.eval_freq == 0:
                dev_em, answers = evaluate(model, eval_dataset, tokenizer, collator, opt)
                if len(best_answers) == 0:
                    best_answers = answers
                model.train()
                if opt.is_main:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        best_answers = answers
                        src.util.save(model, optimizer, scheduler, step, best_dev_em,
                                  opt, checkpoint_path, 'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f} EM |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)    
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_em, step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                    curr_loss = 0.

            if opt.is_main and step % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_em,
                          opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                break
    return best_dev_em, best_answers

def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=collator
    )
    model.eval()
    total = 0
    exactmatch = []
    answers = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            (idx, _, _, context_ids, context_mask) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=50
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['answers']
                score = src.evaluation.ems(ans, gold)
                total += 1
                exactmatch.append(score)
                answers.append(ans)

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    return exactmatch, answers

def write_to_file(em, answers, opt):
    eval_data = json.load(open(opt.eval_data, encoding='utf-8'))
    ans_data = []
    for i, ans in enumerate(tqdm(answers)):
        data = {}
        data['prediction'] = ans
        data['q_id'] = eval_data[i]['q_id']
        data['question'] = eval_data[i]['question']
        data['answers'] = eval_data[i]['answers']
        data['lang'] = eval_data[i]['lang']
        ans_data.append(data)
    logger.info(f"Logging to {opt.output_file}")
    with open(opt.output_file, 'w', encoding='utf-8') as f:
        json.dump(ans_data, f)
    logger.info(f"Best dev EM : {em}")

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    #opt = options.get_options(use_reader=True, use_optim=True)
    opt.is_main = True
    opt.device = torch.device("cuda")
    torch.manual_seed(opt.seed)
    
    if opt.output_file is None:
        opt.output_file = f'output-{opt.name}.json'
        
    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    
    logger_path = Path('logger')/opt.name
    logger_path.mkdir(parents=True, exist_ok=True)
    
    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        logger_path / 'run.log'
    )

    model_name = 'google/mt5-' + opt.model_size
    model_class = src.model.FiDMT5

    #load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    collator = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)
    
    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data
    )
    train_dataset = src.data.Dataset(train_examples, opt.n_context)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context)
    checkpoint_exists = False
    if not checkpoint_exists and opt.model_path == "none":
        t5 = transformers.MT5ForConditionalGeneration.from_pretrained(model_name)
        model = src.model.FiDMT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(opt.device)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0
    elif opt.model_path == "none":
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {load_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        step = 0
        best_dev_em = 0.0
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)

    logger.info("Start training")
    best_em, answers = train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
    )
#     best_em, answers = evaluate(model, eval_dataset, tokenizer, collator, opt)
    write_to_file(best_em, answers, opt)
