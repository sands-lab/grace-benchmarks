# Copyright (c) 2018, deepakn94, codyaustun, robieta. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -----------------------------------------------------------------------
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.jit
from apex.optimizers import FusedAdam
import os
import math
import time
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn

import utils
import dataloading
from neumf import NeuMF

import dllogger

from apex.parallel import DistributedDataParallel as DDP
# from torch.nn.parallel import DistributedDataParallel as DDP
from apex import amp

import datetime
import socket
from mpi4py import MPI
import wandb
import grace_dl
import torch.distributed as dist


def parse_args():
    parser = ArgumentParser(description="Train a Nerual Collaborative"
                                        " Filtering model")
    parser.add_argument('--data', type=str,
                        help='Path to test and training data files')
    parser.add_argument('-e', '--epochs', type=int, default=30,
                        help='Number of epochs for training')
    parser.add_argument('-b', '--batch_size', type=int, default=(2**20),
                        help='Number of examples for each iteration')
    parser.add_argument('--valid_batch_size', type=int, default=2**20,
                        help='Number of examples in each validation chunk')
    parser.add_argument('-f', '--factors', type=int, default=64,
                        help='Number of predictive factors')
    parser.add_argument('--layers', nargs='+', type=int,
                        default=[256, 256, 128, 64],
                        help='Sizes of hidden layers for MLP')
    parser.add_argument('-n', '--negative_samples', type=int, default=4,
                        help='Number of negative examples per interaction')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.0045,
                        help='Learning rate for optimizer')
    parser.add_argument('-k', '--topk', type=int, default=10,
                        help='Rank for test examples to be considered a hit')
    parser.add_argument('--seed', '-s', type=int, default=None,
                        help='Manually set random seed for torch')
    parser.add_argument('--threshold', '-t', type=float, default=1.0,
                        help='Stop training early at threshold')
    parser.add_argument('--beta1', '-b1', type=float, default=0.25,
                        help='Beta1 for Adam')
    parser.add_argument('--beta2', '-b2', type=float, default=0.5,
                        help='Beta1 for Adam')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='Epsilon for Adam')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability, if equal to 0 will not use dropout at all')
    parser.add_argument('--checkpoint_dir', default='', type=str,
                        help='Path to the directory storing the checkpoint file, passing an empty path disables checkpoint saving')
    parser.add_argument('--load_checkpoint_path', default=None, type=str,
                        help='Path to the checkpoint file to be loaded before training/evaluation')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', type=str,
                        help='Passing "test" will only run a single evaluation, otherwise full training will be performed')
    parser.add_argument('--grads_accumulated', default=1, type=int,
                        help='Number of gradients to accumulate before performing an optimization step')
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--log_path', default='log.json', type=str,
                        help='Path for the JSON training log')
    parser.add_argument('--world_size', type=int, default=1, help='world size')
    parser.add_argument('--local_rank', type=int, default=0, help='rank for each node')
    parser.add_argument('--GPU', type=int, default=0, help='which GPU to use')
    parser.add_argument('--ip', default='11.0.0.233', help='distributed master ip address')
    parser.add_argument('--port', default='1234', help='distributed master ip port')
    parser.add_argument('--throughput', action='store_true', help='Enable throughput measurement')
    parser.add_argument('--sparsity_check', action='store_true', help='Enable sparsity measurement')
    parser.add_argument('--eval_at_every_batch', action='store_true', help='Enable evaluation at every batch')
    parser.add_argument('--weak_scaling', action='store_true', help='Use weak scaling to do distributed training')
    parser.add_argument('--grace_config',
                        default='{"compressor": "none", "memory": "none", "communicator": "allreduce"}',
                        type=str, help='grace configuration parameters dictionary, default no compression')
    parser.add_argument('--log_time', action='store_true', help='log time breakdown for training')
    parser.add_argument('--log_volume', action='store_true', help='log data volume for compression')
    parser.add_argument('--extra_wandb_tags', default='', type=str,
                        help='tags for wandb run, separated by comma, ')

    return parser.parse_args()


def init_distributed(args):
    comm = MPI.COMM_WORLD
    args.world_size = comm.Get_size()
    args.local_rank = comm.Get_rank()

    # args.world_size = int(os.environ['WORLD_SIZE'])
    args.distributed = args.world_size > 1

    host_name = socket.gethostname()
    if host_name in ['mcnode37', 'mcnode38', 'mcnode39','mcnode40',]:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)  # default GPU0
    args.GPU = 0

    if args.distributed:
        # args.local_rank = int(os.environ['LOCAL_RANK'])

        '''
        Set cuda device so everything is done on the right GPU.
        THIS MUST BE DONE AS SOON AS POSSIBLE.
        '''
        torch.cuda.set_device(args.GPU)

        '''Initialize distributed communication'''
        if args.local_rank == 0:
            host_name = socket.gethostname()
            host_ip = socket.gethostbyname(host_name)
        else:
            host_ip = None
        host_ip = comm.bcast(host_ip, root=0)
        args.init_method = f'tcp://{host_ip}:{args.port}'

        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            timeout=datetime.timedelta(seconds=20),
            world_size=args.world_size,
            rank=args.local_rank,
        )


def init_wandb(args):

    suffix = ''
    for k, v in args.params.items():
        if k not in ['compressor', 'memory', 'communicator', 'world_size']:
            suffix += "-" + str(v)

    tags = "pytorch,RDMA,100Gb,"
    tags += f"{args.world_size}nodes,"
    tags = (tags + 'throughput,') if args.throughput else tags
    tags = (tags + 'deterministic,') if args.load_checkpoint_path else tags
    tags += args.extra_wandb_tags
    if args.local_rank == 0:
        os.environ['WANDB_PROJECT'] = "pytorch_ncf"
        os.environ['WANDB_ENTITY'] = "xxx"
        os.environ['WANDB_NAME'] = f"{args.params['compressor']}{suffix}"
        os.environ['WANDB_TAGS'] = tags
        os.environ['WANDB_API_KEY'] = "xxx"
        wandb_id = os.environ.get('WANDB_ID', None)
        if wandb_id is None:
            wandb.init(config=args, )
        else:
            wandb.init(config=args, id=f"{wandb_id}", )
        wandb.config.update(args.params, allow_val_change=True)
    else:
        os.environ['WANDB_MODE'] = 'dryrun'


def init_grace(args):
    # from grace_dl.dist.helper import grace_from_params
    from grace_dl.dist.compressor.deepreduce import deepreduce_from_params
    import json

    s = args.grace_config.replace("'", '"')
    params = json.loads(s)
    if args.rank in [0, -1]:
        print(params)

    params['world_size'] = args.world_size
    # grc = grace_from_params(params)
    grc = deepreduce_from_params(params)
    args.grc = grc


def val_epoch(model, x, y, dup_mask, real_indices, K, samples_per_user, num_user,
              epoch=None, distributed=False):
    model.eval()

    with torch.no_grad():
        p = []
        for u,n in zip(x,y):
            p.append(model(u, n, sigmoid=True).detach())

        temp = torch.cat(p).view(-1,samples_per_user)
        del x, y, p

        # set duplicate results for the same item to -1 before topk
        temp[dup_mask] = -1
        out = torch.topk(temp,K)[1]
        # topk in pytorch is stable(if not sort)
        # key(item):value(prediction) pairs are ordered as original key(item) order
        # so we need the first position of real item(stored in real_indices) to check if it is in topk
        ifzero = (out == real_indices.view(-1,1))
        hits = ifzero.sum()
        ndcg = (math.log(2) / (torch.nonzero(ifzero)[:,1].view(-1).to(torch.float)+2).log_()).sum()

    if distributed:
        torch.distributed.all_reduce(hits, op=torch.distributed.reduce_op.SUM)
        torch.distributed.all_reduce(ndcg, op=torch.distributed.reduce_op.SUM)

    hr = hits.item() / num_user
    ndcg = ndcg.item() / num_user

    model.train()
    return hr, ndcg


def main():
    from grace_dl.dist.helper import timer, volume, tensor_bits

    args = parse_args()
    init_distributed(args)
    if args.weak_scaling:
        args.batch_size *= args.world_size
    init_wandb(args)
    init_grace(args)

    if args.local_rank == 0:
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=args.log_path),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE)])
    else:
        dllogger.init(backends=[])

    dllogger.log(data=vars(args), step='PARAMETER')

    if not os.path.exists(args.checkpoint_dir) and args.checkpoint_dir:
        print("Saving results to {}".format(args.checkpoint_dir))
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # sync workers before timing
    if args.distributed:
        torch.distributed.broadcast(torch.tensor([1], device="cuda"), 0)
    torch.cuda.synchronize()

    main_start_time = time.time()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    train_ratings = torch.load(args.data+'/train_ratings.pt', map_location=torch.device('cuda:0'))
    test_ratings = torch.load(args.data+'/test_ratings.pt', map_location=torch.device('cuda:0'))
    test_negs = torch.load(args.data+'/test_negatives.pt', map_location=torch.device('cuda:0'))

    valid_negative = test_negs.shape[1]

    nb_maxs = torch.max(train_ratings, 0)[0]
    nb_users = nb_maxs[0].item() + 1
    nb_items = nb_maxs[1].item() + 1

    all_test_users = test_ratings.shape[0]

    test_users, test_items, dup_mask, real_indices = dataloading.create_test_data(test_ratings, test_negs, args)

    # make pytorch memory behavior more consistent later
    torch.cuda.empty_cache()

    # Create model
    model = NeuMF(nb_users, nb_items,
                  mf_dim=args.factors,
                  mlp_layer_sizes=args.layers,
                  dropout=args.dropout)

    optimizer = FusedAdam(model.parameters(), lr=args.learning_rate,
                          betas=(args.beta1, args.beta2), eps=args.eps)

    criterion = nn.BCEWithLogitsLoss(reduction='none') # use torch.mean() with dim later to avoid copy to host
    # Move model and loss to GPU
    model = model.cuda()
    criterion = criterion.cuda()

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2",
                                          keep_batchnorm_fp32=False, loss_scale='dynamic')

    # if args.distributed:
    #     model = DDP(model)

    local_batch = args.batch_size // args.world_size
    traced_criterion = torch.jit.trace(criterion.forward,
                                       (torch.rand(local_batch,1),torch.rand(local_batch,1)))

    if args.local_rank == 0:
        print(model)
        print("{} parameters".format(utils.count_parameters(model)))
        # [print(parameter) for parameter in model.parameters()]

    if args.load_checkpoint_path:
        state_dict = torch.load(args.load_checkpoint_path)
        state_dict = {k.replace('module.', '') : v for k,v in state_dict.items()}
        model.load_state_dict(state_dict)

    if args.mode == 'test':
        start = time.time()
        hr, ndcg = val_epoch(model, test_users, test_items, dup_mask, real_indices, args.topk,
                             samples_per_user=valid_negative + 1,
                             num_user=all_test_users, distributed=args.distributed)
        val_time = time.time() - start
        eval_size = all_test_users * (valid_negative + 1)
        eval_throughput = eval_size / val_time

        dllogger.log(step=tuple(), data={'best_eval_throughput' : eval_throughput,
                                         'hr@10' : hr})
        return
    
    max_hr = 0
    best_epoch = 0
    train_throughputs, eval_throughputs = [], []

    # broadcast model states from rank0 to other nodes !!! This is important!
    [torch.distributed.broadcast(p.data, src=0) for p in model.parameters()]
    # if args.local_rank == 0:
    #     save_initial_state_path = os.path.join(args.checkpoint_dir, 'model_init.pth')
    #     print("Saving the model to: ", save_initial_state_path)
    #     torch.save(model.state_dict(), save_initial_state_path)

    for epoch in range(args.epochs):

        begin = time.time()
        train_time = 0

        epoch_users, epoch_items, epoch_label = dataloading.prepare_epoch_train_data(train_ratings, nb_items, args)
        num_batches = len(epoch_users)
        for i in range(num_batches // args.grads_accumulated):
            batch_start = time.time()
            for j in range(args.grads_accumulated):
                batch_idx = (args.grads_accumulated * i) + j
                user = epoch_users[batch_idx]
                item = epoch_items[batch_idx]
                label = epoch_label[batch_idx].view(-1,1)

                outputs = model(user, item)
                loss = traced_criterion(outputs, label).float()
                loss = torch.mean(loss.view(-1), 0)

                if args.amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

            # check grad sparsity
            if args.sparsity_check:
                total_nonzero = 0
                total_numel = 0
                for index, (name, p) in enumerate(model.named_parameters()):
                    sparsity = 1.0 - torch.sum(p.grad.data.abs() > 0).float()/p.grad.data.numel()
                    total_nonzero += torch.sum(p.grad.data.abs() > 0).float()
                    total_numel += p.grad.data.numel()
                    if args.local_rank == 0:
                        wandb.log({f"{name}(sparsity)(numel={p.grad.data.numel()})": sparsity, }, commit=False)
                if args.local_rank == 0:
                    wandb.log({f"total_sparsity(numel={total_numel})": 1 - total_nonzero/total_numel, }, commit=True)

            # add grace just before optimizer.step()
            torch.cuda.synchronize()
            comm_start = time.time()
            for index, (name, p) in enumerate(model.named_parameters()):
                new_grad = args.grc.step(p.grad.data, name)
                p.grad.data = new_grad
            torch.cuda.synchronize()
            timer['comm'] = time.time() - comm_start

            # [torch.distributed.all_reduce(p.grad.data) for p in model.parameters()]
            # for param in model.parameters():
            #     dist.all_reduce(param.grad.data)
            #     param.grad.data /= float(args.world_size)

            optimizer.step()
            for p in model.parameters():
                p.grad = None
            if args.throughput:
                torch.cuda.synchronize()

            if args.log_time and args.local_rank == 0:
                timer['batch_time'] = time.time() - batch_start
                timer['computation'] = timer['batch_time'] - timer['comm']
                print("Timer:", timer, '\n')

                timer['en/decoding'] = 0
                timer['batch_time'] = 0
                timer['computation'] = 0
                timer['comm'] = 0

            if args.log_volume and args.local_rank == 0:
                ratio = volume['compress']/volume['nocompress']
                volume['ratio_acc'].append(ratio)
                avg_ratio = sum(volume['ratio_acc'])/len(volume['ratio_acc'])
                print(f"Data volume:: compress {volume['compress']} no_compress {volume['nocompress']} ratio {ratio:.4f} avg_ratio {avg_ratio:.4f}")
                volume['compress'] = 0
                volume['nocompress'] = 0

            batch_throughput = args.batch_size / (time.time() - batch_start) # global throughput
            train_time += time.time() - batch_start
            if (args.throughput or args.eval_at_every_batch) and args.local_rank == 0:
                print(f"Train :: Epoch [{epoch}/{args.epochs}] \t Batch [{i}/{num_batches}] \t "
                      f"Time {time.time()-batch_start:.5f} \t Throughput {batch_throughput:.2f}")

            if args.throughput and i == 3:
                break
            if args.local_rank == 0:
                print(f"Train :: Epoch [{epoch}/{args.epochs}] \t Batch [{i}/{num_batches}] \t "
                      f"Time {time.time()-batch_start:.5f} \t Throughput {batch_throughput:.2f}")
            if args.eval_at_every_batch:
                hr, ndcg = val_epoch(model, test_users, test_items, dup_mask, real_indices, args.topk,
                                     samples_per_user=valid_negative + 1,
                                     num_user=all_test_users, epoch=epoch, distributed=args.distributed)
                if args.local_rank == 0:
                    wandb.log({"eval/hr@10": hr,})

        del epoch_users, epoch_items, epoch_label
        # train_time = time.time() - begin
        begin = time.time()

        epoch_samples = len(train_ratings) * (args.negative_samples + 1)
        train_throughput = epoch_samples / train_time
        if args.throughput:
            train_throughput = batch_throughput
        train_throughputs.append(train_throughput)

        hr, ndcg = val_epoch(model, test_users, test_items, dup_mask, real_indices, args.topk,
                             samples_per_user=valid_negative + 1,
                             num_user=all_test_users, epoch=epoch, distributed=args.distributed)

        val_time = time.time() - begin

        eval_size = all_test_users * (valid_negative + 1)
        eval_throughput = eval_size / val_time
        eval_throughputs.append(eval_throughput)

        dllogger.log(step=(epoch,),
                     data = {'train_throughput': train_throughput,
                             'hr@10': hr,
                             'train_epoch_time': train_time,
                             'validation_epoch_time': val_time,
                             'eval_throughput': eval_throughput})

        if args.local_rank == 0:
            wandb.log({"train_epoch_time": train_time,
                       'validation_epoch_time': val_time,
                       'eval_throughput': eval_throughput,
                       'train_throughput': train_throughput,
                       }, commit=False)
            if not args.eval_at_every_batch:
                wandb.log({"eval/hr@10": hr, }, commit=False)
            wandb.log({"epoch": epoch})

        if hr > max_hr and args.local_rank == 0:
            max_hr = hr
            best_epoch = epoch
            print("New best hr!")
            if args.checkpoint_dir:
                save_checkpoint_path = os.path.join(args.checkpoint_dir, 'model.pth')
                print("Saving the model to: ", save_checkpoint_path)
                torch.save(model.state_dict(), save_checkpoint_path)
            best_model_timestamp = time.time()

        if args.threshold is not None:
            if hr >= args.threshold:
                print("Hit threshold of {}".format(args.threshold))
                break

        if args.throughput:
            break

    if args.local_rank == 0:
        dllogger.log(data={'best_train_throughput': max(train_throughputs),
                           'best_eval_throughput': max(eval_throughputs),
                           'mean_train_throughput': np.mean(train_throughputs),
                           'mean_eval_throughput': np.mean(eval_throughputs),
                           'best_accuracy': max_hr,
                           'best_epoch': best_epoch,
                           'time_to_target': time.time() - main_start_time,
                           'time_to_best_model': best_model_timestamp - main_start_time},
                     step=tuple())

        wandb.log({
                   'best_train_throughput': max(train_throughputs),
                   'best_eval_throughput': max(eval_throughputs),
                   'mean_train_throughput': np.mean(train_throughputs),
                   'mean_eval_throughput': np.mean(eval_throughputs),
                   'best_accuracy': max_hr,
                   'best_epoch': best_epoch,
                   'time_to_target': time.time() - main_start_time,
                   'time_to_best_model': best_model_timestamp - main_start_time
                   })


if __name__ == '__main__':
    main()
