from xtal2dos.data import *
from xtal2dos.xtal2dos import *
from xtal2dos.file_setter import use_property
from xtal2dos.utils import *
from xtal2dos.transformer import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import numpy as np
#import gc
import pickle
from copy import copy, deepcopy
import json
import time

from torch.utils.tensorboard import SummaryWriter
from parser import get_parser
import matplotlib.pyplot as plt

import socket
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn


best_valid_loss=1e+10

def train_model(rank, args):
    if args.sche == "step":
        sche_name = f"_step-{args.step_interval}"
    elif args.sche == "cosine":
        sche_name = f"_cosine-{args.T0}-{args.eta_min}-{args.T_mult}"
    elif args.sche == "lambda":
        sche_name = f"_lambda-{args.d_model}-{args.warmup}-{args.lambda_factor}-{args.lambda_scale}"
    elif args.sche == "const":
        sche_name = "_const"

    if args.swa:
        args.note = f"swa-{args.swa_start}-{args.swa_lr}_" + args.note

    ckpt_dir = 'model_' + args.data_src + '_' + args.label_scaling + '_' \
                 + args.xtal2dos_loss_type +f'_dropout-{args.graph_dropout}-{args.dec_dropout}' \
                 + f'_bs-{args.batch_size*args.gpus}' + f'_lr-{args.lr}' + f'_{args.opt}' + f'_gpu-{args.gpus}' + sche_name \
                 + f'_ep-{args.num_epochs}' + f'_dec_{args.dec_layers}l' \
                 + f'_temp-{args.temp}' + f'_wd-{args.weight_decay}' + f'_rd-{args.rate_decay}' \
                 + f'_weighted-{args.sum_weighted}-{args.sum_scale}' + f'_accum-{args.accum_step}' + f'_h-{args.h}' + f'_d-{args.d_model}' \
                 + f'_clip-{args.clip}' + f'_c-epochs-{args.c_epochs}' \
                 + f'_{args.note}'

    log_dir = './TRAINED/' + ckpt_dir
    if rank == 0:
        mkdirs(log_dir)
        log_file = open(log_dir + '/log.txt', 'w')

    def print_log(msg):
        if rank == 0:
            print(msg)
            print(msg, file=log_file)

    ###########
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    print_log(f"Rank {rank + 1}/{args.world_size} process initialized.\n")
    ###########

    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True

    torch.autograd.set_detect_anomaly(True)
    #device = set_device()
    #print_log("DEVICE:", device)
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True
    device = torch.device(f'cuda:{rank}')

    # GNN --- parameters
    data_src = args.data_src
    RSM = {'radius': 8, 'step': 0.2, 'max_num_nbr': 12}

    number_layers                        = args.num_layers
    number_neurons                       = args.num_neurons
    n_heads                              = args.num_heads
    concat_comp                          = args.concat_comp

    # DATA PARAMETERS
    random_num      =  1; random.seed(random_num)
    np.random.seed(random_num)
    torch.manual_seed(random_num)
    # MODEL HYPER-PARAMETERS
    num_epochs      = args.num_epochs
    learning_rate   = args.lr
    batch_size      = args.batch_size

    stop_patience   = 150
    best_epoch      = 1
    adj_epochs      = 50
    milestones      = [150,250]
    #train_param     = {'batch_size':batch_size, 'shuffle': False} ###########
    train_param     = {'batch_size':batch_size, 'shuffle': False}
    valid_param     = {'batch_size':batch_size, 'shuffle': False}

    # DATALOADER/ TARGET NORMALIZATION
    if args.data_src == 'binned_dos_128':
        pd_data = pd.read_csv(f'../xtal2dos_DATA/label_edos/mpids.csv')
        np_data = np.load(f'../xtal2dos_DATA/label_edos/total_dos_128.npy')
    elif args.data_src == 'ph_dos_51':
        pd_data = pd.read_csv(f'../xtal2dos_DATA/phdos/mpids.csv')
        np_data = np.load(f'../xtal2dos_DATA/phdos/ph_dos.npy')
    else:
        raise ValueError('')

    NORMALIZER = DATA_normalizer(np_data)

    CRYSTAL_DATA = CIF_Dataset(args, pd_data=pd_data, np_data=np_data, root_dir=f'../xtal2dos_DATA/', **RSM)

    if args.data_src == 'ph_dos_51':
        with open('../xtal2dos_DATA/phdos/200801_trteva_indices.pkl', 'rb') as f:
            train_idx, val_idx, test_idx = pickle.load(f)
    else:
        idx_list = list(range(len(pd_data)))
        random.shuffle(idx_list)
        train_idx_all, test_val = train_test_split(idx_list, train_size=args.train_size, random_state=random_num)
        test_idx, val_idx = train_test_split(test_val, test_size=0.5, random_state=random_num)

    if args.trainset_subset_ratio < 1.0:
        train_idx, _ = train_test_split(train_idx_all, train_size=args.trainset_subset_ratio, random_state=random_num)
    elif args.data_src != 'ph_dos_51':
        train_idx = train_idx_all

    if args.finetune:
        assert args.finetune_dataset != 'null'
        if args.data_src == 'binned_dos_128':
            with open(f'../xtal2dos_DATA/label_edos/materials_classes/' + args.finetune_dataset + '/train_idx.json', ) as f:
                train_idx = json.load(f)

            with open(f'../xtal2dos_DATA/label_edos/materials_classes/' + args.finetune_dataset + '/val_idx.json', ) as f:
                val_idx = json.load(f)

            with open(f'../xtal2dos_DATA/label_edos/materials_classes/' + args.finetune_dataset + '/test_idx.json', ) as f:
                test_idx = json.load(f)
        else:
            raise ValueError('Finetuning is only supported on the binned dos 128 dataset.')

    #print_log('total size:', len(idx_list))
    print_log(f'training size: {len(train_idx)}, min/max: {min(train_idx)} {max(train_idx)}')
    print_log(f'validation size: {len(val_idx)}, min/max: {min(val_idx)} {max(val_idx)}')
    print_log(f'testing size: {len(test_idx)}, min/max:, {min(test_idx)}, {max(test_idx)}')
    print_log(f'total size: {len(train_idx)}, {len(val_idx)+len(test_idx)}')


    training_set       =  CIF_Lister(train_idx,CRYSTAL_DATA,df=pd_data)
    validation_set     =  CIF_Lister(val_idx,CRYSTAL_DATA,df=pd_data)
    testing_set        =  CIF_Lister(test_idx, CRYSTAL_DATA, df=pd_data)

    print_log(f'> USING MODEL xtal2dos!')
    net = Xtal2DoS(args)
    #for p in net.parameters():
    #    if p.dim() > 1:
    #        nn.init.xavier_uniform_(p)
    net.cuda(rank)

    swa_net = None
    if args.swa:
        swa_net = AveragedModel(net).cuda(rank)

    ################
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
    net = nn.parallel.DistributedDataParallel(net, 
                                              device_ids=[rank],
                                              find_unused_parameters=True
                                             )
    ################


    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_log(f"model_size: {count_params(net)}")


    if args.finetune:
        # load checkpoint
        check_point = torch.load(args.check_point_path)
        net.load_state_dict(check_point['model'])
        learning_rate = learning_rate/5

    # LOSS & OPTMIZER & SCHEDULER
    #optimizer = optim.AdamW(net.parameters(), lr = learning_rate, weight_decay = 1e-2)
    #optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum=0.9)
    if args.opt == "adam":
        #optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(net.parameters(), lr = learning_rate, eps=1e-9, weight_decay = args.weight_decay)


    if args.sche == "step":
        decay_times = 6
        decay_ratios = 0.5
        one_epoch_iter = np.ceil(len(train_idx) / batch_size)
        if args.finetune:
            decay_ratios = 0.5
        scheduler = lr_scheduler.StepLR(optimizer, args.step_interval, decay_ratios)
    elif args.sche == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=args.eta_min, T_0=args.T0, T_mult=args.T_mult)
    elif args.sche == "lambda":
        shift = args.warmup - args.warmup**(1./args.lambda_scale)
        one_epoch_steps = len(train_idx) // (batch_size * args.gpus) + 1

        print_log(f"## shift: {shift}")
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(
                step, args.d_model, factor=args.lambda_factor, 
                warmup=args.warmup, scale=args.lambda_scale, 
                shift = shift, c_steps = one_epoch_steps * args.c_epochs,
                rate_decay = args.rate_decay
            ),
        )
        '''
        total_steps = (len(train_idx) // (batch_size * args.gpus) + 1) * num_epochs
        print_log(f"## total_steps: {total_steps}")
        print_log(f"## n_train: {len(train_idx)}, bs: {batch_size}, gpus: {args.gpus}, num_epochs: {num_epochs}")
        scheduler = get_cosine_schedule_with_warmup(
                      optimizer = optimizer,
                      num_warmup_steps = int(0.1 * total_steps),
                      num_training_steps = total_steps,
                      eta_min = args.eta_min
                    )
        '''
    
    if args.swa:
        swa_start = args.swa_start
        swa_scheduler = SWALR(optimizer, swa_lr = args.swa_lr)


    loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True).cuda()

    ######################
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        training_set,
        num_replicas = args.world_size,
        rank=rank
    )
    ######################

    print_log(f'> TRAINING MODEL ...')
    train_loader   = torch_DataLoader(dataset = training_set,
                                      batch_size = batch_size,
                                      shuffle = False,
                                      num_workers = 0,
                                      pin_memory=True,
                                      sampler = train_sampler
                                     )
    valid_loader   = torch_DataLoader(dataset=validation_set,
                                      batch_size = batch_size,
                                      shuffle = False,
                                      pin_memory=True
                                     )
    test_loader    = torch_DataLoader(dataset=testing_set,
                                      batch_size = batch_size,
                                      shuffle = False,
                                      pin_memory=True
                                     )

    total_loss = 0
    training_counter=0
    cur_step = 0
    valid_counter=0
    current_step = 0

    prediction = []
    label_gt = []


    summary_dir = './summary/' + ckpt_dir
    writer = None
    if rank == 0:
        mkdirs(summary_dir)
        writer = SummaryWriter(log_dir=summary_dir)

    '''if rank == 0:
        path = 'train_plots/' + ckpt_dir
        mkdirs(path)
        for i, data in enumerate(train_loader):
            #print_log(f"batch-{i}: {torch.min(data.y)} {torch.max(data.y)}")
            ys = data.y
            y_bases = data.y_base
            l = ys.shape[1]
            x = np.arange(l)
            for idx, (y, y_base) in enumerate(zip(ys, y_bases)):
                plt.plot(x, y, color='red')
                plt.plot(x, y_base, color='blue', alpha=0.7)
                plt.savefig(path+f'/{idx}.png')
                plt.clf()
            exit()
    '''
    
    if rank == 0:
        log_file.flush()

    start_time = time.time()
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)

        '''if args.num_epochs == 200 and args.sche == 'step' and 0 < epoch < 199:
            scheduler.step()
        elif num_epochs == 600 and args.sche == 'cosine':
            if 0 < epoch < 299:
                scheduler.step()
            elif epoch == 299:
                for g in optimizer.param_groups:
                    g['lr'] /= 2.
        else:
            scheduler.step()'''

        '''
        if args.num_epochs == 200 and args.sche == 'step':
            if 0 < epoch < 199:
                scheduler.step()
        else:
            scheduler.step()'''

        #if epoch % 18 == 0:
        #    for g in optimizer.param_groups:
        #        g['lr'] /= 2.

        if epoch > 0:
            if args.swa and epoch > swa_start:
                swa_net.update_parameters(net)
                swa_scheduler.step()
                if epoch % args.c_epochs == 0:
                    optimizer.param_groups[0]['lr'] /= 2.
                    optimizer.param_groups[0]['swa_lr'] /= 2.
            else:
                if args.sche == "cosine":
                    if epoch < args.T0 - 1:
                        scheduler.step()
                elif args.sche == "step":
                    scheduler.step()
    
        if 0 < epoch < args.T0-1 and args.sche == "cosine":
            if epoch % args.c_epochs == 0:
                scheduler.base_lrs[0] /= args.rate_decay


        #if epoch == 20:
        #    for g in optimizer.param_groups:
        #        g['lr'] /= 2.

        #for i, data in enumerate(train_loader):
        #    train_label = data.y.to(device)
        #    batch_sum = torch.sum(train_label, dim=1, keepdim=False)
        #    print(torch.mean(batch_sum), torch.median(batch_sum))
        #    if i > 20:
        #        exit()

        # TRAINING-STAGE
        net.train()
        args.train = True
        n_accum = 0

        for data in tqdm(train_loader, mininterval=0.5, desc=f'(EPOCH:{epoch} TRAINING)', position=0, leave=True, ascii=True):
            n_accum += 1
            current_step += 1
            data = data.to(device, non_blocking=True)
            if isinstance(data.y, tuple) or isinstance(data.y, list):
                data.y = data.y[0]
            train_label = data.y.to(device)
            
            if args.label_scaling == 'normalized_sum':
                train_label_normalized = train_label / (torch.sum(train_label, dim=1, keepdim=True) + 1e-10)
                train_label_sum = torch.sum(train_label, dim=1, keepdim=False)
            else:
                raise ValueError('wrong label_scaling')

            pred_logits = net(data)
            loss, pred = compute_loss(train_label_normalized, train_label_sum, pred_logits, loss_fn, args)
            

            loss /= args.accum_step
            loss.backward()
            
            if n_accum % args.accum_step == 0:
                nn.utils.clip_grad_norm_(net.parameters(), args.clip)

            if writer is not None and rank == 0:
                tot_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in net.module.parameters() if p.grad is not None]), 2)
                writer.add_scalar("TotNorm/tot_norm", tot_norm, cur_step)
                for n, p in net.module.named_parameters():
                    if p.grad is not None:
                        #print_log(f"{n}: {torch.max(p.grad):.8f}, {torch.min(p.grad):.8f}, {torch.median(p.grad):.8f}")
                        writer.add_scalar(f"GradNorm/{n}", p.grad.data.norm(2), cur_step)
            
            if n_accum % args.accum_step == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if args.sche == "lambda" and epoch < args.T0-1:
                scheduler.step()
 

            prediction.append(pred.detach().cpu().numpy())
            label_gt.append(train_label.detach().cpu().numpy())

            total_loss += loss
            training_counter +=1

            mae, r2, mse, wd, mae_ori, r2_ori, mse_ori, wd_ori = \
                metrics(train_label.detach().cpu().numpy()+1e-10, pred.detach().cpu().numpy(), args, mode="normalized_sum")

            if writer is not None and rank == 0:
                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("Train/lr", lr, cur_step)
                writer.add_scalar("Train/loss", loss, cur_step)
                writer.add_scalar("Train/r2", r2, cur_step)
                writer.add_scalar("Train/r2_ori", r2_ori, cur_step)
                writer.add_scalar("Train/mae", mae, cur_step)
                writer.add_scalar("Train/mae_ori", mae_ori, cur_step)
                writer.add_scalar("Train/mse", mse, cur_step)
                writer.add_scalar("Train/mse_ori", mse_ori, cur_step)
                writer.add_scalar("Train/wd", wd, cur_step)
                writer.add_scalar("Train/wd_ori", wd_ori, cur_step)

                #for tag, value in net.module.named_parameters():
                #    if value.grad is not None:
                #        writer.add_histogram(tag + "/grad", value.grad.cpu(), cur_step)
               
                cur_step += 1
 
        
        optimizer.zero_grad(set_to_none=True)
        avg_loss = total_loss / training_counter

        prediction = np.concatenate(prediction, axis=0)
        label_gt = np.concatenate(label_gt, axis=0)

        mae, r2, mse, wd, mae_ori, r2_ori, mse_ori, wd_ori = metrics(label_gt, prediction, args, mode="normalized_sum")

        if args.swa and epoch > swa_start:
            update_bn(train_loader, swa_net, device=device)

        if rank == 0:
            print_log(f"\n******* {epoch} TRAINING STATISTIC *****")
            print_log("lr = %.6f" % lr)
            print_log("avg_loss =%.6f\t" % avg_loss)
            print_log("r2=%.6f\t  r2_ori=%.6f" % (r2, r2_ori))
            print_log("mae=%.6f\t  mae_ori=%.6f" % (mae, mae_ori))
            print_log("mse=%.6f\t  mse_ori=%.6f" % (mse, mse_ori))
            print_log("wd=%.6f\t wd_ori=%.6f" % (wd, wd_ori))
            print_log("\n*****************************************")

        training_counter = 0
        total_loss = 0
        total_loss_base = 0
        total_loss_peak = 0
        prediction = []
        label_gt = []

        def valid_test(data_loader, net, swa_net, mode="Valid"):
            global best_valid_loss

            # VALIDATION-PHASE
            valid_counter = 0
            total_loss = 0
            prediction = []
            label_gt = []

            net.eval()

            for data in tqdm(data_loader, mininterval=0.5, desc=f'({mode})', position=0, leave=True, ascii=True):
                data = data.to(device, non_blocking=True)
                if isinstance(data.y, tuple) or isinstance(data.y, list):
                    data.y = data.y[0]
                valid_label = data.y.float().to(device)

                if args.label_scaling == 'normalized_sum':
                    valid_label_normalized = valid_label / (torch.sum(valid_label, dim=1, keepdim=True) + 1e-10)
                    valid_label_sum = torch.sum(valid_label, dim=1, keepdim=False)
                else:
                    raise ValueError('wrong label_scaling')
                
                with torch.no_grad():
                    #pred_base_logits, pred_peak_logits = net(data)
                    if args.swa and epoch > swa_start:
                        pred_logits = swa_net(data)
                    else:
                        pred_logits = net(data)
                    #loss_base, pred_base = compute_loss(valid_label_base_norm, pred_base_logits, loss_fn, args)
                    #loss_peak, pred_peak = compute_loss(valid_label_peak_norm, pred_peak_logits, loss_fn, args)
                    #loss = loss_base + loss_peak
                    loss, pred = compute_loss(valid_label_normalized, valid_label_sum, pred_logits, loss_fn, args)

                #pred = pred_base * torch.sum(valid_label_base, dim=1, keepdim=True) \
                #     + pred_peak * torch.sum(valid_label_peak, dim=1, keepdim=True)
                prediction.append(pred.detach().cpu().numpy())
                label_gt.append(valid_label.detach().cpu().numpy())

                total_loss += loss
                valid_counter += 1

            avg_loss = total_loss / valid_counter

            prediction = np.concatenate(prediction, axis=0)
            label_gt = np.concatenate(label_gt, axis=0)

            mae, r2, mse, wd, mae_ori, r2_ori, mse_ori, wd_ori = metrics(label_gt, prediction, args, mode="normalized_sum")
            
            if writer is not None and rank == 0:
                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar(f"{mode}/lr", lr, epoch)
                writer.add_scalar(f"{mode}/loss", avg_loss, epoch)
                writer.add_scalar(f"{mode}/r2", r2, epoch)
                writer.add_scalar(f"{mode}/r2_ori", r2_ori, epoch)
                writer.add_scalar(f"{mode}/mae", mae, epoch)
                writer.add_scalar(f"{mode}/mae_ori", mae_ori, epoch)
                writer.add_scalar(f"{mode}/mse", mse, epoch)
                writer.add_scalar(f"{mode}/mse_ori", mse_ori, epoch)
                writer.add_scalar(f"{mode}/wd", wd, epoch)
                writer.add_scalar(f"{mode}/wd_ori", wd_ori, epoch)

            if rank == 0:
                print_log(f"\n********** {epoch} {mode} STATISTIC ***********")
                print_log("lr = %.6f\t" % lr)
                print_log("avg_loss =%.6f\t" % avg_loss)
                print_log("r2=%.6f\t  r2_ori=%.6f" % (r2, r2_ori))
                print_log("mae=%.6f\t  mae_ori=%.6f" % (mae, mae_ori))
                print_log("mse=%.6f\t  mse_ori=%.6f" % (mse, mse_ori))
                print_log("wd=%.6f\t wd_ori=%.6f" % (wd, wd_ori))
                print_log("\n*****************************************")

            if mode == "Test" and best_valid_loss > mse_ori and rank == 0:
                best_valid_loss = mse_ori
                print_log("\n********** SAVING MODEL ***********")
                if args.swa and epoch > swa_start:
                    checkpoint = {'model': swa_net.module.state_dict(), 'args': args}
                else:
                    checkpoint = {'model': net.module.state_dict(), 'args': args}
                if not args.finetune:
                    #checkpoint_path = './TRAINED/'
                    save_path = './TRAINED/' + ckpt_dir
                else:
                    save_path = './TRAINED/finetune/' + ckpt_dir

                if args.ablation_LE:
                    save_path = save_path + '_ablation_LE'

                if args.ablation_CL:
                    save_path = save_path + '_ablation_CL'
                mkdirs(save_path)

                save_path = save_path + '/model.ckpt'
                torch.save(checkpoint, save_path)
                print_log("A new model has been saved to " + save_path)
                print_log("\n*****************************************")

            valid_counter = 0
            total_loss = 0
            prediction = []
            label_gt = []
        
        valid_test(valid_loader, net, swa_net, mode="Valid")
        valid_test(test_loader, net, swa_net, mode="Test")

        if rank == 0:
            log_file.flush()

    end_time = time.time()
    e_time = end_time - start_time
    print_log('Best validation loss=%.6f, training time (min)=%.6f'%(best_valid_loss, e_time/60))
    print_log(f"> DONE TRAINING !")

    if rank == 0:
        log_file.close()

def get_unique_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def main():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])

    args.gpus = torch.cuda.device_count()
    print(f"# of gpus: {args.gpus}")
    args.world_size = args.gpus

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(get_unique_port())

    mp.spawn(train_model, nprocs=args.gpus, args=(args,))

if __name__ == '__main__':
    main()

