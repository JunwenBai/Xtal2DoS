import argparse

def get_parser():
    # MOST CRUCIAL DATA PARAMETERS
    parser = argparse.ArgumentParser(description='Xtal2DoS')
    parser.add_argument('--data_src', default='binned_dos_128',choices=['binned_dos_128','binned_dos_32','ph_dos_51'])
    parser.add_argument('--sche', default='cosine',choices=['step','cosine','lambda', 'const'])
    parser.add_argument('--opt', default='adamw',choices=['adam','adamw'])
    parser.add_argument('--part', default='both',choices=['base','peak','both'])

    parser.add_argument('--model', default='gat',choices=['gat','e3nn'])
    parser.add_argument('--exp_num', default=0, type=int)

    parser.add_argument('--label_scaling', default='standardized',choices=['standardized','normalized_sum', 'normalized_max', 'logcosh', 'huber', 'KL', 'bregman'])
    # MOST CRUCIAL MODEL PARAMETERS
    parser.add_argument('--num_layers',default=3, type=int,
                        help='number of AGAT layers to use in model (default:3)')
    parser.add_argument('--num_neurons',default=128, type=int,
                        help='number of neurons to use per AGAT Layer(default:128)')
    parser.add_argument('--num_heads',default=4, type=int,
                        help='number of Attention-Heads to use  per AGAT Layer (default:4)')
    parser.add_argument('--concat_comp',default=False, type=bool,
                        help='option to re-use vector of elemental composition after global summation of crystal feature.(default: False)')
    parser.add_argument('--train_size',default=0.8, type=float, help='ratio size of the training-set (default:0.8)')
    parser.add_argument('--trainset_subset_ratio',default=0.5, type=float, help='ratio size of the training-set subset (default:0.5)')
    parser.add_argument('--use_catached_data', default=True, type=bool)
    parser.add_argument('--use_bin', action="store_true")
    parser.add_argument('--sum_weighted', action="store_true")

    parser.add_argument("--train",action="store_true")  # default value is false
    parser.add_argument('--num-epochs',default=200, type=int)
    parser.add_argument('--c_epochs',default=20, type=int)
    parser.add_argument('--batch-size',default=256, type=int)
    parser.add_argument('--dec_layers',default=2, type=int)
    parser.add_argument('--dec_dropout',default=0.1, type=float)
    parser.add_argument('--temp',default=2., type=float)
    parser.add_argument('--clip',default=1., type=float)

    parser.add_argument('--dec_in_dim', default=256, type=int)
    parser.add_argument('--chunk_dim', default=3, type=int)
    parser.add_argument('--accum_step', default=1, type=int)
    parser.add_argument('--step_interval', default=25, type=int)
    parser.add_argument('--lambda_factor', default=0.5, type=float)
    parser.add_argument('--lambda_scale', default=1.5, type=float)

    parser.add_argument('--d_model',default=512, type=int)
    parser.add_argument('--h',default=4, type=int)
    parser.add_argument('--d_ff',default=2048, type=int)
    parser.add_argument('--warmup',default=3000, type=int)
    parser.add_argument('--sum_scale',default=1000., type=float)
    parser.add_argument('--weight_decay',default=0., type=float)
    parser.add_argument('--rate_decay',default=2., type=float)

    parser.add_argument('--lr',default=0.001, type=float)
    parser.add_argument('--xtal2dos-input-dim',default=128, type=int)
    parser.add_argument('--xtal2dos-label-dim',default=128, type=int)
    parser.add_argument('--xtal2dos-latent-dim',default=128, type=int)
    parser.add_argument('--xtal2dos-emb-size',default=512, type=int)
    parser.add_argument('--graph_dropout',default=0.1, type=float)
    parser.add_argument('--xtal2dos-scale-coeff',default=1.0, type=float)
    parser.add_argument('--xtal2dos-loss-type',default='MAE', type=str, choices=['MAE', 'KL', 'WD', 'MSE'])
    parser.add_argument('--xtal2dos-K',default=10, type=int)
    parser.add_argument("--finetune",action="store_true")  # default value is false
    parser.add_argument("--ablation-LE",action="store_true")  # default value is false
    parser.add_argument("--ablation-CL",action="store_true")  # default value is false
    parser.add_argument("--finetune-dataset",default='null',type=str)
    parser.add_argument("--note",default='',type=str)
    parser.add_argument('--check-point-path', default=None, type=str)

    parser.add_argument('-T0', "--T0", default=100, type=int, help='optimizer T0')
    parser.add_argument('-T_mult', "--T_mult", default=2, type=int, help='T_mult')
    parser.add_argument('-eta_min', "--eta_min", default=2e-4, type=float, help='eta min')
    
    parser.add_argument('-swa_start', "--swa_start", default=100, type=int, help='swa start')
    parser.add_argument('-swa', "--swa", action='store_true', help='whether to use swa')
    parser.add_argument('-swa_lr', "--swa_lr", default=2e-4, type=float, help='swa_lr')

    return parser

