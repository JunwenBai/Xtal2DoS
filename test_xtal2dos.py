from xtal2dos.data import *
from xtal2dos.xtal2dos import *
from xtal2dos.file_setter import use_property
from xtal2dos.utils import *
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import gc
import pickle
from copy import copy, deepcopy
from os import makedirs

from parser import get_parser
from scipy.interpolate import interp1d

torch.autograd.set_detect_anomaly(True)
device = torch.device(f'cuda:0')

# MOST CRUCIAL DATA PARAMETERS
parser = get_parser()
parser.add_argument('--test-mpid', default='mpids.csv', type=str)
args = parser.parse_args(sys.argv[1:])

# GNN --- parameters
data_src = args.data_src
RSM = {'radius': 8, 'step': 0.2, 'max_num_nbr': 12}

number_layers                        = args.num_layers
number_neurons                       = args.num_neurons
n_heads                              = args.num_heads
concat_comp                          = args.concat_comp

# SETTING UP CODE TO RUN ON GPU
#gpu_id = 0
#device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

# DATA PARAMETERS
random_num          =  1; random.seed(random_num)
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
train_param     = {'batch_size':batch_size, 'shuffle': True}
valid_param     = {'batch_size':batch_size, 'shuffle': False}

# DATALOADER/ TARGET NORMALIZATION
if args.data_src == 'binned_dos_128':
    pd_data = pd.read_csv(f'../xtal2dos_DATA/label_edos/'+args.test_mpid)
    np_data = np.load(f'../xtal2dos_DATA/label_edos/total_dos_128.npy')
elif args.data_src == 'ph_dos_51':
    pd_data = pd.read_csv(f'../xtal2dos_DATA/phdos/'+args.test_mpid)
    np_data = np.load(f'../xtal2dos_DATA/phdos/ph_dos.npy')
elif args.data_src == 'no_label_128':
    pd_data = pd.read_csv(f'../xtal2dos_DATA/no_label/'+args.test_mpid)
    np_data = np.random.rand(len(pd_data), 128) # dummy label

NORMALIZER = DATA_normalizer(np_data)

if args.data_src == 'no_label_128':
    mean_tmp = torch.tensor(np.load(f'../xtal2dos_DATA/no_label/label_mean_binned_dos_128.npy'))
    std_tmp = torch.tensor(np.load(f'../xtal2dos_DATA/no_label/label_std_binned_dos_128.npy'))
    NORMALIZER.mean = mean_tmp
    NORMALIZER.std = std_tmp

CRYSTAL_DATA = CIF_Dataset(args, pd_data=pd_data, np_data=np_data, root_dir=f'../xtal2dos_DATA/', **RSM)

ckpt_dir = 'model_' + args.data_src + '_' + args.label_scaling + '_' \
             + args.xtal2dos_loss_type + '_trainsize' + str(args.trainset_subset_ratio) + '_dropout' + str(args.dec_dropout)
print("model:", ckpt_dir)


if args.data_src == 'ph_dos_51':
    with open('../xtal2dos_DATA/phdos/200801_trteva_indices.pkl', 'rb') as f:
        train_idx, val_idx, test_idx = pickle.load(f)
elif args.data_src == 'no_label_128':
    test_idx = list(range(len(pd_data)))
else:
    idx_list = list(range(len(pd_data)))
    random.shuffle(idx_list)
    train_idx_all, test_val = train_test_split(idx_list, train_size=args.train_size, random_state=random_num)
    test_idx, val_idx = train_test_split(test_val, test_size=0.5, random_state=random_num)

if args.trainset_subset_ratio < 1.0:
    train_idx, _ = train_test_split(train_idx_all, train_size=args.trainset_subset_ratio, random_state=random_num)
elif args.data_src != 'ph_dos_51' and args.data_src != 'no_label_128':
    train_idx = train_idx_all

if args.finetune:
    assert args.finetune_dataset != 'null'
    if args.data_src == 'binned_dos_128':
        with open(f'../xtal2dos_DATA/20210619_binned_32_128/materials_classes/' + args.finetune_dataset + '/test_idx.json', ) as f:
            test_idx = json.load(f)
    else:
        raise ValueError('Finetuning is only supported on the binned dos 128 dataset.')

print('testing size:', len(test_idx))

testing_set     =  CIF_Lister(test_idx, CRYSTAL_DATA, df=pd_data)

print(f'> USING MODEL xtal2dos!')
the_network = Xtal2DoS(args)
net = the_network.to(device)
# load checkpoint
if args.finetune:
    check_point_path = './TRAINED/finetune/model_xtal2dos_' + args.data_src + '_' + args.label_scaling \
            + '_' + args.xtal2dos_loss_type + '_finetune_' + args.finetune_dataset + '.chkpt'
else:
    check_point_path = './TRAINED/' + ckpt_dir + '.chkpt'

if args.ablation_LE:
    check_point_path = './TRAINED/model_xtal2dos_binned_dos_128_normalized_sum_KL_trainsize1.0_ablation_LE.chkpt'

if args.ablation_CL:
    check_point_path = './TRAINED/model_xtal2dos_binned_dos_128_normalized_sum_KL_trainsize1.0_ablation_CL.chkpt'

if args.check_point_path is not None:
    check_point = torch.load('./TRAINED/'+args.check_point_path+'/model.ckpt')
else:
    check_point = torch.load(check_point_path)
net.load_state_dict(check_point['model'])

print(f'> TESTING MODEL ...')
test_loader   = torch_DataLoader(dataset=testing_set, **valid_param)

loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True).cuda()

def test():
    training_counter=0
    training_loss=0
    valid_counter=0
    valid_loss=0
    best_valid_loss=1e+10
    check_fre = 10
    current_step = 0
    checkpoint_path = './TRAINED/'

    total_loss = 0
    nll_loss_smooth = 0
    nll_loss_x_smooth = 0
    kl_loss_smooth = 0
    cpc_loss_smooth = 0
    prediction = []
    prediction_x = []
    label_gt = []
    label_norm = []
    cif_ids = []
    losses = []

    start_time = time.time()

    # TESTING-PHASE
    net.eval()
    args.train = True
    for data in tqdm(test_loader, mininterval=0.5, desc='(testing)', position=0, leave=True, ascii=True):
        data = data.to(device)
        if isinstance(data.y, tuple) or isinstance(data.y, list):
            data.y = data.y[0]
        
        valid_label = deepcopy(data.y).float().to(device)
        #valid_label_base = data.y_base.float().to(device).clamp(min=0.)
        #valid_label_peak = (valid_label - valid_label_base).clamp(min=0.)

        #if 'part-base' in args.check_point_path:
        #    valid_label = valid_label_base
        #elif 'part-peak' in args.check_point_path:
        #    valid_label = valid_label_peak

        if args.label_scaling == 'normalized_sum':
            valid_label_normalize = valid_label / (torch.sum(valid_label, dim=1, keepdim=True) + 1e-10)
            valid_label_sum = torch.sum(valid_label, dim=1, keepdim=False)
        else:
            raise ValueError('wrong label_scaling')

        with torch.no_grad():
            pred_logits = net(data)
            #loss, pred = compute_loss(valid_label_normalize, pred_logits, loss_fn, args)
            loss, pred, loss_lst = compute_loss(valid_label_normalize, valid_label_sum, pred_logits, loss_fn, args, verbose=True)

        #prediction.append(pred.detach().cpu().numpy())
        prediction.append((pred*valid_label_sum.unsqueeze(-1)).detach().cpu().numpy())

        label_gt.append(valid_label.detach().cpu().numpy())
        label_norm.append(valid_label_normalize.detach().cpu().numpy())
        cif_ids.append(data.cif_id)
        losses.append(loss_lst.detach().cpu().numpy())

        total_loss += loss
        valid_counter += 1


    avg_loss = total_loss / valid_counter

    prediction = np.concatenate(prediction, axis=0)
    label_gt = np.concatenate(label_gt, axis=0)
    label_norm = np.concatenate(label_norm, axis=0)
    cif_ids = np.concatenate(cif_ids, axis=0)
    losses = np.concatenate(losses, axis=0)

    return prediction, label_gt, label_norm, avg_loss.cpu().numpy(), cif_ids, losses

prediction_list = []
label_gt_list = []
label_norm_list = []
avg_loss_list = []

for i in range(1):
    print(i)
    prediction, label_gt, label_norm, avg_loss, cif_ids, losses = test()
    prediction_list.append(np.expand_dims(prediction, axis=0))
    label_gt_list.append(np.expand_dims(label_gt, axis=0))
    label_norm_list.append(np.expand_dims(label_norm, axis=0))
    avg_loss_list.append(avg_loss)

avg_loss = np.mean(avg_loss_list)
prediction = np.concatenate(prediction_list, axis=0)
label_gt = np.concatenate(label_gt_list, axis=0)
label_norm = np.concatenate(label_norm_list, axis=0)

prediction = np.mean(prediction, axis=0)
label_gt = np.mean(label_gt, axis=0)
label_norm = np.mean(label_norm, axis=0)

print(prediction.shape, label_gt.shape, label_norm.shape)
print(cif_ids.shape, losses.shape)
print(list(zip(losses, cif_ids))[:10])


losses, cif_ids, prediction, label_gt, label_norm = zip(* sorted(zip(losses, cif_ids, prediction, label_gt, label_norm), key=lambda x: x[0]))
losses = np.array(losses)
cif_ids = np.array(cif_ids)
prediction = np.array(prediction)
label_gt = np.array(label_gt)
label_norm = np.array(label_norm)

print(prediction.shape, label_gt.shape, label_norm.shape)
print(cif_ids.shape, losses.shape)
print(list(zip(losses, cif_ids))[:10])



def plot(y1, y2, idx, cif, path):
    
    plot_lower = False
    plot_raw = False

    interval = 3
    spl_method = 'quadratic'

    l = len(y1)
    x = np.arange(l)

    if not plot_lower and not plot_raw:
        plt.plot(x, y1, color='red', label='pred')
    plt.plot(x, y2, color='blue', label='label')
    #plt.plot(x, y1-y2, '--', color='green', alpha=0.3)
    if not plot_lower and not plot_raw:
        plt.legend()

    plt.xlabel('Energy (eV)')
    plt.ylabel('eDoS (states/eV)')
    plt.xticks([0, 31, 63, 95, 127], ['-4', '-2', '0', '2', '4'])
    plt.title(cif)


    if plot_lower:
        y = y2
        x_lst = [0]
        if y[0] < y[1]:
            y_lst = [y[0]]
        else:
            y_lst = [0.]
        for i in range(1, l-1):
            valid = True
            for j in range(1, interval):
                if i-j>=0 and i+j < l and not (y[i-j] >= y[i] <= y[i+j]):
                    valid = False
                    break
            if valid:
                x_lst.append(i)
                y_lst.append(y[i])
        x_lst.append(l-1)
        if y[l-2] < y[l-1]:
            y_lst.append(y_lst[-1])
        else:
            y_lst.append(y[l-1])
        if y[0] >= y[1] and len(y_lst) > 1:
            y_lst[0] = y_lst[1]

        f = interp1d(x_lst, y_lst, spl_method)
        y_base = f(x)

    def refine_curve(x_lst, y_lst, x, y, y_base):
        valid = False
        while not valid:
            l = len(x)
            valid = True
            cont = False
            prev_diff = 0.
            max_diff = 0.
            #print("************")
            #print("x:", x_lst)
            #print("y:", y_lst)
            dic = set(x_lst)
            for i, y_val, y_base_val in zip(x, y, y_base):
                if i < l-1:
                    cur_diff = y[i+1] - y[i]
                if y_val < y_base_val and i not in dic:
                    valid = False
                    if not cont:
                        px = i
                        py = y_val
                        cont = True
                        max_diff = abs(cur_diff - prev_diff)
                    else:
                        if abs(cur_diff - prev_diff) > max_diff:
                            max_diff = abs(cur_diff - prev_diff)
                            px = i
                            py = y_val
                else:
                    if cont and px not in dic:
                        cont = False
                        max_diff = 0.
                        x_lst.append(px)
                        y_lst.append(py)

                prev_diff = cur_diff

            if cont and px not in dic:
                x_lst.append(px)
                y_lst.append(py)
            
            if valid:
                break

            x_lst, y_lst = list(zip(*sorted(list(zip(x_lst, y_lst)), key = lambda x: x[0])))
            x_lst, y_lst = list(x_lst), list(y_lst)

            f = interp1d(x_lst, y_lst, spl_method)
            y_base = f(x)
            y_base = [max(y1, 0.) for y1 in y_base]
        
        y_base = [min(y1, y2) for y1, y2 in zip(y_base, y)]

        return y_base

    if plot_lower:
        y_base = refine_curve(x_lst, y_lst, x, y, y_base)
        plt.plot(x, y_base, color='green', alpha=0.8)

    plt.savefig(path+'/'+f'{idx}_'+cif+'.png')
    plt.clf()

#print(prediction.shape, label_gt.shape)                                                                                                                                      
plot_path = 'plots/' + args.check_point_path.replace('./TRAINED/', '').replace('.chkpt', '')
mkdirs(plot_path)
N = len(prediction)
#for i, (pred, label) in enumerate(zip(prediction, label_norm)):
for i, (pred, label, cif) in enumerate(zip(prediction, label_gt, cif_ids)):
    if i % 500 == 0:
        print(i, '/', N)
    if i < 300:
        plot(pred, label, i, cif, plot_path)


#np.save('./RESULT/prediction_xtal2dos_allsamples_' + args.data_src + '_' + args.label_scaling + '_' \
#        + args.xtal2dos_loss_type + '_trainsize' + str(args.trainset_subset_ratio) + '.npy', prediction_x)


result_path = './RESULT/'
mkdirs(result_path)

if args.finetune:
    result_path = result_path + '/finetune/' + args.finetune_dataset + '/'

if args.ablation_LE:
    result_path = result_path + '/ablation_LE/'

if args.ablation_CL:
    result_path = result_path + '/ablation_CL/'

makedirs(result_path, exist_ok=True)

mae, r2, mse, wd, mae_ori, r2_ori, mse_ori, wd_ori = metrics(label_gt+1e-10, prediction, args, mode="normalized_sum")
print("\n********** TESTING STATISTIC ***********")
print("avg_loss =%.6f\t" % avg_loss)
print("r2=%.6f\t  r2_ori=%.6f" % (r2, r2_ori))
print("mae=%.6f\t  mae_ori=%.6f" % (mae, mae_ori))
print("mse=%.6f\t  mse_ori=%.6f" % (mse, mse_ori))
print("wd=%.6f\t  wd_ori=%.6f" % (wd, wd_ori))
print("\n*****************************************")
exit()


if args.label_scaling == 'standardized':
    print('\n > label scaling: std')
    mean = NORMALIZER.mean.detach().numpy()
    std = NORMALIZER.std.detach().numpy()
    label_gt_standardized = (label_gt - mean) / std
    mae = np.mean(np.abs((prediction) - label_gt_standardized))
    mae_x = np.mean(np.abs((prediction_x) - label_gt_standardized))
    #if args.data_src != 'no_label_128' and args.data_src != 'no_label_32':
    prediction = prediction * std + mean
    prediction_x = prediction_x * std + mean
    prediction_x_std = prediction_x_std * std
    prediction[prediction < 0] = 1e-6
    prediction_x[prediction_x < 0] = 1e-6
    mae_ori = np.mean(np.abs((prediction)-label_gt))
    mae_x_ori = np.mean(np.abs((prediction_x)-label_gt))

    ## save results ##
    if args.data_src != 'no_label_128' and args.data_src != 'no_label_32':
        np.save(result_path + 'label_gt_' + args.data_src + '_' + args.label_scaling + '_' \
                + args.xtal2dos_loss_type + '_trainsize' + str(args.trainset_subset_ratio) + '.npy', label_gt)
        np.save(result_path + 'label_mean_' + args.data_src + '_' + args.label_scaling + '_' \
                + args.xtal2dos_loss_type + '_trainsize' + str(args.trainset_subset_ratio) + '.npy', mean)
        np.save(result_path + 'label_std_' + args.data_src + '_' + args.label_scaling + '_' \
                + args.xtal2dos_loss_type + '_trainsize' + str(args.trainset_subset_ratio) + '.npy', std)
    np.save(result_path + 'prediction_xtal2dos_' + args.data_src + '_' + args.label_scaling + '_' \
            + args.xtal2dos_loss_type + '_trainsize' + str(args.trainset_subset_ratio) + '.npy', prediction_x)
    #np.save(result_path + 'prediction_xtal2dos_standard_deviation_' + args.data_src + '_' + args.label_scaling + '_' \
    #                + args.xtal2dos_loss_type + '_trainsize' + str(args.trainset_subset_ratio) + '.npy', prediction_x_std)
    testing_mpid = pd_data.iloc[test_idx]
    testing_mpid.to_csv(result_path + 'testing_mpids' + args.data_src + '_' + args.label_scaling + '_' \
                        + args.xtal2dos_loss_type + '_trainsize' + str(args.trainset_subset_ratio) + '.csv', index=False, header=True)

elif args.label_scaling == 'normalized_max':
    print('\n > label scaling: norm max')
    label_max = np.expand_dims(np.max(label_gt, axis=1), axis=1)
    label_gt_standardized = label_gt / label_max
    mae = np.mean(np.abs((prediction) - label_gt_standardized))
    mae_x = np.mean(np.abs((prediction_x) - label_gt_standardized))
    if args.data_src != 'no_label_128' and args.data_src != 'no_label_32':
        prediction = prediction * label_max
        prediction_x = prediction_x * label_max
        prediction_x_std = prediction_x_std * label_max
    mae_ori = np.mean(np.abs((prediction) - label_gt))
    mae_x_ori = np.mean(np.abs((prediction_x) - label_gt))

    ## save results ##
    if args.data_src != 'no_label_128' and args.data_src != 'no_label_32':
        np.save(result_path + 'label_gt_' + args.data_src + '_' + args.label_scaling + '_' \
                + args.xtal2dos_loss_type + '_trainsize' + str(args.trainset_subset_ratio) + '.npy', label_gt)
        np.save(result_path + 'label_max_' + args.data_src + '_' + args.label_scaling + '_' \
                + args.xtal2dos_loss_type + '_trainsize' + str(args.trainset_subset_ratio) + '.npy', label_max)
    np.save(result_path + 'prediction_xtal2dos_' + args.data_src + '_' + args.label_scaling + '_' \
            + args.xtal2dos_loss_type + '_trainsize' + str(args.trainset_subset_ratio) + '.npy', prediction_x)
    #np.save(result_path + 'prediction_xtal2dos_standard_deviation_' + args.data_src + '_' + args.label_scaling + '_' \
    #                + args.xtal2dos_loss_type + '_trainsize' + str(args.trainset_subset_ratio) + '.npy', prediction_x_std)
    testing_mpid = pd_data.iloc[test_idx]
    testing_mpid.to_csv('testing_mpids' + args.data_src + '_' + args.label_scaling + '_' \
                    + args.xtal2dos_loss_type + '_trainsize' + str(args.trainset_subset_ratio) + '.csv', index=False, header=True)

elif args.label_scaling == 'normalized_sum':
    print('\n > label scaling: norm sum')
    assert args.xtal2dos_loss_type == 'KL' or args.xtal2dos_loss_type == 'WD'
    label_sum = np.sum(label_gt, axis=1, keepdims=True)
    label_gt_standardized = label_gt / label_sum
    mae = np.mean(np.abs((prediction) - label_gt_standardized))
    mae_x = np.mean(np.abs((prediction_x) - label_gt_standardized))
    if args.data_src != 'no_label_128' and args.data_src != 'no_label_32':
        prediction = prediction * label_sum
        prediction_x = prediction_x * label_sum
        prediction_x_std = prediction_x_std * label_sum
    mae_ori = np.mean(np.abs((prediction) - label_gt))
    mae_x_ori = np.mean(np.abs((prediction_x) - label_gt))

    ## save results ##
    if args.data_src != 'no_label_128' and args.data_src != 'no_label_32':
        np.save(result_path + 'label_gt_' + args.data_src + '_' + args.label_scaling + '_' \
                + args.xtal2dos_loss_type + '_trainsize' + str(args.trainset_subset_ratio) + '.npy', label_gt)
        np.save(result_path + 'label_sum_' + args.data_src + '_' + args.label_scaling + '_' \
                + args.xtal2dos_loss_type + '_trainsize' + str(args.trainset_subset_ratio) + '.npy', label_sum)
    np.save(result_path + 'prediction_xtal2dos_' + args.data_src + '_' + args.label_scaling + '_' \
            + args.xtal2dos_loss_type + '_trainsize' + str(args.trainset_subset_ratio) + '.npy', prediction_x)
    #np.save(result_path + 'prediction_xtal2dos_standard_deviation_' + args.data_src + '_' + args.label_scaling + '_' \
    #                + args.xtal2dos_loss_type + '_trainsize' + str(args.trainset_subset_ratio) + '.npy', prediction_x_std)
    testing_mpid = pd_data.iloc[test_idx]
    testing_mpid.to_csv(result_path + 'testing_mpids_' + args.data_src + '_' + args.label_scaling + '_' \
                        + args.xtal2dos_loss_type + '_trainsize' + str(args.trainset_subset_ratio) + '.csv', index=False, header=True)

print("\n********** TESTING STATISTIC ***********")
print("total_loss =%.6f\t nll_loss =%.6f\t nll_loss_x =%.6f\t kl_loss =%.6f\t" %
      (total_loss_smooth, nll_loss_smooth, nll_loss_x_smooth, kl_loss_smooth))
print("mae=%.6f\t mae_x=%.6f\t mae_ori=%.6f\t mae_x_ori=%.6f" % (mae, mae_x, mae_ori, mae_x_ori))
print("\n*****************************************")

print(f"> DONE TESTING !")
