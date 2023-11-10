import copy
from time import time

from random import Random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc
from torch import nn
from torch.autograd import Variable
from torch.utils import data

torch.manual_seed(2)  # reproducible torch:2 np:3
np.random.seed(3)
from argparse import ArgumentParser
from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import *
from tqdm import tqdm


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print("DEVCIE : ",device)

parser = ArgumentParser(description='MolTrans Training.')
parser.add_argument('-m', '--mode', type=str, default='classification', choices=['classification','regression'])
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--task', choices=['biosnap', 'bindingdb', 'davis', 'mydata'],
                    default='', type=str, metavar='TASK',
                    help='Task name. Could be biosnap, bindingdb and davis.')
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-d', '--duplicates', default='y', type=str, help='duplicates allow')
parser.add_argument('--random_state', '-rs', default=42, type=int)

args = parser.parse_args()

def get_task(task_name):
    if task_name.lower() == 'biosnap':
        return './dataset/BIOSNAP/full_data'
    elif task_name.lower() == 'bindingdb':
        return './dataset/BindingDB'
    elif task_name.lower() == 'davis':
        return './dataset/DAVIS'
    elif task_name.lower() == 'mydata':
        return './dataset/Mydata/'

def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    
    if args.mode == 'classification':
        print("Classification testing")
        for i, (d, p, d_mask, p_mask, label, pka) in enumerate(data_generator):
            score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())
            m = torch.nn.Sigmoid()

            logits = torch.squeeze(m(score))
            
            loss_fct = torch.nn.BCELoss()
            
            label = Variable(torch.from_numpy(np.array(label)).float()).cuda()
            
            loss = loss_fct(logits, label)
            loss_accumulate += loss
            count += 1
            logits = logits.detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()

        loss = loss_accumulate / count
        fpr, tpr, thresholds = roc_curve(y_label, y_pred)
        precision = tpr / (tpr + fpr)

        f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
        thred_optim = thresholds[5:][np.argmax(f1[5:])]

        print("optimal threshold : " + str(round(thred_optim,6)))
        y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)] # optimal pred_s for assess precision, recall

        auc_k = auc(fpr, tpr)
        print("AUROC : " + str(round(auc_k,6)))
        print("AUPRC : " + str(round(average_precision_score(y_label, y_pred),6)))

        cm1 = confusion_matrix(y_label, y_pred_s)
        print('Confusion Matrix : \n', cm1)
        print('Recall      :', round(recall_score(y_label, y_pred_s),6))
        print('Precision   :', round(precision_score(y_label, y_pred_s),6))

        total1 = sum(sum(cm1))
        #####from confusion matrix calculate accuracy
        accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
        print('Accuracy    :', round(accuracy1,6))

        sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
        print('Sensitivity :', round(sensitivity1,6))

        specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
        print('Specificity :', round(specificity1,6))

        outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)]) # Hard prediction
        return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), y_pred, loss.item()
 
    if args.mode == 'regression':
        print("Regression testing")
        for i, (d, p, d_mask, p_mask, label, pka) in enumerate(data_generator):
            score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())
            # m = torch.nn.Sigmoid()
            logits = torch.squeeze(score)
            loss_fct = torch.nn.MSELoss()
            pka = Variable(torch.from_numpy(np.array(pka)).float()).cuda()
            loss = loss_fct(logits, pka)
            loss_accumulate += loss
            count += 1
            logits = logits.detach().cpu().numpy()
            label_ids = pka.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist() # Ground Truth
            y_pred = y_pred + logits.flatten().tolist() # Predicted Value

            G = np.asarray(y_label).flatten()
            P = np.asarray(y_pred).flatten()

        loss = loss_accumulate / count
        ret = [rmse(G,P), mse(G,P), pearson(G,P), spearman(G,P), ci(G,P)]
        # print("CurStep RMSE : {0}, MSE : {1}, CI : {2}, pearson : {3}, spearman : {4}".format(ret[0], ret[1], ret[-1], ret[2], ret[3]))

        return ret


def main():
    config = BIN_config_DBPE()
    config['batch_size'] = args.batch_size

    loss_history = []

    model = BIN_Interaction_Flat(**config)
    
    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim=0)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print('--- Data Preparation ---')
    params = {'batch_size': args.batch_size,
              'num_workers': args.workers,
              'drop_last': True}

    df = get_task(args.task)
    
    try:
        print("Create Fold STRAT!")
        df = pd.read_csv(df + 'Mydata.csv')
        df_train, df_val, df_test = create_fold(df, random_state=args.random_state)
        print("Create Fold DONE!")
    except:
        print("Moltrans origianl split mode START!")
        df_train = pd.read_csv(df + '/train.csv')
        df_val = pd.read_csv(df + '/val.csv')
        df_test = pd.read_csv(df + '/test.csv')
        print("Moltrans Origianl split mode DONE!")
        
    # SMILES, Target Sequence, Label, pka

    training_set = BIN_Data_Encoder(df_train.index.values, df_train.Label.values, df_train.pka.values, df_train)
    training_generator = data.DataLoader(training_set, **params, shuffle=True)

    validation_set = BIN_Data_Encoder(df_val.index.values, df_val.Label.values, df_val.pka.values, df_val)
    validation_generator = data.DataLoader(validation_set, **params, shuffle=False)

    testing_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test.pka.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params, shuffle=False)

    if args.mode == 'classification':
        print("Classification training start")
        
        # early stopping
        max_auc = 0
        model_max = copy.deepcopy(model)

        with torch.set_grad_enabled(False):
            auc, auprc, f1, logits, loss = test(testing_generator, model_max)
            print('Initial Testing AUROC :' + str(round(auc,4)) + ', AUPRC :' + str(round(auprc,4)) + ', F1 :' + str(round(f1,4)) + \
                    ' , Test loss: ' + str(round(loss,4)))

        print('----- Go for Training -----')
        torch.backends.cudnn.benchmark = True
        for epo in range(args.epochs):
            model.train()
            for i, (d, p, d_mask, p_mask, label, pka) in enumerate(tqdm(training_generator)):
                score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())

                label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

                loss_fct = torch.nn.BCELoss()
                m = torch.nn.Sigmoid()
                n = torch.squeeze(m(score))

                loss = loss_fct(n, label)
                loss_history.append(loss.item())

                opt.zero_grad()
                loss.backward()
                opt.step()

                if (i % 500 == 0):
                    print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                        loss.cpu().detach().numpy()))

            # every epoch test
            with torch.set_grad_enabled(False):
                AUC, auprc, f1, logits, loss = test(validation_generator, model_max)
                if AUC > max_auc:
                    model_max = model
                    max_auc = AUC
                    counter = 0
                else:
                    counter +=1
                    print("Patience : {} Cur Count : {}".format(config['patience'], counter))

                if counter >= config['patience']:
                    print("Early stopping at epoch #{}".format(epo+1))
                    break

            print('Validation at Epoch ' + str(epo + 1) + ' , AUROC: ' + str(AUC) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1))
            print("=================================\n")

        print('================ Go for Testing ===============\n')
        try:
            with torch.set_grad_enabled(False):
                auc, auprc, f1, logits, loss = test(testing_generator, model_max)
                print('Testing AUROC: ' + str(round(auc,6)) + ' , AUPRC: ' + str(round(auprc,6)) + ' , F1: ' + str(round(f1,6)) + \
                        ' , Test loss: ' + str(round(loss,6)))
        except:
            print('testing failed')
        return model_max, loss_history

    elif args.mode == 'regression':
        print("Regression training start")
        # early stopping
        max_RMSE = 1000
        model_max = copy.deepcopy(model)

        with torch.set_grad_enabled(False):
            result = test(testing_generator, model_max) # rmse(G,P), mse(G,P), pearson(G,P), spearman(G,P), ci(G,P)
            print('Initial Testing RMSE: ' + str(result[0]) + ' , MSE: ' + str(result[1]) + ' , pearson: ' + str(result[2]) + ' , spearman : ' + str(result[3]) + ' , ci : ' + str(result[-1]))

        print('--- Go for Training ---')
        torch.backends.cudnn.benchmark = True
        for epo in range(args.epochs):
            model.train()
            for i, (d, p, d_mask, p_mask, label,pka) in enumerate(tqdm(training_generator)):
                score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())

                label = Variable(torch.from_numpy(np.array(pka)).float()).cuda()

                loss_fct = torch.nn.MSELoss()
                n = torch.squeeze(score)

                loss = loss_fct(n, label)
                loss_history.append(loss.item())

                opt.zero_grad()
                loss.backward()
                opt.step()

            # every epoch test
            with torch.set_grad_enabled(False):
                result = test(validation_generator, model_max)
                #AUC, auprc, f1, logits, loss = test(validation_generator, model_max)
                if result[0] < max_RMSE:
                    model_max = model
                    max_RMSE = result[0]
                    counter = 0
                else:
                    counter +=1
                    print("Patience : {} Cur Count : {}".format(config['patience'], counter))

                if counter >= config['patience']:
                    print("Early stopping at epoch #{}".format(epo+1))
                    break

            print('['+str(epo+1)+'] Validation RMSE: ' + str(result[0]) + ' , MSE: ' + str(result[1]) + ' , pearson: ' + str(result[2]) + ' , spearman : ' + str(result[3]) + ' , ci : ' + str(result[-1]))
            print("=================================\n")

        print('================ Go for Testing ===============\n')
        try:
            with torch.set_grad_enabled(False):
                result = test(testing_generator, model_max)
                print('Testing RMSE: ' + str(result[0]) + ' , MSE: ' + str(result[1]) + ' , pearson: ' + str(result[2]) + ' , spearman : ' + str(result[3]) + ' , ci : ' + str(result[-1]))

        except:
            print('testing failed')
        return model_max, loss_history


s = time()
model_max, loss_history = main()
e = time()
print(e - s)
