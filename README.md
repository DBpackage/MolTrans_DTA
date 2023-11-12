# MolTrans: Molecular Interaction Transformer for Drug Target Interaction Prediction

Drug target interaction (DTI) prediction is a foundational task for in-silico drug discovery, which is costly and time-consuming due to the need of experimental search over large drug compound space. Recent years have witnessed promising progress for deep learning in DTI predictions. However, the following challenges are still open: (1) existing molecular representation learning approaches ignore the sub-structural nature of DTI, thus produce results that are less accurate and difficult to explain; (2) existing methods focus on limited labeled data while ignoring the value of massive unlabelled molecular data. We propose a Molecular Interaction Transformer (MolTrans) to address these limitations via: (1) knowledge inspired sub-structural pattern mining algorithm and interaction modeling module for more accurate and interpretable DTI prediction; (2) an augmented transformer encoder to better extract and capture the semantic relations among substructures extracted from massive unlabeled biomedical data. We evaluate MolTrans on real world data and show it improved DTI prediction performance compared to state-of-the-art baselines.


## Datasets

In the dataset folder, I added a Mydata/Mydata.csv file for running both classification/regression. You can make your own dataset for running too.
Just make csv file with this format 

| SMILES  | Target Sequence | pka | Label |
| ------------- | ------------- |------------- |------------- |
| COc1cc(CCCOC(=O)  | MDVLSPGQGNNTTS  |10.34969248 | 1 |
| OC(=O)C=C | MSWATRPPF  |5.568636236 | 0

You can make Label column from pka column by setting any threshold value.
Make Mydata.csv file and just replace the Mydata.csv which is I uploaded.

INDEED you can make other directories or other csv file names, but if you want to do, then you need to fix some codes in train.py for data importing.

1. def get_task(task_name):
2. line 165 ~ 177

Should be adjusted to your data names.


## Run

if you want to run regression mode, then

**python train.py --task mydata -b 4 --mode regression -rs 42**

else you want to run classification mode, then

**python train.py --task mydata -b 4 --mode classification -rs 42**

There are some arguements for running models.

```
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
parser.add_argument('--random_state', '-rs', default=42, type=int)
```
