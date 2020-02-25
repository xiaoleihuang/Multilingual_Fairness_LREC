'''Scripts for evaluation,
    metrics: (macro) F1, AUC, FNED, FPED

    Because the data is skewed distributed, therefore,
    we use the macro f1 score to measure the performance.
'''
import pandas as pd
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
import numpy as np

import json
from collections import Counter


def cal_fpr(fp, tn):
    '''False positive rate'''
    return fp/(fp+tn)


def cal_fnr(fn, tp):
    '''False negative rate'''
    return fn/(fn+tp)


def cal_tpr(tp, fn):
    '''True positive rate'''
    return tp/(tp+fn)


def cal_tnr(tn, fp):
    '''True negative rate'''
    return tn/(tn+fp)


def eval(dpath, opt):
    '''Fairness Evaluation
        dpath: input eval file path
        opt: output results path
    '''
    df = pd.read_csv(dpath, sep='\t', na_values='x')
    # get the task name from the file, gender or ethnicity
    tasks = ['gender', 'age', 'country', 'ethnicity']

    scores = {
        'accuracy': 0.0,
        'f1-macro': 0.0, # macro f1 score
        'f1-weight': 0.0, # weighted f1 score
        'auc': 0.0,
    }

    # accuracy, f1, auc
    scores['accuracy'] = metrics.accuracy_score(
        y_true=df.label, y_pred=df.pred
    )
    scores['f1-macro'] = metrics.f1_score(
        y_true=df.label, y_pred=df.pred,
        average='macro'
    )
    scores['f1-weight'] = metrics.f1_score(
        y_true=df.label, y_pred=df.pred,
        average='weighted'
    )
    fpr, tpr, _ = metrics.roc_curve(
        y_true=df.label, y_score=df.pred_prob,
    )
    scores['auc'] = metrics.auc(fpr, tpr)

    '''fairness gaps'''
    for task in tasks:

        '''Filter out some tasks'''
        if ('Polish' in dpath or 'Italian' in dpath) and\
             task in ['country', 'ethnicity']:
            continue

        scores[task] = {
            'fned': 0.0, # gap between fnr
            'fped': 0.0, # gap between fpr
            'tped': 0.0, # gap between tpr
            'tned': 0.0, # gap between tnr
        }
        # filter out the one does not have attributes
        task_df = df[df[task].notnull()]
    
        # get overall confusion matrix
        tn, fp, fn, tp = metrics.confusion_matrix(
            y_true=task_df.label, y_pred=task_df.pred
        ).ravel()

        # get the unique types of demographic groups
        uniq_types = task_df[task].unique()
        for group in uniq_types:
            # calculate group specific confusion matrix
            group_df = task_df[task_df[task] == group]
            
            g_tn, g_fp, g_fn, g_tp = metrics.confusion_matrix(
                y_true=group_df.label, y_pred=group_df.pred
            ).ravel()

            # calculate and accumulate the gaps
            scores[task]['fned'] = scores[task]['fned'] + abs(
                cal_fnr(fn, tp)-cal_fnr(g_fn, g_tp)
            )
            scores[task]['fped'] = scores[task]['fped'] + abs(
                cal_fpr(fp, tn)-cal_fpr(g_fp, g_tn)
            )
            scores[task]['tped'] = scores[task]['tped'] + abs(
                cal_tpr(tp, fn)-cal_tpr(g_tp, g_fn)
            )
            scores[task]['tned'] = scores[task]['tned'] + abs(
                cal_tnr(tn, fp)-cal_tnr(g_tn, g_fp)
            )
    with open(opt, 'w') as wfile:
        wfile.write(json.dumps(scores))
    print(scores)


def data_iter(datap, batch_size=64, if_shuffle=True, if_sample=False):
    doc_idx = 2
    data = {'x': [], 'y': []}
    class_wt = dict()
    
    with open(datap) as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split('\t')
            # split indices
            data['x'].append(list(map(int, line[doc_idx].split())))
            data['y'].append(int(line[-1]))

    # if over sample the minority    
    if if_sample:
        label_count = Counter(data['y'])
        for label_tmp in label_count:
            sample_num = label_count.most_common(1)[0][1] - label_count[label_tmp]
            if sample_num == 0:
                continue
            sample_indices = np.random.choice(
                list(range(len(data['y']))),
                size=sample_num
            )
            for idx in sample_indices:
                data['x'].append(data['x'][idx])
                data['y'].append(data['y'][idx])
            
    # calculate the class weight
    class_wt = dict(zip(
        np.unique(data['y']), compute_class_weight(
            'balanced', np.unique(data['y']), 
            data['y']
        )
    ))

    # if shuffle the dataset
    if if_shuffle:
        data['x'], data['y'] = shuffle(data['x'], data['y'])

    steps = len(data['x']) // batch_size
    if len(data['x']) % batch_size != 0:
        steps += 1

    for step in range(steps):
        yield class_wt, \
            np.asarray(data['x'][step*batch_size: (step+1)*batch_size]),\
            np.asarray(data['y'][step*batch_size: (step+1)*batch_size])

