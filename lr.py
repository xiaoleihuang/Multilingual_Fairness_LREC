'''Run logistic regression classifier
'''
import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle

import evaluator


def build_lr(lang, odir):
    '''Train, valid, test lr
    
        lang: The language name
        odir: output directory of prediction results
    '''
    doc_idx = 2
    encode_dir = './data/encode/'+lang+'/'
    split_dir = './data/split/'+lang+'/'
    res_dir = './resources/classifier/lr/'
    
    vec_path = res_dir + lang + '.vect'
    clf_path = res_dir + lang + '.clf'
    # don't load classifier for debug usage

    print('Building vectorizer...')
    if os.path.exists(vec_path):
        vect = pickle.load(open(vec_path, 'rb'))
    else:
        corpus = []
        with open(encode_dir + 'corpus.tsv') as dfile:
            dfile.readline() # skip column names
            for line in dfile:
                line = line.strip().split('\t')
                corpus.append(line[doc_idx])

        vect = TfidfVectorizer(
            ngram_range=(1, 3), max_features=15000)
        vect.fit(corpus)
        pickle.dump(vect, open(vec_path, 'wb'))

    print('Building classifier...')
    # load training data
    data = {'x':[], 'y':[]}
    with open(split_dir+'train.tsv') as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split('\t')
            data['x'].append(line[doc_idx])
            data['y'].append(int(line[-1]))

    # calculate the weight of labels
    weights = dict(zip(
        np.unique(data['y']), compute_class_weight(
            'balanced', np.unique(data['y']), 
            data['y']
        )
    ))
    # shuffle the data before training
    data['x'], data['y'] = shuffle(data['x'], data['y'])

    # build classifier
    clf = LogisticRegression(
        class_weight=weights, solver='liblinear')
    clf.fit(vect.transform(data['x']), data['y'])
    # save the classifier
    pickle.dump(clf, open(clf_path, 'wb'))
    
    # test the classifier
    data = []
    with open(split_dir+'test.tsv') as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split('\t')
            data.append(line[doc_idx])

    data = vect.transform(data)
    y_preds = clf.predict(data)
    y_probs = clf.predict_proba(data)

    # save the test results
    with open(odir+lang+'.tsv', 'w') as wfile:
        with open(split_dir+'test.tsv') as dfile:
            wfile.write(
                dfile.readline().strip()+'\tpred\tpred_prob\n'
            )
            for idx, line in enumerate(dfile):
                # 1 is the hate speech label
                wfile.write(line.strip()+'\t'+str(y_preds[idx])+'\t'+str(y_probs[idx][1])+'\n')

    # save the predicted results
    evaluator.eval(
        odir+lang+'.tsv', 
        odir+lang+'.score'
    )


if __name__ == '__main__':
    langs = [
        'English', 'Italian', 'Polish', 
        'Portuguese', 'Spanish'
    ]
    odir = './results/lr/'
    if not os.path.exists(odir):
        os.mkdir(odir)

    for lang in langs:
        print('Working on: ', lang)
        build_lr(lang, odir)

