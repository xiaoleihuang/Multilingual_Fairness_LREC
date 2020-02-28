'''Supervised perspectives (Three levels of predictability) : word; topic; part-of-speech;

Draw a heatmap
'''
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import numpy as np
import pandas as pd

import analysis_utils
import pickle
import os
import json
from collections import OrderedDict


def plot_coefficients(
        clf, feature_names, top_features=20, 
        title='Top Predictable Features of Gender',
        opt='test.pdf'
    ):
    '''
        Visualize top word features for demographic predictability
    '''
    coef = clf.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features*20:]
    top_positive_coefficients = [
        item for item in top_positive_coefficients 
            if item < len(feature_names) and len(feature_names[item].split())<2
    ][:top_features]
    top_negative_coefficients = np.argsort(coef)[:top_features*20]
    top_negative_coefficients = [
        item for item in top_negative_coefficients 
            if item < len(feature_names) and len(feature_names[item].split())<2
    ][:top_features]

    top_coefficients = np.hstack(
        [top_negative_coefficients, top_positive_coefficients]
    )

    # create plot
    plt.figure(figsize=(15, 10))
    colors = [
        'orange' if c < 0 else 'deepskyblue' 
            for c in coef[top_coefficients]
    ]
    plt.bar(
        np.arange(2 * top_features), 
        coef[top_coefficients], color=colors
    )
    feature_names = np.array(feature_names)
    plt.xticks(
        np.arange(0, 2 * top_features), 
        feature_names[top_coefficients], 
        rotation=30, ha='center'
    )
    plt.title(title)
    plt.savefig(opt, format='pdf')
    plt.close()


def word_level(dpath):
    '''Explore how predictability of texts towards demographic attributes
    '''
    vect_path = './predictability/word/vect.pkl' # the path to save vectorizer
    clf_dir = './predictability/word/' # the directory path to save classifiers
    text_idx = 2 # the column idx of tweets in tsv file
    seed = 33 # random seed for split data
    results = dict()
    
    # load or build vectorizer
    print('Building Vectorizer...')
    if os.path.exists(vect_path):
        vect = pickle.load(open(vect_path, 'rb'))
    else:
        vect = TfidfVectorizer(
            ngram_range=(1, 3), max_df=0.8,
            min_df=2, max_features=15000
        )
        docs = []
        with open(dpath) as dfile:
            dfile.readline() # skip the column names

            for line in dfile:
                line = line.strip().split('\t')
                docs.append(line[text_idx])

        vect.fit(docs)
        pickle.dump(vect, open(vect_path, 'wb'))

    # loop through attributes
    att_idx = {
        'age': 5,
        'country': 7,
        'ethnicity': 8,
        'gender': 4,
        'region': 6,
    }

    for att in att_idx:
        print('Working on: ', att)
        # load the data
        data = {'x':[], 'y':[]}
        with open(dpath) as dfile:
            dfile.readline()
            for line in dfile:
                line = line.strip().split('\t')

                # skip the line that the target attribute is null
                if line[att_idx[att]] == 'x':
                    continue

                data['x'].append(line[text_idx])
                data['y'].append(int(line[att_idx[att]]))

        # split the data
        print('Split Data... ', att)
        data_idx = list(range(len(data['x'])))
        np.random.seed(seed) # set for reproductivity
        np.random.shuffle(data_idx)

        train_idx = data_idx[:int(len(data_idx)*.8)]
        test_idx = data_idx[int(len(data_idx)*.8):]

        train_data = {
            'x': vect.transform([data['x'][item] for item in train_idx]),
            'y': [data['y'][item] for item in train_idx]
        }
        test_data = {
            'x': vect.transform([data['x'][item] for item in test_idx]),
            'y': [data['y'][item] for item in test_idx]
        }
        data = None
        del data
        
        # build clf
        print('Training Classifier... ', att)
        if os.path.exists(clf_dir+att+'_'+str(seed)+'.clf'):
            clf = pickle.load(open(clf_dir+att+'_'+str(seed)+'.clf', 'rb'))
        else:
            if len(np.unique(train_data['y'])) > 2:
                clf = LogisticRegression(
                    class_weight='balanced', solver='lbfgs', n_jobs=-1,
                    multi_class='multinomial', max_iter=2000,
                )
            else:
                clf = LogisticRegression(
                    class_weight='balanced', solver='liblinear'
                )

            clf.fit(train_data['x'], train_data['y'])
            # save the trained clf
            pickle.dump(clf, open(clf_dir+att+'_'+str(seed)+'.clf', 'wb')) 
        
        plot_coefficients(
            clf, vect.get_feature_names(), top_features=10, 
            title='Top Predictable Features of ' + att.capitalize(),
            opt='./images/word_'+att+'.pdf'
        )
        # evaluate the predictability by accuracy
        results[att] = clf.score(test_data['x'], test_data['y'])

    with open('./predictability/word/results.json', 'w') as wfile:
        wfile.write(json.dumps(results))

    print('Word-level, ', results)
    return results


def pos_level(dpath):
    '''Explore how predictability of texts towards demographic attributes 
        using part of speech as input features
    '''
    clf_dir = './predictability/pos/' # the directory path to save classifiers
    text_idx = 2 # the column idx of tweets in tsv file
    seed = 33 # random seed for split data
    feap = './predictability/pos/feas.pkl' # feature saving path
    results = dict()

    # obtain data
    tag_data = analysis_utils.get_pos(dpath)
    text_idx = 2 # the column idx of tweets in tsv file

    att_idx = {
        'age': 5,
        'country': 7,
        'ethnicity': 8,
        'gender': 4,
        'region': 6,
    }

    # obtain features
    if os.path.exists(feap):
        features = pickle.load(open(feap, 'rb'))
    else:
        features = dict()
        for idx, line in enumerate(tag_data):
            # filter out the attributes are all null
            if line[4] == 'x' and line[5] == 'x' and \
                line[6] == 'x' and line[7] == 'x' and line[8] == 'x':
                continue

            fea_dict = {
                'N':0.0, 'O':0.0, '^':0.0, 'S':0.0, 'Z':0.0,
                'V':0.0, 'A':0.0, 'R':0.0, '!':0.0, 'Y':0.0,
                'D':0.0, 'P':0.0, '&':0.0, 'T':0.0, 'X':0.0, 
                '#':0.0, '@':0.0, '~':0.0, 'U':0.0, 'E':0.0, 
                '$':0.0, ',':0.0, 'G':0.0, 'L':0.0, 'M':0.0,
            }

            doc = json.loads(line[text_idx])
            if len(doc) < 1:
                features[idx] = [0.0] * len(fea_dict)
                continue

            # count tags
            for item in doc:
                fea_dict[item[1]] += 1

            # normalize the values
            sum_val = sum(fea_dict.values())
            if sum_val != 0:
                features[idx] = [fea_dict[key]/sum_val for key in fea_dict]
            else:
                features[idx] = list(fea_dict.values())

        pickle.dump(features, open(feap, 'wb'))

    # loop through attributes
    for att in att_idx:
        print('Working on: ', att)
        # load the data
        data = {'x':[], 'y':[]}
        with open(dpath) as dfile:
            dfile.readline()
            for idx, line in enumerate(dfile):
                line = line.strip().split('\t')

                # skip the line that the target attribute is null
                if line[att_idx[att]] == 'x':
                    continue

                data['x'].append(features[idx])
                data['y'].append(int(line[att_idx[att]]))

        # split the data
        print('Split Data... ', att)
        data_idx = list(range(len(data['x'])))
        np.random.seed(seed) # set for reproductivity
        np.random.shuffle(data_idx)

        train_idx = data_idx[:int(len(data_idx)*.8)]
        test_idx = data_idx[int(len(data_idx)*.8):]

        train_data = {
            'x': [data['x'][item] for item in train_idx],
            'y': [data['y'][item] for item in train_idx]
        }
        test_data = {
            'x': [data['x'][item] for item in test_idx],
            'y': [data['y'][item] for item in test_idx]
        }
        data = None
        del data
        
        # build clf
        print('Training Classifier... ', att)
        if os.path.exists(clf_dir+att+'_'+str(seed)+'.clf'):
            clf = pickle.load(open(clf_dir+att+'_'+str(seed)+'.clf', 'rb'))
        else:
            if len(np.unique(train_data['y'])) > 2:
                clf = LogisticRegression(
                    class_weight='balanced', solver='lbfgs', n_jobs=-1,
                    multi_class='multinomial', max_iter=2000,
                )
            else:
                clf = LogisticRegression(
                    class_weight='balanced', solver='liblinear'
                )

            clf.fit(train_data['x'], train_data['y'])
            # save the trained clf
            pickle.dump(clf, open(clf_dir+att+'_'+str(seed)+'.clf', 'wb')) 
        
        # evaluate the predictability by accuracy
        results[att] = clf.score(test_data['x'], test_data['y'])

    with open('./predictability/pos/results.json', 'w') as wfile:
        wfile.write(json.dumps(results))

    print('POS-level, ', results)
    return results


def topic_level(dpath):
    '''Explore how predictability of texts towards demographic attributes 
        using topic distributions as input features
    '''
    clf_dir = './predictability/topic/' # the directory path to save classifiers
    text_idx = 2 # the column idx of tweets in tsv file
    seed = 33 # random seed for split data
    feap = './predictability/topic/feas.pkl' # feature saving path
    results = dict()

    # obtain data
    lda_dict, lda_model = analysis_utils.train_topics(dpath)
    text_idx = 2 # the column idx of tweets in tsv file

    att_idx = {
        'age': 5,
        'country': 7,
        'ethnicity': 8,
        'gender': 4,
        'region': 6,
    }

    # obtain features
    if os.path.exists(feap):
        features = pickle.load(open(feap, 'rb'))
    else:
        features = dict()
        with open(dpath) as dfile:
            dfile.readline() # columns

            for idx, line in enumerate(dfile):
                line = line.strip().split('\t')
                # filter out the attributes are all null
                if line[4] == 'x' and line[5] == 'x' and \
                    line[6] == 'x' and line[7] == 'x' and line[8] == 'x':
                    continue

                feas = lda_model[lda_dict.doc2bow(line[text_idx].split())]
                # the topic distributions are normalized
                features[idx] = [0.0] * lda_model.num_topics
                for item in feas:
                    features[idx][item[0]] = item[1]

        pickle.dump(features, open(feap, 'wb'))

    # loop through attributes
    for att in att_idx:
        print('Working on: ', att)
        # load the data
        data = {'x':[], 'y':[]}
        with open(dpath) as dfile:
            dfile.readline()
            for idx, line in enumerate(dfile):
                line = line.strip().split('\t')

                # skip the line that the target attribute is null
                if line[att_idx[att]] == 'x':
                    continue

                data['x'].append(features[idx])
                data['y'].append(int(line[att_idx[att]]))

        # split the data
        print('Split Data... ', att)
        data_idx = list(range(len(data['x'])))
        np.random.seed(seed) # set for reproductivity
        np.random.shuffle(data_idx)

        train_idx = data_idx[:int(len(data_idx)*.8)]
        test_idx = data_idx[int(len(data_idx)*.8):]

        train_data = {
            'x': [data['x'][item] for item in train_idx],
            'y': [data['y'][item] for item in train_idx]
        }
        test_data = {
            'x': [data['x'][item] for item in test_idx],
            'y': [data['y'][item] for item in test_idx]
        }
        data = None
        del data
        
        # build clf
        print('Training Classifier... ', att)
        if os.path.exists(clf_dir+att+'_'+str(seed)+'.clf'):
            clf = pickle.load(open(clf_dir+att+'_'+str(seed)+'.clf', 'rb'))
        else:
            if len(np.unique(train_data['y'])) > 2:
                clf = LogisticRegression(
                    class_weight='balanced', solver='lbfgs', n_jobs=-1,
                    multi_class='multinomial', max_iter=2000,
                )
            else:
                clf = LogisticRegression(
                    class_weight='balanced', solver='liblinear'
                )

            clf.fit(train_data['x'], train_data['y'])
            # save the trained clf
            pickle.dump(clf, open(clf_dir+att+'_'+str(seed)+'.clf', 'wb')) 
        
        # evaluate the predictability by accuracy
        results[att] = clf.score(test_data['x'], test_data['y'])

    with open('./predictability/topic/results.json', 'w') as wfile:
        wfile.write(json.dumps(results))

    print('Topic-level, ', results)
    return results


def viz_perform(df, title='default', outpath='./predictability/predictability.pdf'):
    """
    Heatmap visualization of performance improvements.
    :param df: an instance of pandas DataFrame
    :return:
    """
    a4_dims = (16.7, 12.57)
    fig, ax = plt.subplots(figsize=a4_dims)
#    sns.set(font_scale=1.2)
    viz_plot = sns.heatmap(df, annot=True, cbar=False, ax=ax, annot_kws={"size": 36}, cmap="YlGnBu", vmin=df.values.min(), fmt='.1f', square=True)
    plt.xticks(rotation=0, fontsize=32)
    plt.yticks(rotation=0, fontsize=32)
    plt.xlabel('Demographic Factors', fontsize=32)
    plt.ylabel('Features', fontsize=32)
    #plt.title(title, fontsize=25)
    viz_plot.get_figure().savefig(outpath, format='pdf')
    plt.close()


if __name__ == '__main__':
    dpath = './all_data_hash.tsv'
    if os.path.exists('./predictability/results.json'):
        results = json.load(open('./predictability/results.json'))
    else:
        results = {
#            'baseline':{
#                'age': 0.500,
#                'country': 0.500,
#                'ethnicity': 0.500,
#                'gender': 0.500,
#                'region': 0.250,
#            }
        }
        results['pos'] = pos_level(dpath)
        results['topic'] = topic_level(dpath)
        results['word'] = word_level(dpath)

        json.dump(results, open('./predictability/results.json', 'w'))
    
    results = pd.DataFrame(OrderedDict(results))
    viz_perform(results.transpose())
