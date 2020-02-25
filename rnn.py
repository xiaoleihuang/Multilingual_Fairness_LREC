'''Recurrent Neural Network Classifier with GRU unit

'''
import os
import pickle

import numpy as np
from sklearn.metrics import f1_score
from keras.layers import Input, Embedding
from keras.layers import Bidirectional, GRU
from keras.layers import Dense, Dropout
from keras.models import Model
from keras import optimizers

import evaluator


def build_rnn(lang, odir):
    '''Train, valid, test RNN
    
        lang: The language name
        odir: output directory of prediction results
    '''
    doc_idx = 2
    rnn_size = 200
    max_len = 40 # sequence length
    epochs = 10

    encode_dir = './data/encode/'+lang+'/'
    indices_dir = './data/indices/'+lang+'/'
    wt_dir = './resources/weight/'
    res_dir = './resources/classifier/rnn/'
    
    clf_path = res_dir + lang + '.clf'
    # don't reload classifier for debug usage

    # load embedding weights
    weights = np.load(wt_dir+lang+'.npy')

    # build model architecture
    text_input = Input(
        shape=(max_len,), dtype='int32', name='input'
    )
    embeds = Embedding(
        weights.shape[0], weights.shape[1],
        weights=[weights], input_length=max_len,
        trainable=True, name='embedding'
    )(text_input)
    bigru = Bidirectional(GRU(
        rnn_size, kernel_initializer="glorot_uniform")
    )(embeds)
    dp = Dropout(rate=.2)(bigru)
    predicts = Dense(
        1, activation='sigmoid', name='predict'
    )(dp) # binary prediction

    model = Model(inputs=text_input, outputs=predicts)
    model.compile(
        loss='binary_crossentropy', optimizer='rmsprop',
        metrics=['accuracy']
    )
    print(model.summary())

    best_valid_f1 = 0.0
    best_model = None

    for e in range(epochs):
        accuracy = 0.0
        loss = 0.0
        step = 1
        print('--------------Epoch: {}--------------'.format(e))

        # load training and batch dataset
        train_iter = evaluator.data_iter(
            indices_dir+'train.tsv', batch_size=64
        )

        # train model
        for class_wt, x_train, y_train in train_iter:
            if len(np.unique(y_train)) == 1:
                continue
            
            tmp = model.train_on_batch(
                [x_train], y_train,
                class_weight=class_wt
            )

            loss += tmp[0]
            loss_avg = loss / step
            accuracy += tmp[1]
            accuracy_avg = accuracy / step
            if step % 30 == 0:
                print('Step: {}'.format(step))
                print('\tLoss: {}. Accuracy: {}'.format(
                    loss_avg, accuracy_avg)
                )
                print('--------------------------------------')
                step += 1

        # valid model to find the best model
        print('---------------Validation------------')
        valid_iter = evaluator.data_iter(
            indices_dir+'valid.tsv', batch_size=64,
            if_shuffle=False
        )
        y_preds = []
        y_valids = []

        for _, x_valid, y_valid in valid_iter:
            tmp_preds = model.predict([x_valid])
            
            for item_tmp in tmp_preds:
                y_preds.append(round(item_tmp[0]))
            y_valids.extend(y_valid)

        valid_f1 = f1_score(
            y_true=y_valids, y_pred=y_preds, 
            average='weighted',
        )
        print('Validating f1-macro score: ' + str(valid_f1))

        if best_valid_f1 < valid_f1:
            best_valid_f1 = valid_f1
            best_model = model

            pickle.dump(best_model, open(clf_path, 'wb'))

            # test moddel
            print('--------------Test--------------------')
            y_preds = []
            y_probs = []

            test_iter = evaluator.data_iter(
                indices_dir+'test.tsv', batch_size=64,
                if_shuffle=False
            )

            for _, x_test, y_test in test_iter:
                tmp_preds = model.predict([x_test])
                for item_tmp in tmp_preds:
                    y_probs.append(item_tmp[0])
                    y_preds.append(int(round(item_tmp[0])))

            with open(odir+lang+'.tsv', 'w') as wfile:
                with open(indices_dir+'test.tsv') as dfile:
                    wfile.write(
                        dfile.readline().strip()+'\tpred\tpred_prob\n')
                    for idx, line in enumerate(dfile):
                        wfile.write(line.strip()+'\t'+str(y_preds[idx])+'\t'+str(y_probs[idx])+'\n')

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
    odir = './results/rnn/'
    if not os.path.exists(odir):
        os.mkdir(odir)

    for lang in langs:
        build_rnn(lang, odir)

