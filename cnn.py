'''Convolutional Neural Network Classifier with GRU unit
    Kim-2014-version
'''
import os
import pickle

import numpy as np
from sklearn.metrics import f1_score
from keras.layers import Input, Embedding
from keras.layers import Conv1D, MaxPool1D, Flatten
from keras.layers import Dense, Dropout
from keras.models import Model
from keras import optimizers
import keras

import evaluator


def build_cnn(lang, odir):
    '''Train, valid, test CNN
    
        lang: The language name
        odir: output directory of prediction results
    '''
    doc_idx = 2
    max_len = 40 # sequence length
    epochs = 10

    encode_dir = './data/encode/'+lang+'/'
    indices_dir = './data/indices/'+lang+'/'
    wt_dir = './resources/weight/'
    res_dir = './resources/classifier/cnn/'
    
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
    # convolution
    conv3 = Conv1D(
        kernel_size=3, filters=100,
        padding='same', name='conv3'
    )(embeds)
    maxp3 = MaxPool1D()(conv3)
    conv4 = Conv1D(
        kernel_size=4, filters=100,
        padding='same', name='conv4'
    )(embeds)
    maxp4 = MaxPool1D()(conv4)
    conv5 = Conv1D(
        kernel_size=5, filters=100,
        padding='same', name='conv5'
    )(embeds)
    maxp5 = MaxPool1D()(conv5)
    # merge
    merge_convs = keras.layers.concatenate(
        [maxp3, maxp4, maxp5], axis=-1)
    # flatten
    flat_l = Flatten()(merge_convs)
    # dense, according to kim'14 paper, 
    # regularizer applies to the both kernel and bias
    dense_l = Dense(
        100, activation='softplus', name='dense',
        kernel_regularizer=keras.regularizers.l1_l2(0, 0.03),
        bias_regularizer=keras.regularizers.l1_l2(0, 0.03),
    )(flat_l)
    dp_l = Dropout(0.3, name='dropout')(dense_l)
    # predict, binary prediction
    predicts = Dense(
        1, activation='sigmoid', name='predict'
    )(dp_l)
    # model
    model = Model(inputs=text_input, outputs=predicts)
    model.compile(
        loss='binary_crossentropy', optimizer='adam',
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
                tmp_preds = best_model.predict([x_test])
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
    odir = './results/cnn/'
    if not os.path.exists(odir):
        os.mkdir(odir)

    for lang in langs:
        build_cnn(lang, odir)

