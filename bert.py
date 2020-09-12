'''Build Bert document classifier, the code is revised from 
https://colab.research.google.com/drive/1ywsvwO6thOVOrfagjjfuxEf6xVRxbUNO#scrollTo=6J-FYdx6nFE_

'''

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
import torch.nn.functional as F

from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from nltk.tokenize import sent_tokenize

import io
import os
from collections import Counter

import evaluator


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def flat_f1(preds, labels):
    macro_score = f1_score(
        y_true=labels, y_pred=preds,
        average='macro',
    )
    weighted_score = f1_score(
        y_true=labels, y_pred=preds,
        average='weighted',
    )
    print('Weighted F1-score: ', weighted_score)
    print('Macro F1-score: ', macro_score)
    return macro_score, weighted_score


def build_bert(lang, odir, params=None):
    '''Google Bert Classifier
        lang: The language name
        odir: output directory of prediction results
    '''
    if not params:
        params = dict()
        params['balance_ratio'] = 0.9
        params['freeze'] = False
        params['decay_rate'] = .001
        params['lr'] = 2e-5
        params['warm_steps'] = 100
        params['train_steps'] = 1000
        params['batch_size'] = 16
        params['balance'] = True

    split_dir = './data/split/'+lang+'/'

    if torch.cuda.is_available():
        device = str(get_freer_gpu())
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    print(device)

    n_gpu = torch.cuda.device_count()
    print(torch.cuda.get_device_name())
    print('Number of GPUs: ', n_gpu)

    print('Loading Datasets and oversample training data...')
    train_df = pd.read_csv(split_dir+'train.tsv', sep='\t', na_values='x')

    # oversample the minority class
    if params['balance']:
        label_count = Counter(train_df.label)
        for label_tmp in label_count:
            sample_num = label_count.most_common(1)[0][1] - label_count[label_tmp]
            if sample_num == 0:
                continue
            train_df = pd.concat([train_df,
                train_df[train_df.label==label_tmp].sample(
                    int(sample_num*params['balance_ratio']), replace=True
                )])
        train_df = train_df.reset_index() # to prevent index key error

        valid_df = pd.read_csv(split_dir+'valid.tsv', sep='\t', na_values='x')
        test_df = pd.read_csv(split_dir+'test.tsv', sep='\t', na_values='x')
        data_df = [train_df, valid_df, test_df]

    # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
    for doc_df in data_df:
        doc_df.text = doc_df.text.apply(lambda x: '[CLS] '+ x +' [SEP]')

    if lang == 'English':
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', 
            do_lower_case=True
        )
    elif lang == 'Chinese':
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-chinese', 
            do_lower_case=True
        )
    else:
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-multilingual-uncased', 
            do_lower_case=True
        )

    print('Padding Datasets...')
    for doc_df in data_df:
        doc_df.text = doc_df.text.apply(lambda x: tokenizer.tokenize(x))

    # convert to indices and pad the sequences
    max_len = 25
    for doc_df in data_df:
        doc_df.text = doc_df.text.apply(
            lambda x: pad_sequences(
                [tokenizer.convert_tokens_to_ids(x)],
                maxlen=max_len, dtype="long"
                )[0])

    # create attention masks
    for doc_df in data_df:
        attention_masks = []
        for seq in doc_df.text:
            seq_mask = [float(idx>0) for idx in seq]
            attention_masks.append(seq_mask)
        doc_df['masks'] = attention_masks

    # format train, valid, test
    train_inputs = torch.tensor(data_df[0].text)
    train_labels = torch.tensor(data_df[0].label)
    train_masks = torch.tensor(data_df[0].masks)
    valid_inputs = torch.tensor(data_df[1].text)
    valid_labels = torch.tensor(data_df[1].label)
    valid_masks = torch.tensor(data_df[1].masks)
    test_inputs = torch.tensor(data_df[2].text)
    test_labels = torch.tensor(data_df[2].label)
    test_masks = torch.tensor(data_df[2].masks)

    batch_size = params['batch_size']

    train_data = TensorDataset(
        train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size)
    valid_data = TensorDataset(
        valid_inputs, valid_masks, valid_labels)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(
        valid_data, sampler=valid_sampler, batch_size=batch_size)
    test_data = TensorDataset(
        test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=batch_size)

    # load the pretrained model
    print('Loading Pretrained Model...')
    if lang == 'English':
        model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2)
    elif lang == 'Chinese':
        model = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese', num_labels=2)
    else: # for Spanish, Italian, Portuguese and Polish
        model = BertForSequenceClassification.from_pretrained(
        'bert-base-multilingual-uncased', num_labels=2)
    model.to(device)

    # organize parameters
    param_optimizer = list(model.named_parameters())
    if params['freeze']:
        no_decay = ['bias', 'bert'] # , 'bert' freeze all bert parameters
    else:
        no_decay = ['bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': params['decay_rate']},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=params['lr'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=params['warm_steps'], 
        num_training_steps=params['train_steps']
    )

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 10 

    # Training
    print('Training the model...')
    for _ in trange(epochs, desc='Epoch'):
        model.train()
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # train batch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(
                b_input_ids, token_type_ids=None, 
                attention_mask=b_input_mask, labels=b_labels
            )
            # backward pass
    #            outputs[0].backward()
            outputs.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += outputs[0].item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))

        '''Validation'''
        best_valid_f1 = 0.0
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()
        # tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # batch eval
        y_preds = []
        for batch in valid_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(
                    b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask)
            # Move logits and labels to CPU
            logits = outputs[0].detach().cpu().numpy()
            # record the prediction
            pred_flat = np.argmax(logits, axis=1).flatten()
            y_preds.extend(pred_flat)

            label_ids = b_labels.to('cpu').numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

        # evaluate the validation f1 score
        f1_m_valid, f1_w_valid = flat_f1(y_preds, valid_df.label)
        if f1_m_valid > best_valid_f1:
            print('Test....')
            best_valid_f1 = f1_m_valid
            y_preds = []
            y_probs = []

            # test if valid gets better results
            for batch in test_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                    outputs = model(
                        b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
                probs = F.softmax(outputs[0], dim=1)
                probs = probs.detach().cpu().numpy()
                pred_flat = np.argmax(probs, axis=1).flatten()
                y_preds.extend(pred_flat)
                y_probs.extend([item[1] for item in probs])
            # save the predicted results
            with open(odir+lang+'.tsv', 'w') as wfile:
                with open(split_dir+'test.tsv') as dfile:
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
    odir = './results/bert/'
    if not os.path.exists(odir):
        os.mkdir(odir)

    for lang in langs:
        print('Working on: ', lang)
        build_bert(lang, odir)

