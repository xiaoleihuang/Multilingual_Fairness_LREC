import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim

import pickle
import os
from collections import Counter


# tokenize and anonymize one tweet
def token_tweet(tweet):
    '''The data has been tokenized during building process
        No need for more process
    '''
    pass


def anonymize_data(idir, odir):
    '''Anonymize dataset
        
        idir: input directory
        odir: output directory
    '''
    # output anonymized dataset
    opath_utable = odir + 'utable.tsv'
    opath_corpus = odir + 'corpus.tsv'
    hashtable = {'user': dict(), 'corpus': dict()}

    if os.path.exists(opath_utable) and os.path.exists(opath_corpus):
        # return the existing directory
        return odir
    else:
        # create the directory if not exist
        if not os.path.exists(odir):
            os.mkdir(odir)

        # user table
        with open(idir+'utable.tsv') as dfile:
            with open(opath_utable, 'w') as wfile:
                wfile.write(dfile.readline())
                for line in dfile:
                    line = line.strip().split('\t')
                    hash_uid = str(hash(line[0]))
                    hashtable['user'][line[0][:]] = hash_uid
                    line[0] = hash_uid

                    wfile.write('\t'.join(line)+'\n')
    
        # corpus
        with open(idir + 'corpus.tsv') as dfile:
            with open(opath_corpus, 'w') as wfile:
                wfile.write(dfile.readline())
                for line in dfile:
                    line = line.strip().split('\t')
                    hash_did = str(hash(line[0]))
                    hashtable['corpus'][line[0][:]] = hash_did
                    line[0] = hash_did
                    line[1] = hashtable['user'][line[1]]
                    wfile.write('\t'.join(line)+'\n')

        # pickle to save hash table, save to the raw dir
        with open(idir + 'hash_table.tsv', 'wb') as wfile:
            pickle.dump(hashtable, wfile)


def encode_data(idir, odir):
    '''Encode attribute and label of dataset
        
        idir: input directory
        odir: output directory
    '''
    # output encoded utable and corpus
    opath_utable = odir + 'utable.tsv'
    opath_corpus = odir + 'corpus.tsv'

    if os.path.exists(opath_utable) and os.path.exists(opath_corpus):
        # return the existing directory
        return odir
    else:
        # create the directory if not exist
        if not os.path.exists(odir):
            os.mkdir(odir)

        # load utable: know how to binarize attributes
        utable = pd.read_csv(
            idir+'utable.tsv', sep='\t', na_values='x')
        # find the median age
        median = utable.age.median()
        # find the country
        country = Counter(
            utable[utable.country.notnull()].country
        ).most_common(1)[0][0]
        
        # load utable: encode the table
        utable = dict()
        with open(idir+'utable.tsv') as dfile:
            cols = dfile.readline()
            for line in dfile:
                line = line.strip().split('\t')
#                print(len(line))

                # encode gender
                if line[1] != 'x':
                    if line[1] == 'female':
                        line[1] = '1'
                    else:
                        line[1] = '0'

                # encode age
                if line[2] != 'x':
                    if float(line[2]) > median:
                        line[2] = '0'
                    else:
                        line[2] = '1'

                # encode country
                if line[5] != 'x':
                    if line[5] == country:
                        line[5] = '1'
                    else:
                        line[5] = '0'
                
                # encode race
                if line[6] != 'x':
                    if line[6] == 'white':
                        line[6] = '0'
                    else:
                        line[6] = '1'

                utable[line[0]] = line[1:]
                
        # write user table to encoded file
        with open(opath_utable, 'w') as wfile:
            wfile.write(cols)
            for uid in utable:
                wfile.write(uid+'\t'+'\t'.join(utable[uid])+'\n')

        # load the corpus
        with open(idir+'corpus.tsv') as dfile:
            with open(opath_corpus, 'w') as wfile:
                wfile.write(dfile.readline())
                for line in dfile:
                    line = line.strip().split('\t')
                    # attributes
                    line[4:-1] = utable[line[1]]
                    # labels
                    if line[-1] in ['0', 'no', 'neither', 'normal']:
                        line[-1] = '0'
                    else:
                        line[-1] = '1'
                    wfile.write('\t'.join(line)+'\n')


# build tokenizer
def build_tok(idir, odir, name):
    doc_idx = 2 # column of document
    
    if not os.path.exists(odir):
        os.mkdir(odir)
    
    opath = odir + name + '.tkn'
    if os.path.exists(opath):
        return pickle.load(open(opath, 'rb'))
    else:
        # load corpus
        corpus = []
        tkn = Tokenizer(num_words=15000)

        with open(idir+'corpus.tsv') as dfile:
            dfile.readline() # skip column names
            for line in dfile:
                corpus.append(
                    line.strip().split('\t')[doc_idx]
                )
        tkn.fit_on_texts(corpus)

        with open(opath, 'wb') as wfile:
            pickle.dump(tkn, wfile)
        return tkn


# split data
def split_data(idir, odir):
    '''Split dataset into 
        70/15/15: train/valid/test
        
        idir: input directory
        odir: output directory
    '''
    datap = idir + 'corpus.tsv'

    if not os.path.exists(odir):
        os.mkdir(odir)

    trainp = odir + 'train.tsv'
    validp = odir + 'valid.tsv'
    testp = odir + 'test.tsv'
    writers = [
        open(trainp, 'w'),
        open(validp, 'w'),
        open(testp, 'w'),
    ]

    # load data
    corpus = []
    with open(datap) as dfile:
        cols = dfile.readline()
        for line in dfile:
            corpus.append(line)
            
    indices = list(range(len(corpus)))
    np.random.shuffle(indices)
    
    # train
    with open(trainp, 'w') as wfile:
        wfile.write(cols)
        for idx in range(int(len(indices)*.7)):
            wfile.write(corpus[idx])

    # valid
    with open(validp, 'w') as wfile:
        wfile.write(cols)
        for idx in range(int(len(indices)*.7), int(len(indices)*.85)):
            wfile.write(corpus[idx])

    # test
    with open(testp, 'w') as wfile:
        wfile.write(cols)
        for idx in range(int(len(indices)*.85), len(indices)):
            wfile.write(corpus[idx])


def build_indices(split_dir, tok, indices_dir, max_len=40):
    '''Convert the raw text into indices
    '''
    files = [
        'train.tsv', 'valid.tsv', 'test.tsv'
    ]
    doc_idx = 2
    if not os.path.exists(indices_dir):
        os.mkdir(indices_dir)
    
    for filen in files:
        filep = split_dir + filen # input
        wfile = open(indices_dir + filen, 'w') # output

        with open(filep) as dfile:
            wfile.write(dfile.readline())
            for line in dfile:
                line = line.strip().split('\t')
                line[doc_idx] = ' '.join(
                    map(str, 
                        pad_sequences(
                            tok.texts_to_sequences([line[doc_idx]]),
                            maxlen=max_len
                        )[0]
                    )
                )
                wfile.write('\t'.join(line)+'\n')
        wfile.flush()
        wfile.close()


def build_wt(lang, tkn, emb_path, opt):
    '''Create embedding weights
        
        lang: The langugage name
        tkn: tokenizer
        emb_path: embedding file path
        opt: output path of weight file
    '''
    embed_len = len(tkn.word_index)
    if embed_len > tkn.num_words:
        embed_len = tkn.num_words
    
    # obtain the size of embedding
    if emb_path.endswith('.bin'):
        embeds = gensim.models.KeyedVectors.load_word2vec_format(
            emb_path, binary=True, unicode_errors='ignore'
        )
        for pair in zip(embeds.wv.index2word, embeds.wv.syn0):
            size = len([float(item) for item in pair[1]])
            break
    else:
        with open(emb_path) as dfile:
            for line in dfile:
                line = line.strip().split()
                if line[0] in tkn.word_index:
                    size = len([float(item) for item in line[1:]])
                    break

    emb_matrix = np.zeros((embed_len + 1, size))
    if emb_path.endswith('.bin'):
        embeds = gensim.models.KeyedVectors.load_word2vec_format(
            emb_path, binary=True, unicode_errors='ignore'
        )
        for pair in zip(embeds.wv.index2word, embeds.wv.syn0):
            if pair[0] in tkn.word_index and \
                tkn.word_index[pair[0]] < tkn.num_words:
                emb_matrix[tkn.word_index[pair[0]]] = [
                    float(item) for item in pair[1]
                ]
    else:
        with open(emb_path) as dfile:
            for line in dfile:
                line = line.strip().split()
                if line[0] in tkn.word_index and \
                    tkn.word_index[line[0]] < tkn.num_words:
                    emb_matrix[tkn.word_index[line[0]]] = [
                        float(item) for item in line[1:]
                    ]
    np.save(opt, emb_matrix)


if __name__ == '__main__':
    dir_list = [
        'English', 'Italian', 'Polish',
        'Portuguese', 'Spanish',
    ]
    
    for dirp in dir_list:
        raw_dir = './data/raw/'+dirp+'/'
        anonymize_dir = './data/anonymize/'+dirp+'/'
        encode_dir = './data/encode/'+dirp+'/'
        split_dir = './data/split/'+dirp+'/'
        indices_dir = './data/indices/'+dirp+'/'
        tok_dir = './resources/tokenizer/'
        emb_dir = './resources/embedding/'
        wt_dir = './resources/weight/'

        # anonymize data, this is only for author usage
#        print('anonymize data: ', dirp)
#        anonymize_data(
#            raw_dir, 
#            anonymize_dir
#        )

        # encode data
        print('encode data: ', dirp)
        encode_data(
            anonymize_dir, 
            encode_dir
        )

        # build tokenizer
        print('build tokenizer: ', dirp)
        tok = build_tok(encode_dir, tok_dir, dirp)

        # split data
        print('split data: ', dirp)
        split_data(
            encode_dir,
            split_dir
        )

        # encode into indices
        print('data to indices: ', dirp)
        build_indices(
            split_dir, tok, indices_dir
        )

        # build weights
        opt = wt_dir + dirp
        emb_path = emb_dir + dirp

        if os.path.exists(emb_path + '.vec'):
            build_wt(dirp, tok, emb_path + '.vec', opt)
        else:
            build_wt(dirp, tok, emb_path + '.bin', opt)
