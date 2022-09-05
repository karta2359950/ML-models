# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="AFEKWoh3p1Mv"
# # Homework Description
# - English to Chinese (Traditional) Translation
#   - Input: an English sentence         (e.g.		tom is a student .)
#   - Output: the Chinese translation  (e.g. 		湯姆 是 個 學生 。)
#
# - TODO
#     - Train a simple RNN seq2seq to acheive translation
#     - Switch to transformer model to boost performance
#     - Apply Back-translation to furthur boost performance

# + id="3Vf1Q79XPQ3D" outputId="081ab89d-7825-4d1c-e789-2e0e4c9aba7e" colab={"base_uri": "https://localhost:8080/"}
# !nvidia-smi

# + [markdown] id="59neB_Sxp5Ub"
# # Download and import required packages

# + id="rRlFbfFRpZYT" outputId="39e5a2bf-0d41-45bf-b9db-409379dea083" colab={"base_uri": "https://localhost:8080/"}
# !pip install 'torch>=1.6.0' editdistance matplotlib sacrebleu sacremoses sentencepiece tqdm wandb
# !pip install --upgrade jupyter ipywidgets

# + id="fSksMTdmp-Wt" outputId="08a9039e-b3da-4d8e-e04c-475f2c2dae0b" colab={"base_uri": "https://localhost:8080/"}
# !git clone https://github.com/pytorch/fairseq.git
# !cd fairseq && git checkout 9a1c497
# !pip install --upgrade ./fairseq/

# + id="uRLTiuIuqGNc"
import sys
import pdb
import pprint
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import tqdm.auto as tqdm
from pathlib import Path
from argparse import Namespace
from fairseq import utils

import matplotlib.pyplot as plt

# + [markdown] id="0n07Za1XqJzA"
# # Fix random seed

# + id="xllxxyWxqI7s"
norm_list = []
seed = 73
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
np.random.seed(seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# + [markdown] id="N5ORDJ-2qdYw"
# # Dataset
#
# ## En-Zh Bilingual Parallel Corpus
# * [TED2020](#reimers-2020-multilingual-sentence-bert)
#     - Raw: 398,066 (sentences)   
#     - Processed: 393,980 (sentences)
#     
#
# ## Testdata
# - Size: 4,000 (sentences)
# - **Chinese translation is undisclosed. The provided (.zh) file is psuedo translation, each line is a '。'**

# + [markdown] id="GQw2mY4Dqkzd"
# ## Dataset Download

# + id="SXT42xQtqijD" outputId="4bf383c0-825b-4b55-8c16-a4d7cbc7bf87" colab={"base_uri": "https://localhost:8080/"}
data_dir = './DATA/rawdata'
dataset_name = 'ted2020'
urls = (
    "https://github.com/yuhsinchan/ML2022-HW5Dataset/releases/download/v1.0.2/ted2020.tgz",
    "https://github.com/yuhsinchan/ML2022-HW5Dataset/releases/download/v1.0.2/test.tgz",
)
file_names = (
    'ted2020.tgz', # train & dev
    'test.tgz', # test
)
prefix = Path(data_dir).absolute() / dataset_name

prefix.mkdir(parents=True, exist_ok=True)
for u, f in zip(urls, file_names):
    path = prefix/f
    if not path.exists():
        # !wget {u} -O {path}
    if path.suffix == ".tgz":
        # !tar -xvf {path} -C {prefix}
    elif path.suffix == ".zip":
        # !unzip -o {path} -d {prefix}
# !mv {prefix/'raw.en'} {prefix/'train_dev.raw.en'}
# !mv {prefix/'raw.zh'} {prefix/'train_dev.raw.zh'}
# !mv {prefix/'test/test.en'} {prefix/'test.raw.en'}
# !mv {prefix/'test/test.zh'} {prefix/'test.raw.zh'}
# !rm -rf {prefix/'test'}

# + [markdown] id="YLkJwNiFrIwZ"
# ## Language

# + id="_uJYkCncrKJb"
src_lang = 'en'
tgt_lang = 'zh'

data_prefix = f'{prefix}/train_dev.raw'
test_prefix = f'{prefix}/test.raw'

# + id="0t2CPt1brOT3" outputId="bed983df-f1be-4be5-c322-c7bc2a675947" colab={"base_uri": "https://localhost:8080/"}
# !head {data_prefix+'.'+src_lang} -n 5
# !head {data_prefix+'.'+tgt_lang} -n 5

# + [markdown] id="pRoE9UK7r1gY"
# ## Preprocess files

# + id="3tzFwtnFrle3"
import re

def strQ2B(ustring):
    """Full width -> half width"""
    # reference:https://ithelp.ithome.com.tw/articles/10233122
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # Full width space: direct conversion
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # Full width chars (except space) conversion
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)
                
def clean_s(s, lang):
    if lang == 'en':
        s = re.sub(r"\([^()]*\)", "", s) # remove ([text])
        s = s.replace('-', '') # remove '-'
        s = re.sub('([.,;!?()\"])', r' \1 ', s) # keep punctuation
    elif lang == 'zh':
        s = strQ2B(s) # Q2B
        s = re.sub(r"\([^()]*\)", "", s) # remove ([text])
        s = s.replace(' ', '')
        s = s.replace('—', '')
        s = s.replace('“', '"')
        s = s.replace('”', '"')
        s = s.replace('_', '')
        s = re.sub('([。,;!?()\"~「」])', r' \1 ', s) # keep punctuation
    s = ' '.join(s.strip().split())
    return s

def len_s(s, lang):
    if lang == 'zh':
        return len(s)
    return len(s.split())

def clean_corpus(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):
    if Path(f'{prefix}.clean.{l1}').exists() and Path(f'{prefix}.clean.{l2}').exists():
        print(f'{prefix}.clean.{l1} & {l2} exists. skipping clean.')
        return
    with open(f'{prefix}.{l1}', 'r') as l1_in_f:
        with open(f'{prefix}.{l2}', 'r') as l2_in_f:
            with open(f'{prefix}.clean.{l1}', 'w') as l1_out_f:
                with open(f'{prefix}.clean.{l2}', 'w') as l2_out_f:
                    for s1 in l1_in_f:
                        s1 = s1.strip()
                        s2 = l2_in_f.readline().strip()
                        s1 = clean_s(s1, l1)
                        s2 = clean_s(s2, l2)
                        s1_len = len_s(s1, l1)
                        s2_len = len_s(s2, l2)
                        if min_len > 0: # remove short sentence
                            if s1_len < min_len or s2_len < min_len:
                                continue
                        if max_len > 0: # remove long sentence
                            if s1_len > max_len or s2_len > max_len:
                                continue
                        if ratio > 0: # remove by ratio of length
                            if s1_len/s2_len > ratio or s2_len/s1_len > ratio:
                                continue
                        print(s1, file=l1_out_f)
                        print(s2, file=l2_out_f)


# + id="h_i8b1PRr9Nf"
clean_corpus(data_prefix, src_lang, tgt_lang)
clean_corpus(test_prefix, src_lang, tgt_lang, ratio=-1, min_len=-1, max_len=-1)

# + id="gjT3XCy9r_rj" outputId="860b9f86-62d8-4c4b-dd9b-f74250547408" colab={"base_uri": "https://localhost:8080/"}
# !head {data_prefix+'.clean.'+src_lang} -n 5
# !head {data_prefix+'.clean.'+tgt_lang} -n 5

# + [markdown] id="nKb4u67-sT_Z"
# ## Split into train/valid

# + id="AuFKeDz3sGHL"
valid_ratio = 0.01 # 3000~4000 would suffice
train_ratio = 1 - valid_ratio

# + id="QR2NVldqsXyY"
if (prefix/f'train.clean.{src_lang}').exists() \
and (prefix/f'train.clean.{tgt_lang}').exists() \
and (prefix/f'valid.clean.{src_lang}').exists() \
and (prefix/f'valid.clean.{tgt_lang}').exists():
    print(f'train/valid splits exists. skipping split.')
else:
    line_num = sum(1 for line in open(f'{data_prefix}.clean.{src_lang}'))
    labels = list(range(line_num))
    random.shuffle(labels)
    for lang in [src_lang, tgt_lang]:
        train_f = open(os.path.join(data_dir, dataset_name, f'train.clean.{lang}'), 'w')
        valid_f = open(os.path.join(data_dir, dataset_name, f'valid.clean.{lang}'), 'w')
        count = 0
        for line in open(f'{data_prefix}.clean.{lang}', 'r'):
            if labels[count]/line_num < train_ratio:
                train_f.write(line)
            else:
                valid_f.write(line)
            count += 1
        train_f.close()
        valid_f.close()

# + [markdown] id="n1rwQysTsdJq"
# ## Subword Units 
# Out of vocabulary (OOV) has been a major problem in machine translation. This can be alleviated by using subword units.
# - We will use the [sentencepiece](#kudo-richardson-2018-sentencepiece) package
# - select 'unigram' or 'byte-pair encoding (BPE)' algorithm

# + id="Ecwllsa7sZRA"
import sentencepiece as spm
vocab_size = 8000
if (prefix/f'spm{vocab_size}.model').exists():
    print(f'{prefix}/spm{vocab_size}.model exists. skipping spm_train.')
else:
    spm.SentencePieceTrainer.train(
        input=','.join([f'{prefix}/train.clean.{src_lang}',
                        f'{prefix}/valid.clean.{src_lang}',
                        f'{prefix}/train.clean.{tgt_lang}',
                        f'{prefix}/valid.clean.{tgt_lang}']),
        model_prefix=prefix/f'spm{vocab_size}',
        vocab_size=vocab_size,
        character_coverage=1,
        model_type='unigram', # 'bpe' works as well
        input_sentence_size=1e6,
        shuffle_input_sentence=True,
        normalization_rule_name='nmt_nfkc_cf',
    )

# + id="lQPRNldqse_V"
spm_model = spm.SentencePieceProcessor(model_file=str(prefix/f'spm{vocab_size}.model'))
in_tag = {
    'train': 'train.clean',
    'valid': 'valid.clean',
    'test': 'test.raw.clean',
}
for split in ['train', 'valid', 'test']:
    for lang in [src_lang, tgt_lang]:
        out_path = prefix/f'{split}.{lang}'
        if out_path.exists():
            print(f"{out_path} exists. skipping spm_encode.")
        else:
            with open(prefix/f'{split}.{lang}', 'w') as out_f:
                with open(prefix/f'{in_tag[split]}.{lang}', 'r') as in_f:
                    for line in in_f:
                        line = line.strip()
                        tok = spm_model.encode(line, out_type=str)
                        print(' '.join(tok), file=out_f)

# + id="4j6lXHjAsjXa" outputId="a39fa7d0-e390-4b3c-de97-5f569023e3d5" colab={"base_uri": "https://localhost:8080/"}
# !head {data_dir+'/'+dataset_name+'/train.'+src_lang} -n 5
# !head {data_dir+'/'+dataset_name+'/train.'+tgt_lang} -n 5

# + [markdown] id="59si_C0Wsms7"
# ## Binarize the data with fairseq

# + id="w-cHVLSpsknh" outputId="96dadc5b-5fd3-40c6-83d5-d1db2f1fa2d6" colab={"base_uri": "https://localhost:8080/"}
binpath = Path('./DATA/data-bin', dataset_name)
if binpath.exists():
    print(binpath, "exists, will not overwrite!")
else:
    # !python -m fairseq_cli.preprocess \
#         --source-lang {src_lang}\
#         --target-lang {tgt_lang}\
#         --trainpref {prefix/'train'}\
#         --validpref {prefix/'valid'}\
#         --testpref {prefix/'test'}\
#         --destdir {binpath}\
#         --joined-dictionary\
#         --workers 2

# + [markdown] id="szMuH1SWLPWA"
# # Configuration for experiments

# + id="5Luz3_tVLUxs"
config = Namespace(
    datadir = "./DATA/data-bin/ted2020",
    savedir = "./checkpoints/transformer",
    source_lang = "en",
    target_lang = "zh",
    
    # cpu threads when fetching & processing data.
    num_workers=2,  
    # batch size in terms of tokens. gradient accumulation increases the effective batchsize.
    max_tokens=8192,
    accum_steps=2,
    
    # the lr s calculated from Noam lr scheduler. you can tune the maximum lr by this factor.
    lr_factor=2.,
    lr_warmup=4000,
    
    # clipping gradient norm helps alleviate gradient exploding
    clip_norm=1.0,
    
    # maximum epochs for training
    max_epoch=30,
    start_epoch=1,
    
    # beam size for beam search
    beam=5, 
    # generate sequences of maximum length ax + b, where x is the source length
    max_len_a=1.2, 
    max_len_b=10, 
    # when decoding, post process sentence by removing sentencepiece symbols and jieba tokenization.
    post_process = "sentencepiece",
    
    # checkpoints
    keep_last_epochs=5,
    resume=None, # if resume from checkpoint name (under config.savedir)
    
    # logging
    use_wandb=False,
)

# + [markdown] id="cjrJFvyQLg86"
# # Logging
# - logging package logs ordinary messages
# - wandb logs the loss, bleu, etc. in the training process

# + id="-ZiMyDWALbDk"
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO", # "DEBUG" "WARNING" "ERROR"
    stream=sys.stdout,
)
proj = "hw5.seq2seq"
logger = logging.getLogger(proj)
if config.use_wandb:
    import wandb
    wandb.init(project=proj, name=Path(config.savedir).stem, config=config)

# + [markdown] id="BNoSkK45Lmqc"
# # CUDA Environments

# + id="oqrsbmcoLqMl" outputId="e51ad5f9-b4cf-4a57-c620-b69ec8a2c19f" colab={"base_uri": "https://localhost:8080/"}
cuda_env = utils.CudaEnvironment()
utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# + [markdown] id="TbJuBIHLLt2D"
# # Dataloading

# + [markdown] id="oOpG4EBRLwe_"
# ## We borrow the TranslationTask from fairseq
# * used to load the binarized data created above
# * well-implemented data iterator (dataloader)
# * built-in task.source_dictionary and task.target_dictionary are also handy
# * well-implemented beach search decoder

# + id="3gSEy1uFLvVs" outputId="9aeba378-5f4f-47ff-d892-eb89740e9528" colab={"base_uri": "https://localhost:8080/"}
from fairseq.tasks.translation import TranslationConfig, TranslationTask

## setup task
task_cfg = TranslationConfig(
    data=config.datadir,
    source_lang=config.source_lang,
    target_lang=config.target_lang,
    train_subset="train",
    required_seq_len_multiple=8,
    dataset_impl="mmap",
    upsample_primary=1,
)
task = TranslationTask.setup_task(task_cfg)

# + id="mR7Bhov7L4IU" outputId="940c1974-e4ea-4f07-9f42-747675275fcf" colab={"base_uri": "https://localhost:8080/"}
logger.info("loading data for epoch 1")
task.load_dataset(split="train", epoch=1, combine=True) # combine if you have back-translation data.
task.load_dataset(split="valid", epoch=1)

# + id="P0BCEm_9L6ig" outputId="f5f91e3f-1b70-4196-9b32-838225ac9a1c" colab={"base_uri": "https://localhost:8080/"}
sample = task.dataset("valid")[1]
pprint.pprint(sample)
pprint.pprint(
    "Source: " + \
    task.source_dictionary.string(
        sample['source'],
        config.post_process,
    )
)
pprint.pprint(
    "Target: " + \
    task.target_dictionary.string(
        sample['target'],
        config.post_process,
    )
)


# + [markdown] id="UcfCVa2FMBSE"
# # Dataset iterator

# + [markdown] id="yBvc-B_6MKZM"
# * Controls every batch to contain no more than N tokens, which optimizes GPU memory efficiency
# * Shuffles the training set for every epoch
# * Ignore sentences exceeding maximum length
# * Pad all sentences in a batch to the same length, which enables parallel computing by GPU
# * Add eos and shift one token
#     - teacher forcing: to train the model to predict the next token based on prefix, we feed the right shifted target sequence as the decoder input.
#     - generally, prepending bos to the target would do the job (as shown below)
# ![seq2seq](https://i.imgur.com/0zeDyuI.png)
#     - in fairseq however, this is done by moving the eos token to the begining. Empirically, this has the same effect. For instance:
#     ```
#     # output target (target) and Decoder input (prev_output_tokens): 
#                    eos = 2
#                 target = 419,  711,  238,  888,  792,   60,  968,    8,    2
#     prev_output_tokens = 2,  419,  711,  238,  888,  792,   60,  968,    8
#     ```
#
#

# + id="OWFJFmCnMDXW" outputId="a9e93298-0366-4dc0-a2b4-a585df27d513" colab={"base_uri": "https://localhost:8080/"}
def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            max_tokens,
        ),
        ignore_invalid_inputs=True,
        seed=seed,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=not cached,
        # Set this to False to speed up. However, if set to False, changing max_tokens beyond 
        # first call of this method has no effect. 
    )
    return batch_iterator

demo_epoch_obj = load_data_iterator(task, "valid", epoch=1, max_tokens=20, num_workers=1, cached=False)
demo_iter = demo_epoch_obj.next_epoch_itr(shuffle=True)
sample = next(demo_iter)
sample

# + [markdown] id="p86K-0g7Me4M"
# * each batch is a python dict, with string key and Tensor value. Contents are described below:
# ```python
# batch = {
#     "id": id, # id for each example 
#     "nsentences": len(samples), # batch size (sentences)
#     "ntokens": ntokens, # batch size (tokens)
#     "net_input": {
#         "src_tokens": src_tokens, # sequence in source language
#         "src_lengths": src_lengths, # sequence length of each example before padding
#         "prev_output_tokens": prev_output_tokens, # right shifted target, as mentioned above.
#     },
#     "target": target, # target sequence
# }
# ```

# + [markdown] id="9EyDBE5ZMkFZ"
# # Model Architecture
# * We again inherit fairseq's encoder, decoder and model, so that in the testing phase we can directly leverage fairseq's beam search decoder.

# + id="Hzh74qLIMfW_"
from fairseq.models import (
    FairseqEncoder, 
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel
)


# + [markdown] id="OI46v1z7MotH"
# # Encoder

# + [markdown] id="Wn0wSeLLMrbc"
# - The Encoder is a RNN or Transformer Encoder. The following description is for RNN. For every input token, Encoder will generate a output vector and a hidden states vector, and the hidden states vector is passed on to the next step. In other words, the Encoder sequentially reads in the input sequence, and outputs a single vector at each timestep, then finally outputs the final hidden states, or content vector, at the last timestep.
# - Parameters:
#   - *args*
#       - encoder_embed_dim: the dimension of embeddings, this compresses the one-hot vector into fixed dimensions, which achieves dimension reduction
#       - encoder_ffn_embed_dim is the dimension of hidden states and output vectors
#       - encoder_layers is the number of layers for Encoder RNN
#       - dropout determines the probability of a neuron's activation being set to 0, in order to prevent overfitting. Generally this is applied in training, and removed in testing.
#   - *dictionary*: the dictionary provided by fairseq. it's used to obtain the padding index, and in turn the encoder padding mask. 
#   - *embed_tokens*: an instance of token embeddings (nn.Embedding)
#
# - Inputs: 
#     - *src_tokens*: integer sequence representing english e.g. 1, 28, 29, 205, 2 
# - Outputs: 
#     - *outputs*: the output of RNN at each timestep, can be furthur processed by Attention
#     - *final_hiddens*: the hidden states of each timestep, will be passed to decoder for decoding
#     - *encoder_padding_mask*: this tells the decoder which position to ignore
#

# + id="WcX3W4iGMq-S"
'''class RNNEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.embed_tokens = embed_tokens
        
        self.embed_dim = args.encoder_embed_dim
        self.hidden_dim = args.encoder_ffn_embed_dim
        self.num_layers = args.encoder_layers
        
        self.dropout_in_module = nn.Dropout(args.dropout)
        self.rnn = nn.GRU(
            self.embed_dim, 
            self.hidden_dim, 
            self.num_layers, 
            dropout=args.dropout, 
            batch_first=False, 
            bidirectional=True
        )
        self.dropout_out_module = nn.Dropout(args.dropout)
        
        self.padding_idx = dictionary.pad()
        
    def combine_bidir(self, outs, bsz: int):
        out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
        return out.view(self.num_layers, bsz, -1)

    def forward(self, src_tokens, **unused):
        bsz, seqlen = src_tokens.size()
        
        # get embeddings
        x = self.embed_tokens(src_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        
        # pass thru bidirectional RNN
        h0 = x.new_zeros(2 * self.num_layers, bsz, self.hidden_dim)
        x, final_hiddens = self.rnn(x, h0)
        outputs = self.dropout_out_module(x)
        # outputs = [sequence len, batch size, hid dim * directions]
        # hidden =  [num_layers * directions, batch size  , hid dim]
        
        # Since Encoder is bidirectional, we need to concatenate the hidden states of two directions
        final_hiddens = self.combine_bidir(final_hiddens, bsz)
        # hidden =  [num_layers x batch x num_directions*hidden]
        
        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()
        return tuple(
            (
                outputs,  # seq_len x batch x hidden
                final_hiddens,  # num_layers x batch x num_directions*hidden
                encoder_padding_mask,  # seq_len x batch
            )
        )
    
    def reorder_encoder_out(self, encoder_out, new_order):
        # This is used by fairseq's beam search. How and why is not particularly important here.
        return tuple(
            (
                encoder_out[0].index_select(1, new_order),
                encoder_out[1].index_select(1, new_order),
                encoder_out[2].index_select(1, new_order),
            )
        )
'''

# + [markdown] id="6ZlE_1JnMv56"
# ## Attention

# + [markdown] id="ZSFSKt_ZMzgh"
# - When the input sequence is long, "content vector" alone cannot accurately represent the whole sequence, attention mechanism can provide the Decoder more information.
# - According to the **Decoder embeddings** of the current timestep, match the **Encoder outputs** with decoder embeddings to determine correlation, and then sum the Encoder outputs weighted by the correlation as the input to **Decoder** RNN.
# - Common attention implementations use neural network / dot product as the correlation between **query** (decoder embeddings) and **key** (Encoder outputs), followed by **softmax**  to obtain a distribution, and finally **values** (Encoder outputs) is **weighted sum**-ed by said distribution.
#
# - Parameters:
#   - *input_embed_dim*: dimensionality of key, should be that of the vector in decoder to attend others
#   - *source_embed_dim*: dimensionality of query, should be that of the vector to be attended to (encoder outputs)
#   - *output_embed_dim*: dimensionality of value, should be that of the vector after attention, expected by the next layer
#
# - Inputs: 
#     - *inputs*: is the key, the vector to attend to others
#     - *encoder_outputs*:  is the query/value, the vector to be attended to
#     - *encoder_padding_mask*: this tells the decoder which position to ignore
# - Outputs: 
#     - *output*: the context vector after attention
#     - *attention score*: the attention distribution
#

# + id="1Atf_YuCMyyF"
'''class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = nn.Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = nn.Linear(
            input_embed_dim + source_embed_dim, output_embed_dim, bias=bias
        )

    def forward(self, inputs, encoder_outputs, encoder_padding_mask):
        # inputs: T, B, dim
        # encoder_outputs: S x B x dim
        # padding mask:  S x B
        
        # convert all to batch first
        inputs = inputs.transpose(1,0) # B, T, dim
        encoder_outputs = encoder_outputs.transpose(1,0) # B, S, dim
        encoder_padding_mask = encoder_padding_mask.transpose(1,0) # B, S
        
        # project to the dimensionality of encoder_outputs
        x = self.input_proj(inputs)

        # compute attention
        # (B, T, dim) x (B, dim, S) = (B, T, S)
        attn_scores = torch.bmm(x, encoder_outputs.transpose(1,2))

        # cancel the attention at positions corresponding to padding
        if encoder_padding_mask is not None:
            # leveraging broadcast  B, S -> (B, 1, S)
            encoder_padding_mask = encoder_padding_mask.unsqueeze(1)
            attn_scores = (
                attn_scores.float()
                .masked_fill_(encoder_padding_mask, float("-inf"))
                .type_as(attn_scores)
            )  # FP16 support: cast to float and back

        # softmax on the dimension corresponding to source sequence
        attn_scores = F.softmax(attn_scores, dim=-1)

        # shape (B, T, S) x (B, S, dim) = (B, T, dim) weighted sum
        x = torch.bmm(attn_scores, encoder_outputs)

        # (B, T, dim)
        x = torch.cat((x, inputs), dim=-1)
        x = torch.tanh(self.output_proj(x)) # concat + linear + tanh
        
        # restore shape (B, T, dim) -> (T, B, dim)
        return x.transpose(1,0), attn_scores
'''

# + [markdown] id="doSCOA2gM7fK"
# # Decoder

# + [markdown] id="2M8Vod2gNABR"
# * The hidden states of **Decoder** will be initialized by the final hidden states of **Encoder** (the content vector)
# * At the same time, **Decoder** will change its hidden states based on the input of the current timestep (the outputs of previous timesteps), and generates an output
# * Attention improves the performance
# * The seq2seq steps are implemented in decoder, so that later the Seq2Seq class can accept RNN and Transformer, without furthur modification.
# - Parameters:
#   - *args*
#       - decoder_embed_dim: is the dimensionality of the decoder embeddings, similar to encoder_embed_dim，
#       - decoder_ffn_embed_dim: is the dimensionality of the decoder RNN hidden states, similar to encoder_ffn_embed_dim
#       - decoder_layers: number of layers of RNN decoder
#       - share_decoder_input_output_embed: usually, the projection matrix of the decoder will share weights with the decoder input embeddings
#   - *dictionary*: the dictionary provided by fairseq
#   - *embed_tokens*: an instance of token embeddings (nn.Embedding)
# - Inputs: 
#     - *prev_output_tokens*: integer sequence representing the right-shifted target e.g. 1, 28, 29, 205, 2 
#     - *encoder_out*: encoder's output.
#     - *incremental_state*: in order to speed up decoding during test time, we will save the hidden state of each timestep. see forward() for details.
# - Outputs: 
#     - *outputs*: the logits (before softmax) output of decoder for each timesteps
#     - *extra*: unsused

# + id="QfvgqHYDM6Lp"
'''class RNNDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.embed_tokens = embed_tokens
        
        assert args.decoder_layers == args.encoder_layers, f"""seq2seq rnn requires that encoder 
        and decoder have same layers of rnn. got: {args.encoder_layers, args.decoder_layers}"""
        assert args.decoder_ffn_embed_dim == args.encoder_ffn_embed_dim*2, f"""seq2seq-rnn requires 
        that decoder hidden to be 2*encoder hidden dim. got: {args.decoder_ffn_embed_dim, args.encoder_ffn_embed_dim*2}"""
        
        self.embed_dim = args.decoder_embed_dim
        self.hidden_dim = args.decoder_ffn_embed_dim
        self.num_layers = args.decoder_layers
        
        
        self.dropout_in_module = nn.Dropout(args.dropout)
        self.rnn = nn.GRU(
            self.embed_dim, 
            self.hidden_dim, 
            self.num_layers, 
            dropout=args.dropout, 
            batch_first=False, 
            bidirectional=False
        )
        self.attention = AttentionLayer(
            self.embed_dim, self.hidden_dim, self.embed_dim, bias=False
        ) 
        # self.attention = None
        self.dropout_out_module = nn.Dropout(args.dropout)
        
        if self.hidden_dim != self.embed_dim:
            self.project_out_dim = nn.Linear(self.hidden_dim, self.embed_dim)
        else:
            self.project_out_dim = None
        
        if args.share_decoder_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
        
    def forward(self, prev_output_tokens, encoder_out, incremental_state=None, **unused):
        # extract the outputs from encoder
        encoder_outputs, encoder_hiddens, encoder_padding_mask = encoder_out
        # outputs:          seq_len x batch x num_directions*hidden
        # encoder_hiddens:  num_layers x batch x num_directions*encoder_hidden
        # padding_mask:     seq_len x batch
        
        if incremental_state is not None and len(incremental_state) > 0:
            # if the information from last timestep is retained, we can continue from there instead of starting from bos
            prev_output_tokens = prev_output_tokens[:, -1:]
            cache_state = self.get_incremental_state(incremental_state, "cached_state")
            prev_hiddens = cache_state["prev_hiddens"]
        else:
            # incremental state does not exist, either this is training time, or the first timestep of test time
            # prepare for seq2seq: pass the encoder_hidden to the decoder hidden states
            prev_hiddens = encoder_hiddens
        
        bsz, seqlen = prev_output_tokens.size()
        
        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
                
        # decoder-to-encoder attention
        if self.attention is not None:
            x, attn = self.attention(x, encoder_outputs, encoder_padding_mask)
                        
        # pass thru unidirectional RNN
        x, final_hiddens = self.rnn(x, prev_hiddens)
        # outputs = [sequence len, batch size, hid dim]
        # hidden =  [num_layers * directions, batch size  , hid dim]
        x = self.dropout_out_module(x)
                
        # project to embedding size (if hidden differs from embed size, and share_embedding is True, 
        # we need to do an extra projection)
        if self.project_out_dim != None:
            x = self.project_out_dim(x)
        
        # project to vocab size
        x = self.output_projection(x)
        
        # T x B x C -> B x T x C
        x = x.transpose(1, 0)
        
        # if incremental, record the hidden states of current timestep, which will be restored in the next timestep
        cache_state = {
            "prev_hiddens": final_hiddens,
        }
        self.set_incremental_state(incremental_state, "cached_state", cache_state)
        
        return x, None
    
    def reorder_incremental_state(
        self,
        incremental_state,
        new_order,
    ):
        # This is used by fairseq's beam search. How and why is not particularly important here.
        cache_state = self.get_incremental_state(incremental_state, "cached_state")
        prev_hiddens = cache_state["prev_hiddens"]
        prev_hiddens = [p.index_select(0, new_order) for p in prev_hiddens]
        cache_state = {
            "prev_hiddens": torch.stack(prev_hiddens),
        }
        self.set_incremental_state(incremental_state, "cached_state", cache_state)
        return
'''

# + [markdown] id="UDAPmxjRNEEL"
# ## Seq2Seq
# - Composed of **Encoder** and **Decoder**
# - Recieves inputs and pass to **Encoder** 
# - Pass the outputs from **Encoder** to **Decoder**
# - **Decoder** will decode according to outputs of previous timesteps as well as **Encoder** outputs  
# - Once done decoding, return the **Decoder** outputs

# + id="oRwKdLa0NEU6"
class Seq2Seq(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
    
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
    ):
        """
        Run the forward pass for an encoder-decoder model.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        logits, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return logits, extra


# + [markdown] id="zu3C2JfqNHzk"
# # Model Initialization

# + id="nyI9FOx-NJ2m"
# # HINT: transformer architecture
from fairseq.models.transformer import (
    TransformerEncoder, 
    TransformerDecoder,
)

def build_model(args, task):
    """ build a model instance based on hyperparameters """
    src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

    # token embeddings
    encoder_embed_tokens = nn.Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
    decoder_embed_tokens = nn.Embedding(len(tgt_dict), args.decoder_embed_dim, tgt_dict.pad())
    
    # encoder decoder
    # HINT: TODO: switch to TransformerEncoder & TransformerDecoder
    encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
    decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
    # encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
    # decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)

    # sequence to sequence model
    model = Seq2Seq(args, encoder, decoder)
    
    # initialization for seq2seq model is important, requires extra handling
    def init_params(module):
        from fairseq.modules import MultiheadAttention
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, MultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.RNNBase):
            for name, param in module.named_parameters():
                if "weight" in name or "bias" in name:
                    param.data.uniform_(-0.1, 0.1)
            
    # weight initialization
    model.apply(init_params)
    return model


# + [markdown] id="ce5n4eS7NQNy"
# ## Architecture Related Configuration
#
# For strong baseline, please refer to the hyperparameters for *transformer-base* in Table 3 in [Attention is all you need](#vaswani2017)

# + id="Cyn30VoGNT6N"
arch_args = Namespace(
    encoder_embed_dim=256,
    encoder_ffn_embed_dim=1024,
    encoder_layers=6,
    decoder_embed_dim=256,
    decoder_ffn_embed_dim=1024,
    decoder_layers=6,
    share_decoder_input_output_embed=True,
    dropout=0.1,
)

# HINT: these patches on parameters for Transformer
def add_transformer_args(args):
    args.encoder_attention_heads=4
    args.encoder_normalize_before=True
    
    args.decoder_attention_heads=4
    args.decoder_normalize_before=True
    
    args.activation_fn="relu"
    args.max_source_positions=1024
    args.max_target_positions=1024
    
    # patches on default parameters for Transformer (those not set above)
    from fairseq.models.transformer import base_architecture
    base_architecture(arch_args)

add_transformer_args(arch_args)


# + id="Nbb76QLCNZZZ"
if config.use_wandb:
    wandb.config.update(vars(arch_args))

# + id="7ZWfxsCDNatH" outputId="b48a48f4-ac2c-424a-dbab-6db1aba452e5" colab={"base_uri": "https://localhost:8080/"}
model = build_model(arch_args, task)
logger.info(model)


# + [markdown] id="aHll7GRNNdqc"
# # Optimization

# + [markdown] id="rUB9f1WCNgMH"
# ## Loss: Label Smoothing Regularization
# * let the model learn to generate less concentrated distribution, and prevent over-confidence
# * sometimes the ground truth may not be the only answer. thus, when calculating loss, we reserve some probability for incorrect labels
# * avoids overfitting
#
# code [source](https://fairseq.readthedocs.io/en/latest/_modules/fairseq/criterions/label_smoothed_cross_entropy.html)

# + id="IgspdJn0NdYF"
class LabelSmoothedCrossEntropyCriterion(nn.Module):
    def __init__(self, smoothing, ignore_index=None, reduce=True):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduce = reduce
    
    def forward(self, lprobs, target):
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        # nll: Negative log likelihood，the cross-entropy when target is one-hot. following line is same as F.nll_loss
        nll_loss = -lprobs.gather(dim=-1, index=target)
        #  reserve some probability for other labels. thus when calculating cross-entropy, 
        # equivalent to summing the log probs of all labels
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if self.ignore_index is not None:
            pad_mask = target.eq(self.ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if self.reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        # when calculating cross-entropy, add the loss of other labels
        eps_i = self.smoothing / lprobs.size(-1)
        loss = (1.0 - self.smoothing) * nll_loss + eps_i * smooth_loss
        return loss

# generally, 0.1 is good enough
criterion = LabelSmoothedCrossEntropyCriterion(
    smoothing=0.1,
    ignore_index=task.target_dictionary.pad(),
)


# + [markdown] id="aRalDto2NkJJ"
# ## Optimizer: Adam + lr scheduling
# Inverse square root scheduling is important to the stability when training Transformer. It's later used on RNN as well.
# Update the learning rate according to the following equation. Linearly increase the first stage, then decay proportionally to the inverse square root of timestep.
# $$lrate = d_{\text{model}}^{-0.5}\cdot\min({step\_num}^{-0.5},{step\_num}\cdot{warmup\_steps}^{-1.5})$$

# + id="sS7tQj1ROBYm"
def get_rate(d_model, step_num, warmup_step):
    # TODO: Change lr from constant to the equation shown above
    lr = d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_step ** (-1.5))
    return lr


# + id="J8hoAjHPNkh3"
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups
        
    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""                
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(c)
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return 0 if not step else self.factor * get_rate(self.model_size, step, self.warmup)


# + [markdown] id="VFJlkOMONsc6"
# ## Scheduling Visualized

# + id="A135fwPCNrQs" outputId="e83dace3-7fad-4c7b-e77c-79902eee681a" colab={"base_uri": "https://localhost:8080/", "height": 265}
optimizer = NoamOpt(
    model_size=arch_args.encoder_embed_dim, 
    factor=config.lr_factor, 
    warmup=config.lr_warmup, 
    optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001))
plt.plot(np.arange(1, 100000), [optimizer.rate(i) for i in range(1, 100000)])
plt.legend([f"{optimizer.model_size}:{optimizer.warmup}"])
None

# + [markdown] id="TOR0g-cVO5ZO"
# # Training Procedure

# + [markdown] id="f-0ZjbK3O8Iv"
# ## Training

# + id="foal3xM1O404"
from fairseq.data import iterators
from torch.cuda.amp import GradScaler, autocast

def train_one_epoch(epoch_itr, model, task, criterion, optimizer, accum_steps=1):
    itr = epoch_itr.next_epoch_itr(shuffle=True)
    itr = iterators.GroupedIterator(itr, accum_steps) # gradient accumulation: update every accum_steps samples
    
    stats = {"loss": []}
    scaler = GradScaler() # automatic mixed precision (amp) 
    
    model.train()
    progress = tqdm.tqdm(itr, desc=f"train epoch {epoch_itr.epoch}", leave=False)
    for samples in progress:
        model.zero_grad()
        accum_loss = 0
        sample_size = 0
        # gradient accumulation: update every accum_steps samples
        for i, sample in enumerate(samples):
            if i == 1:
                # emptying the CUDA cache after the first step can reduce the chance of OOM
                torch.cuda.empty_cache()

            sample = utils.move_to_cuda(sample, device=device)
            target = sample["target"]
            sample_size_i = sample["ntokens"]
            sample_size += sample_size_i
            
            # mixed precision training
            with autocast():
                net_output = model.forward(**sample["net_input"])
                lprobs = F.log_softmax(net_output[0], -1)            
                loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))
                
                # logging
                accum_loss += loss.item()
                # back-prop
                scaler.scale(loss).backward()                
        
        scaler.unscale_(optimizer)
        optimizer.multiply_grads(1 / (sample_size or 1.0)) # (sample_size or 1.0) handles the case of a zero gradient
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm) # grad norm clipping prevents gradient exploding
        norm_list.append(gnorm)
        scaler.step(optimizer)
        scaler.update()
        
        # logging
        loss_print = accum_loss/sample_size
        stats["loss"].append(loss_print)
        progress.set_postfix(loss=loss_print)
        if config.use_wandb:
            wandb.log({
                "train/loss": loss_print,
                "train/grad_norm": gnorm.item(),
                "train/lr": optimizer.rate(),
                "train/sample_size": sample_size,
            })
        
    loss_print = np.mean(stats["loss"])
    logger.info(f"training loss: {loss_print:.4f}")
    return stats


# + [markdown] id="Gt1lX3DRO_yU"
# ## Validation & Inference
# To prevent overfitting, validation is required every epoch to validate the performance on unseen data.
# - the procedure is essensially same as training, with the addition of inference step
# - after validation we can save the model weights
#
# Validation loss alone cannot describe the actual performance of the model
# - Directly produce translation hypotheses based on current model, then calculate BLEU with the reference translation
# - We can also manually examine the hypotheses' quality
# - We use fairseq's sequence generator for beam search to generate translation hypotheses

# + id="2og80HYQPAKq"
# fairseq's beam search generator
# given model and input seqeunce, produce translation hypotheses by beam search
sequence_generator = task.build_generator([model], config)

def decode(toks, dictionary):
    # convert from Tensor to human readable sentence
    s = dictionary.string(
        toks.int().cpu(),
        config.post_process,
    )
    return s if s else "<unk>"

def inference_step(sample, model):
    gen_out = sequence_generator.generate([model], sample)
    srcs = []
    hyps = []
    refs = []
    for i in range(len(gen_out)):
        # for each sample, collect the input, hypothesis and reference, later be used to calculate BLEU
        srcs.append(decode(
            utils.strip_pad(sample["net_input"]["src_tokens"][i], task.source_dictionary.pad()), 
            task.source_dictionary,
        ))
        hyps.append(decode(
            gen_out[i][0]["tokens"], # 0 indicates using the top hypothesis in beam
            task.target_dictionary,
        ))
        refs.append(decode(
            utils.strip_pad(sample["target"][i], task.target_dictionary.pad()), 
            task.target_dictionary,
        ))
    return srcs, hyps, refs


# + id="y1o7LeDkPDsd"
import shutil
import sacrebleu

def validate(model, task, criterion, log_to_wandb=True):
    logger.info('begin validation')
    itr = load_data_iterator(task, "valid", 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)
    
    stats = {"loss":[], "bleu": 0, "srcs":[], "hyps":[], "refs":[]}
    srcs = []
    hyps = []
    refs = []
    
    model.eval()
    progress = tqdm.tqdm(itr, desc=f"validation", leave=False)
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # validation loss
            sample = utils.move_to_cuda(sample, device=device)
            net_output = model.forward(**sample["net_input"])

            lprobs = F.log_softmax(net_output[0], -1)
            target = sample["target"]
            sample_size = sample["ntokens"]
            loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1)) / sample_size
            progress.set_postfix(valid_loss=loss.item())
            stats["loss"].append(loss)
            
            # do inference
            s, h, r = inference_step(sample, model)
            srcs.extend(s)
            hyps.extend(h)
            refs.extend(r)
            
    tok = 'zh' if task.cfg.target_lang == 'zh' else '13a'
    stats["loss"] = torch.stack(stats["loss"]).mean().item()
    stats["bleu"] = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok) # 計算BLEU score
    stats["srcs"] = srcs
    stats["hyps"] = hyps
    stats["refs"] = refs
    
    if config.use_wandb and log_to_wandb:
        wandb.log({
            "valid/loss": stats["loss"],
            "valid/bleu": stats["bleu"].score,
        }, commit=False)
    
    showid = np.random.randint(len(hyps))
    logger.info("example source: " + srcs[showid])
    logger.info("example hypothesis: " + hyps[showid])
    logger.info("example reference: " + refs[showid])
    
    # show bleu results
    logger.info(f"validation loss:\t{stats['loss']:.4f}")
    logger.info(stats["bleu"].format())
    return stats


# + [markdown] id="1sRF6nd4PGEE"
# # Save and Load Model Weights
#

# + id="edBuLlkuPGr9"
def validate_and_save(model, task, criterion, optimizer, epoch, save=True):   
    stats = validate(model, task, criterion)
    bleu = stats['bleu']
    loss = stats['loss']
    if save:
        # save epoch checkpoints
        savedir = Path(config.savedir).absolute()
        savedir.mkdir(parents=True, exist_ok=True)
        
        check = {
            "model": model.state_dict(),
            "stats": {"bleu": bleu.score, "loss": loss},
            "optim": {"step": optimizer._step}
        }
        torch.save(check, savedir/f"checkpoint{epoch}.pt")
        shutil.copy(savedir/f"checkpoint{epoch}.pt", savedir/f"checkpoint_last.pt")
        logger.info(f"saved epoch checkpoint: {savedir}/checkpoint{epoch}.pt")
    
        # save epoch samples
        with open(savedir/f"samples{epoch}.{config.source_lang}-{config.target_lang}.txt", "w") as f:
            for s, h in zip(stats["srcs"], stats["hyps"]):
                f.write(f"{s}\t{h}\n")

        # get best valid bleu    
        if getattr(validate_and_save, "best_bleu", 0) < bleu.score:
            validate_and_save.best_bleu = bleu.score
            torch.save(check, savedir/f"checkpoint_best.pt")
            
        del_file = savedir / f"checkpoint{epoch - config.keep_last_epochs}.pt"
        if del_file.exists():
            del_file.unlink()
    
    return stats

def try_load_checkpoint(model, optimizer=None, name=None):
    name = name if name else "checkpoint_last.pt"
    checkpath = Path(config.savedir)/name
    if checkpath.exists():
        check = torch.load(checkpath)
        model.load_state_dict(check["model"])
        stats = check["stats"]
        step = "unknown"
        if optimizer != None:
            optimizer._step = step = check["optim"]["step"]
        logger.info(f"loaded checkpoint {checkpath}: step={step} loss={stats['loss']} bleu={stats['bleu']}")
    else:
        logger.info(f"no checkpoints found at {checkpath}!")


# + [markdown] id="KyIFpibfPJ5u"
# # Main
# ## Training loop

# + id="hu7RZbCUPKQr"
model = model.to(device=device)
criterion = criterion.to(device=device)

# + id="5xxlJxU2PeAo" outputId="14fad86b-eda9-46c8-cfe9-4a9d858192ee" colab={"base_uri": "https://localhost:8080/"}
logger.info("task: {}".format(task.__class__.__name__))
logger.info("encoder: {}".format(model.encoder.__class__.__name__))
logger.info("decoder: {}".format(model.decoder.__class__.__name__))
logger.info("criterion: {}".format(criterion.__class__.__name__))
logger.info("optimizer: {}".format(optimizer.__class__.__name__))
logger.info(
    "num. model params: {:,} (num. trained: {:,})".format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
)
logger.info(f"max tokens per batch = {config.max_tokens}, accumulate steps = {config.accum_steps}")

# + id="MSPRqpQUPfaX" outputId="42ec80cf-484a-4607-9007-bd8f0d091db5" colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["4a47f7a18885448e9a9e0f042881d0be", "b1b66fdcfa924fa4b3fdf5000a3e5e33", "f53d3542302a4e019bec701037216c8c", "3f5fbebf4d8349ffb115a7e360ccb1dc", "d01e0b80819245daae1674b65b4cad68", "1b59c06f39884a32b087d61b95713b41", "5e6a5d16fd534b5e869d8606a52633b1", "a0ed85c0e1c64d3082cc9c64b5d7c7a2", "b890e4a17a4c4ce085cddd965834a151", "c42ccd1169e7422193783f8d4fb04233", "e4f1813588f248b3bd58bbd28c261543", "addbd853b3a748b6a4e5485669c50625", "1f6ffc8ee1b84c908f75734c1e0255da", "bcf2410204e94e56acc74bd48a727486", "cce695fdebeb49f3a0854751bda76b22", "70877d6c19724cf3baf2645d70c63ccf", "5ba5fbaa42e24988ba119086bfc8a95c", "b6d88361763a4a15b21cf41e44050900", "662376acf982466b8aefd4d3c400f4cd", "ea899227eea64a6084ab87d85d628d35", "3fd56c1be9d940c5b2b7c116768953d7", "5cb9409fb9194abca7693eb55370ea90", "374f21a9471c406babc06fe1420e005f", "fc71731fd1c9444b8891e7033c3a3ba0", "0f119c8a6db14f488fce9a18715f8847", "55cf9c92866e4e0f9d61fd5761766763", "6dc9c10e3b9b4d5daa3c9d6f564de7f3", "3677ab78c4a7439281f0e5fa86db471c", "ecce94387b084da0b881b86b0c2c6640", "53efe76aecea4f74aafa34340b05fdc1", "5e52881fad5c40438f7c0704cfcd36cf", "8b2a83f0c85a4fd8b4d8b38f644a9dff", "44914596e6874983b97329040e05584c", "87d8e994bdc942f4abe423e6a2ba90e3", "71fdedb2bb86420abeca441270c827b0", "defeb6b8aa2b4d3ab3ed597f8731235e", "303572fdc4c6427295369aa5f8d34fee", "6c25349ce98c4345b313b5ca8ceb5c81", "ed14727679104906a5db7ff3d1f81d7e", "fec6a7888a76432e969d6de072606714", "08689b3aa37c4dff9c0f6c834698341b", "fee0081853ba42ae9d6e25e2040a451e", "168b0f08e77a49ad980a1c4449281493", "163b421577424facb99563b9f721137a", "3ab908138ee648f6816fa611b75b4572", "32a8c2411f4445cf862a6dea01c1c9e7", "9c744512b35a4cd4afcfeaf67167c06a", "e2f271128b7046b4a2bdf8a274420bda", "4bef8f42a68a4e06962ccaa0628074f9", "8adcd529f8814e1d9b59cc7fa18ada26", "bbf0b7702779496da292ec2fc9a0384e", "f83af64bd73c4e59a85736fd8f59ea54", "263f8a12bb8f40658e1c4f770b836be2", "b6f6f9da7c6b48a1b6270b241bb6a466", "540349bca3864f69adff61c9dad9f911", "051b4553a2a543939e72baced9adf02c", "4f0e71a58a1d489eabbed6bbe17f3dce", "fe33af271c944ed2a54032f1e04db86b", "b937504bfc0e4d7a89ad03efc9dbf68b", "caa5690bbe984f8281f336dd0145faa8", "8439f8b762e1449a90ed2930d9460bc4", "745bfb93ea3f4ae2b53dc57815ab30cc", "9b102aa176b347689ec41618d88aa27f", "953300c9895e4d21a3b401b75e841cac", "a7730edb1b5a4fb2b8264f2a54f9fb64", "19f4bf1944ae48b2a26cc3068f6810e6", "a775aac7482a404fbd479e64d4ff8f9f", "5deeee868b174c9ca9ef91912349b919", "2ee5471d9129432e83a99fbdb8218b69", "966dcf88eba742a79d4473505aec26c7", "9d7b402091614fd999c769e8b3d27a4a", "6aad71bd75754a56b714df3113a93a98", "2251667bf7ba45ffb49f58bc7386c25e", "82da15020ecf4dda80f36fbf3cfde52c", "d82bb8669f7c4a09a1a375eda4be5ed5", "7174e5fe0ed247e8aac7c21c23ed13c2", "095410f3e2d74e1d9804f09a91f907e0", "9702b42d25064393a17990e263d3e639", "37dc01a1d91a4c69ab1f8f4794e7e9ae", "eefebb118fc745bfa3937057ff94bc5c", "b36905ccc92b49ce900eb3452af5e5e4", "0aef0afd92074d6a8c9704cf1feb851e", "cdaf3dac1a974b6f9ad33286d308cc77", "1d7f55b7fe49405d856bd7555d03c14d", "199974db7a96414ba6be7af7167e9a7e", "6e2ef5cab9954191addf6c5fde8bb493", "98c14acb0b4e4d22b0620f9c016adff7", "c3199327964142429eac049b79a676cf", "33c27b10c3ed497881cd26d8f58ddab5", "d5b3ef6549424390b5b133f83af0f608", "d1951753917442c68b0f0f8bce218545", "b5722c403bea4443a6b9f7315ac27583", "91712f2e898d4a0389e40fc740d2244f", "9db05703ae7b4d4e96b59bea290d0682", "e2d0a1daad7441c88d755625eaa5ed1d", "bea92d0a29724bbc9e6800147906e9ec", "889c059e7e8d454e85a2f4d0d384a649", "142aff61ce274141ac990457cbd94332", "398400d31d9641b7b73ac59e237c88c2", "0859059a2c7e44d1aa3954990d33491c", "e033766b4adb4acda989ca7ff5b821ed", "151d807dd8a64c9abe9ab69fdb217370", "e8202b785ae14ee0ab4d6b878e1a3878", "3cfb3049497d487abbbae47eaa2e82e0", "6af5e4f584864af981faaaa720f1083e", "6a66adae049040cba315131a242c60c8", "c2fed20e75ba4ce1912705d1d8a380f9", "2089f63b2dab445c8370d91ca36a0874", "a79e121e11bc4d7cbbfd57cc420e0da2", "64cc58d28b7b465e9b9ca318fd72986a", "ea264491c7544269b62621dc2f528cdf", "ac4a01efbec64fd1a54f81cbcd0d40a1", "efcf8dc2d96046b6b93eb2f7895e6f18", "1730cc473e4f4ee7a4adea8e210e9a22", "a338c907af5e4cc88121306bc1b01944", "d992b57ec9f04bec80899be629ca4b39", "225068d03f144f669b74b2c038218fe4", "1d929b9473ab44ae9f6e19f8040235d4", "9f21ad4755c34e5c92528cbd508182ec", "17237b928273424591445b909019fb4b", "09e55800f99a411ba0b9bf8f251603be", "c850be72a7254b4a9590678aeef2481f", "8571ef3a0a61432986e3ddb0f4ace082", "6796b3f51308473292ea055629711c8a", "630901e6bbc34294b42bf537379ea649", "df946dd14f7e493aa8e4cbd6b58804ff", "08656cc55e8b47bb83788cd9c1b01072", "c56070a37c2d44f990c174b83095f7ee", "964a3adb83be4fcfbb55ceb8e64292e6", "6f95bd888f5344d585df742eca6ff747", "97b2e436877b46c0aaaeb087263fb1e0", "b5c4fa6018504e638adafc3167decbce", "604f66b675964039b6046810e6ec6e59", "c9cffac13e4a4867baefde450fb3ea66", "0bd2b41c384a44f4ac1ee77c9ffafa0b", "d4776f35cdfc44aaba8bf580c632b118", "c9acd4e4e63b4bc69ad21d635dda6eb8", "efebfdd2b6a6482d87ea13b11fa7dd63", "05b5b3aa513e4d04a961685a78eac771", "74823e2a983846e3bead99afee8d96b0", "7e0b407aa303479ab097bdfd50a59f00", "c34297ec3fb74e919cbb8e3b543c92b1", "5d450348b19f4cd7bbede175cfd04932", "873845e15f2e4ecf931ecb2b1976527a", "9fd4c21b04f5411d893b31a31a19a0a0", "f6f3b1d9c1a34d759db85de2d43628c4", "90c48d42030144e89b2362be22cd2c20", "00ff9dc50781468fb7dbf8108421ac52", "36e53809d64e4f6793def52ae29500b8", "c7bb5b3e8571494abbd3ccf4f6475dad", "b0b8b8c954034078ad103e58cacf3202", "d76a074e360c4fcda18e97ae1d619a94", "070f759d6d85493997417557022454f5", "9b8a5dc0ec30489faeacad3deffd625c", "0aae43e4c64346c5b64672b2d75a5cb2", "a0983db0d1224d94b49903d638783c18", "58316fc34a9a48a3a538b65accba5394", "f81455a0d7fa419ebe27b95ca8c3b6b8", "f4fb6ec700594f15895d1a9d34c23fd9", "e7dbd0c2e6db4357ae26b3efabd7fdad", "54601153b07e4768872703bf6a8da65a", "5eda41c8273947fd9563fbcf4d932309", "064931edb68146e08ec877839192a422", "0dde76c348c1401aabe661ce7b98e1a4", "f7c8a42bf0b4470489f60649cbb270d0", "72f22049b2a5443cb723e2360d69c91c", "684f720feda94222b60990a48baa96aa", "f2acafbcb4604ab28d78ad04b52d47db", "8dbb002af5f74c808a62098842cd7d05", "32bfd808f5224487920144f4541d2610", "1a56035f50db44c0b49b4d363cd6f509", "9be065d3d27641108da39771cc4ff146", "caa2337ac38e430aa0d505df05dab6cc", "669064a6c0834c34973445d2a00719d8", "a8660c8724f8435faea6b22e759b58a3", "f1788471456043f094fcfa8c3fac69c1", "a6e39ea09f6342659d9b6b0893d3e31e", "67d99c2706ca4199acd5c02364ad8904", "66a0cc6d0ef84e12a083d708b80f9a4f", "7a77373d483a49e29fd0e512d6b5a3bc", "498668e5012240bf8ddcd0344c6d0b4a", "f8c8f210826c4901a1c9381971bf863b", "28b8be97004e49cab32ffc38ed512537", "7976ddd239ca4740aa68e1f415dbced2", "1e28350d28784af0aeca70dadfc7dce5", "79bbc79f6ce04fa082e26b3941eeb79c", "80692c0d6bf24470be03d16d2c8353eb", "282dfaa8792b456aa1f1ea2f57331f40", "55e4e3ee829f4c389fce225b6cccb9a1", "4ce0b3231f824045a35158fb99661ef3", "4d1e6932b33d4301af2c4bd8eefb48d9", "cf9d7d5c90354e74a101db87781bfdbb", "5085f642a00746e691675a8b984e3d39", "9d75319a2a8c476984207846a5b9f0c1", "37c52fe6a66f434ca9a17072b263a081", "0a0f090777a74321970c5fdba0487148", "aa202989fc3b4ce3a117943e629fe4ba", "70b0dbb8d98b4fe7a674c2e85c03ac08", "4ec20f3f493346bdb5a28c642fdc2927", "f00ca01ed3684c81b558161e07359031", "2f7e4943da204514ab77c8dea1bedaca", "1eaa65173bf147aa849ed8eea733ac58", "72a9ec4451e54f80a147c860b0b85810", "7203412953104751b416a305a5bf8746", "bfba6731ef324da2b8514e93f612072f", "eac409af26a24212ac8b950fb7c45b41", "65b4e867abf24ef08984f8371462a58d", "5db955ed64bb49629e9037b9ae094567", "7ff750ed21b04692ac64c5dfd9a3d7f4", "6c09115463654a0e8cc73c8a7c68a1f4", "1586bf557755418694625f83f7cd263b", "e71d38c179584849aa38a505fc8c4ae9", "4376fd69c09d457487d1b22ae7bcda49", "fbc5d7239159424088a202a53b6191e0", "77448686393643fcb5c9e7c3957629f9", "69853e4018f04a78a204e9426259eb6f", "20e084e226b846b79dd1052f517bec4b", "deebfb57854145959e66f035a1f4389f", "8a070eeab9274c9abeda0d8e2eb0ac19", "6ee47efd689c424c8a237c8af250d5d1", "370896b28cbf4fd1973a50dc21387bda", "e8d784efab9a4b558b81a6973736ad62", "0ea66bd56cad4b21bafb18d400ffac3f", "09282a897f3549a5bc5afa53e909a46b", "a975782e6ead46c1a429a315d84f3c01", "320c5004e15c433a8135d7771219aa21", "4ccd5cfd79ee46069ba2481df2f4a83f", "11a99b1ba6ba43d08f2133db229394cc", "a13ebc83459f47acbc8634de899e3124", "08f0a8a71f024ab78e0cbfbb90230654", "35d3e779d3f7462a8d3f3844d85b9d8d", "7c30c4d043c84cf29e27e42c351e4837", "8769913c235c40bda99aa2de851708c1", "88f98b922a704a948590f8357a043a84", "6b7c5c52b5624678b2b35c96b67fb23b", "d5b25015ab954963805b2494456b0196", "7412d686f2c244c4a8a2e37c50afdc6b", "76c08969994a4c928aa9d463fea752a5", "4840a5cdf50341ddaf88f529228c3a97", "6a829ae4176c48d29f92b0a472fda43d", "d8a4f9606b984fa1bec09e5fa9625042", "3649ec89102a4485b98b45431bdfd5d5", "5aeca22e1aea47a783fffeb89c6cc3db", "99e7484927c54fb4a5494b58130120e0", "3402cd09cbbb4325a4e200089fd22747", "660a014e896a4e7080d6880de0f4a9de", "c22e4cb03ae5447f91df604aaf374d1b", "697a9315f6404f5ab172a2ff1e4a72f7", "deee88a84af44c45b1fb13bd38d9d817", "0aaeef8657204eb0a27dcabd57cf8cc2", "4cc2eeb8cca54fc8a541aad9dff1d547", "b32c0d40ed964572859108fdf8916a8c", "c38fc9b80d104482a7f0b0c700baf639", "1dac76f4fe5543e1800def39820e4f1b", "b87b82808f494ac3a3286f4d1f8633e3", "3befad80f8ef4cf5bc7bee4ee0733f94", "d6e6fca249c34ae4bacb5fdf9a0fca40", "6bba263169f94b73bd4641752a9d8394", "0eb26b57c9c74f46a625a9857411094f", "6862011a951e4e269fd52d8b5daf31b4", "e311cb1458c24c678386156043534969", "2b550bcc98f748c3887636c8ad4383f3", "78aaac04e9314229a3713d83c7fccd6c", "25ea8e3acadc4c90b224450b803b5ae3", "5a5fa6953ca547f68152c6c0bcf4200c", "02ee4778dbd74cde8e438494ea2ae252", "f44d37fe34d244079981fa6f253f90b4", "ef54eaf734ec4f8d97c420edc962b5d9", "1e02003075f24c618e13b3505b32ed98", "c8d4b103ea8a498da1a86f4fef2fb81b", "5898b1cff11649f1935c6cfab96bc9ba", "eb64b4d293ad478daef18fd4184662ff", "25a0f313ffd14b2da062eddc227c052b", "3e8ea36d84de48afabffac3793e48e26", "5959b544a4474a5081d26a3b5d335b11", "f79a69f1c94147c4a91de342e1bfb794", "8635d64dae6446beb32fe9f1a4ee1ca3", "8099f7e10a6b42a2beb5f5fd808f323b", "906ef54ea9f94f46807a9dab589fbefc", "4c4c986a783b461ea15f0006b7b942bd", "c9bda966818243b6af157e3964411a1b", "9d1e52c1abee4184a8138b6974845d7a", "064414c8aadc4a8e80e1b29be9a297da", "28e1beeef81e4655ad93ff15a072f4f3", "9c43fa7f576540bdb6193e4cc1d3e4f9", "fe873a6d790c445880625a7a585dcfd7", "aea797ebecbe4f388cff71ff6f65b17c", "f2f0dd6d5f6a417095ae8b709f58d8ee", "489cb56522c746a5bf247fb94a50f9bc", "6acae2482ac64d2180d9c6be4e222019", "674268ad7c51464ab9ae638af506f566", "2b0a82fbdb174c65b1e081462dc8f0e6", "83a629d0c8264a3b821e311588fb8c8c", "ea57ac0547b449a7b8d2cb8b4c9323bd", "bcde3ab0745841c08ee090494dcedf4c", "316778728b1a4de295f2182dcd064778", "ea1e0ab3cf5c47b1b139b853365e294c"]}
epoch_itr = load_data_iterator(task, "train", config.start_epoch, config.max_tokens, config.num_workers)
try_load_checkpoint(model, optimizer, name=config.resume)
while epoch_itr.next_epoch_idx <= config.max_epoch:
    # train for one epoch
    train_one_epoch(epoch_itr, model, task, criterion, optimizer, config.accum_steps)
    stats = validate_and_save(model, task, criterion, optimizer, epoch=epoch_itr.epoch)
    logger.info("end of epoch {}".format(epoch_itr.epoch))    
    epoch_itr = load_data_iterator(task, "train", epoch_itr.next_epoch_idx, config.max_tokens, config.num_workers)

# + [markdown] id="KyjRwllxPjtf"
# # Submission

# + id="N70Gc6smPi1d"
# averaging a few checkpoints can have a similar effect to ensemble
checkdir=config.savedir
# !python ./fairseq/scripts/average_checkpoints.py \
# --inputs {checkdir} \
# --num-epoch-checkpoints 5 \
# --output {checkdir}/avg_last_5_checkpoint.pt

# + [markdown] id="BAGMiun8PnZy"
# ## Confirm model weights used to generate submission

# + id="tvRdivVUPnsU"
# checkpoint_last.pt : latest epoch
# checkpoint_best.pt : highest validation bleu
# avg_last_5_checkpoint.pt:　the average of last 5 epochs
try_load_checkpoint(model, name="avg_last_5_checkpoint.pt")
validate(model, task, criterion, log_to_wandb=False)
None


# + [markdown] id="ioAIflXpPsxt"
# ## Generate Prediction

# + id="oYMxA8FlPtIq"
def generate_prediction(model, task, split="test", outfile="./prediction.txt"):    
    task.load_dataset(split=split, epoch=1)
    itr = load_data_iterator(task, split, 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)
    
    idxs = []
    hyps = []

    model.eval()
    progress = tqdm.tqdm(itr, desc=f"prediction")
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # validation loss
            sample = utils.move_to_cuda(sample, device=device)

            # do inference
            s, h, r = inference_step(sample, model)
            
            hyps.extend(h)
            idxs.extend(list(sample['id']))
            
    # sort based on the order before preprocess
    hyps = [x for _,x in sorted(zip(idxs,hyps))]
    
    with open(outfile, "w") as f:
        for h in hyps:
            f.write(h+"\n")


# + id="Le4RFWXxjmm0"
generate_prediction(model, task)



# + [markdown] id="1z0cJE-wPzaU"
# # Back-translation

# + [markdown] id="5-7uPJ2CP0sm"
# ## Train a backward translation model

# + [markdown] id="ppGHjg2ZP3sV"
# 1. Switch the source_lang and target_lang in **config** 
# 2. Change the savedir in **config** (eg. "./checkpoints/transformer-back")
# 3. Train model

# + [markdown] id="waTGz29UP6WI"
# ## Generate synthetic data with backward model 

# + [markdown] id="sIeTsPexP8FL"
# ### Download monolingual data

# + id="i7N4QlsbP8fh"
mono_dataset_name = 'mono'

# + id="396saD9-QBPY"
mono_prefix = Path(data_dir).absolute() / mono_dataset_name
mono_prefix.mkdir(parents=True, exist_ok=True)

urls = (
    "https://github.com/yuhsinchan/ML2022-HW5Dataset/releases/download/v1.0.2/ted_zh_corpus.deduped.gz",
)
file_names = (
    'ted_zh_corpus.deduped.gz',
)

for u, f in zip(urls, file_names):
    path = mono_prefix/f
    if not path.exists():
        # !wget {u} -O {path}
    else:
        print(f'{f} is exist, skip downloading')
    if path.suffix == ".tgz":
        # !tar -xvf {path} -C {prefix}
    elif path.suffix == ".zip":
        # !unzip -o {path} -d {prefix}
    elif path.suffix == ".gz":
        # !gzip -fkd {path}

# + [markdown] id="JOVQRHzGQU4-"
# ### TODO: clean corpus
#
# 1. remove sentences that are too long or too short
# 2. unify punctuation
#
# hint: you can use clean_s() defined above to do this

# + id="eIYmxfUOQSov"



# + [markdown] id="jegH0bvMQVmR"
# ### TODO: Subword Units
#
# Use the spm model of the backward model to tokenize the data into subword units
#
# hint: spm model is located at DATA/raw-data/\[dataset\]/spm\[vocab_num\].model

# + id="vqgR4uUMQZGY"



# + [markdown] id="a65glBVXQZiE"
# ### Binarize
#
# use fairseq to binarize data

# + id="b803qA5aQaEu"
binpath = Path('./DATA/data-bin', mono_dataset_name)
src_dict_file = './DATA/data-bin/ted2020/dict.en.txt'
tgt_dict_file = src_dict_file
monopref = str(mono_prefix/"mono.tok") # whatever filepath you get after applying subword tokenization
if binpath.exists():
    print(binpath, "exists, will not overwrite!")
else:
    # !python -m fairseq_cli.preprocess\
#         --source-lang 'zh'\
#         --target-lang 'en'\
#         --trainpref {monopref}\
#         --destdir {binpath}\
#         --srcdict {src_dict_file}\
#         --tgtdict {tgt_dict_file}\
#         --workers 2

# + [markdown] id="smA0JraEQdxz"
# ### TODO: Generate synthetic data with backward model
#
# Add binarized monolingual data to the original data directory, and name it with "split_name"
#
# ex. ./DATA/data-bin/ted2020/\[split_name\].zh-en.\["en", "zh"\].\["bin", "idx"\]
#
# then you can use 'generate_prediction(model, task, split="split_name")' to generate translation prediction

# + id="jvaOVHeoQfkB"
# Add binarized monolingual data to the original data directory, and name it with "split_name"
# ex. ./DATA/data-bin/ted2020/\[split_name\].zh-en.\["en", "zh"\].\["bin", "idx"\]
# !cp ./DATA/data-bin/mono/train.zh-en.zh.bin ./DATA/data-bin/ted2020/mono.zh-en.zh.bin
# !cp ./DATA/data-bin/mono/train.zh-en.zh.idx ./DATA/data-bin/ted2020/mono.zh-en.zh.idx
# !cp ./DATA/data-bin/mono/train.zh-en.en.bin ./DATA/data-bin/ted2020/mono.zh-en.en.bin
# !cp ./DATA/data-bin/mono/train.zh-en.en.idx ./DATA/data-bin/ted2020/mono.zh-en.en.idx

# + id="fFEkxPu-Qhlc"
# hint: do prediction on split='mono' to create prediction_file
# generate_prediction( ... ,split=... ,outfile=... )

# + [markdown] id="Jn4XeawpQjLk"
# ### TODO: Create new dataset
#
# 1. Combine the prediction data with monolingual data
# 2. Use the original spm model to tokenize data into Subword Units
# 3. Binarize data with fairseq

# + id="3R35JTaTQjkm"
# Combine prediction_file (.en) and mono.zh (.zh) into a new dataset.
# 
# hint: tokenize prediction_file with the spm model
# spm_model.encode(line, out_type=str)
# output: ./DATA/rawdata/mono/mono.tok.en & mono.tok.zh
#
# hint: use fairseq to binarize these two files again
# binpath = Path('./DATA/data-bin/synthetic')
# src_dict_file = './DATA/data-bin/ted2020/dict.en.txt'
# tgt_dict_file = src_dict_file
# monopref = ./DATA/rawdata/mono/mono.tok # or whatever path after applying subword tokenization, w/o the suffix (.zh/.en)
# if binpath.exists():
#     print(binpath, "exists, will not overwrite!")
# else:
# #     !python -m fairseq_cli.preprocess\
# #         --source-lang 'zh'\
# #         --target-lang 'en'\
# #         --trainpref {monopref}\
# #         --destdir {binpath}\
# #         --srcdict {src_dict_file}\
# #         --tgtdict {tgt_dict_file}\
# #         --workers 2

# + id="MSkse1tyQnsR"
# create a new dataset from all the files prepared above
# !cp -r ./DATA/data-bin/ted2020/ ./DATA/data-bin/ted2020_with_mono/

# !cp ./DATA/data-bin/synthetic/train.zh-en.zh.bin ./DATA/data-bin/ted2020_with_mono/train1.en-zh.zh.bin
# !cp ./DATA/data-bin/synthetic/train.zh-en.zh.idx ./DATA/data-bin/ted2020_with_mono/train1.en-zh.zh.idx
# !cp ./DATA/data-bin/synthetic/train.zh-en.en.bin ./DATA/data-bin/ted2020_with_mono/train1.en-zh.en.bin
# !cp ./DATA/data-bin/synthetic/train.zh-en.en.idx ./DATA/data-bin/ted2020_with_mono/train1.en-zh.en.idx

# + [markdown] id="YVdxVGO3QrSs"
# Created new dataset "ted2020_with_mono"
#
# 1. Change the datadir in **config** ("./DATA/data-bin/ted2020_with_mono")
# 2. Switch back the source_lang and target_lang in **config** ("en", "zh")
# 2. Change the savedir in **config** (eg. "./checkpoints/transformer-bt")
# 3. Train model

# + [markdown] id="_CZU2beUQtl3"
# 1. <a name=ott2019fairseq></a>Ott, M., Edunov, S., Baevski, A., Fan, A., Gross, S., Ng, N., ... & Auli, M. (2019, June). fairseq: A Fast, Extensible Toolkit for Sequence Modeling. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations) (pp. 48-53).
# 2. <a name=vaswani2017></a>Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017, December). Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems (pp. 6000-6010).
# 3. <a name=reimers-2020-multilingual-sentence-bert></a>Reimers, N., & Gurevych, I. (2020, November). Making Monolingual Sentence Embeddings Multilingual Using Knowledge Distillation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) (pp. 4512-4525).
# 4. <a name=tiedemann2012parallel></a>Tiedemann, J. (2012, May). Parallel Data, Tools and Interfaces in OPUS. In Lrec (Vol. 2012, pp. 2214-2218).
# 5. <a name=kudo-richardson-2018-sentencepiece></a>Kudo, T., & Richardson, J. (2018, November). SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (pp. 66-71).
# 6. <a name=sennrich-etal-2016-improving></a>Sennrich, R., Haddow, B., & Birch, A. (2016, August). Improving Neural Machine Translation Models with Monolingual Data. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 86-96).
# 7. <a name=edunov-etal-2018-understanding></a>Edunov, S., Ott, M., Auli, M., & Grangier, D. (2018). Understanding Back-Translation at Scale. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (pp. 489-500).
# 8. https://github.com/ajinkyakulkarni14/TED-Multilingual-Parallel-Corpus
# 9. https://ithelp.ithome.com.tw/articles/10233122
# 10. https://nlp.seas.harvard.edu/2018/04/03/attention.html
# 11. https://colab.research.google.com/github/ga642381/ML2021-Spring/blob/main/HW05/HW05.ipynb

# + id="Rrfm6iLJQ0tS"
# print(norm_list)
import matplotlib.pyplot as plt
gnorm_list =[]
for i in range(len(norm_list)):
    gnorm_list.append(norm_list[i].item())  




pos_emb = model.decoder.embed_positions.weights.cpu().detach()
print(pos_emb.shape)
sim_matrix = np.zeros((1026,1026))
for i in range(1026):
    for j in range(1026):
        sim_matrix[i][j] = nn.functional.cosine_similarity(pos_emb[i],pos_emb[j],0)



plt.plot(gnorm_list, 'b')
#labels = range(0, len(gnorm_list))
#plt.xticks(labels)
plt.show()
plt.xlabel('Step')
plt.ylabel('grad norm')


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(sim_matrix, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()
plt.show()
