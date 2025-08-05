# library
## basic ----------------------------
import os
import pickle
import pandas as pd
import numpy as np
import warnings
import json
import random
from tap import Tap
from tqdm import tqdm, trange
from datetime import datetime
from pathlib import Path
from datetime import datetime
from typing import Any
import unicodedata
import ast
import re
import multiprocessing
import itertools
import sqlite3
import scipy
import gc
# natural language ---------------
import nltk
from nltk.corpus import wordnet
import MeCab
import ipadic
## visualization -----------------
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import networkx as nx
## machine learning ---------------
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, f1_score
## deep learning -------------------
import optuna
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils import BatchEncoding
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from lime.lime_text import LimeTextExplainer


# Intializing envs ========================================
# torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
warnings.simplefilter('ignore')
np.set_printoptions(precision=3)


# args =======================================================
class Args(Tap):
    # notes ---------------------------------------
    note: str = '論文用:ハイパーパラメータチューニングして性能を検証する'
    previous_result = '' # None
    # envs -----------------------------------------
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed: int = 42
    # model ---------------------------------
    language: str = 'en'
    model_name: str = ''
    model_path: str = ''
    """model_path_list
    - roberta-large
    - microsoft/deberta-v3-large
    - studio-ousia/luke-large-lite
    """
    # hyper parameter -------------------
    batch_size: int = 16
    encoder_lr: float = 1e-4
    pooler_lr: float = 1e-4
    cls_lr: float = 1e-4
    pooler_dropout: float = 0.2
    cls_dropout: float = 0.0
    max_length: int = 512 # 440 was best
    num_layer: int = 1 # 更新するencoder layerの数
    smooth_eps: float = 0.1
    thred: float = 0.5 # probからpredを生成する閾値
    # optuna ----------------------
    use_optuna: bool = True
    inner_cvs: int = 4 # 20% for inner val data and 60% are train data
    optuna_epochs: int = 16
    n_trials: int = 128
    hypara_cand: dict = { # ?? stepを指定する？
        'batch_size': [4, 32],
        'encoder_lr': [1e-5, 1e-3],
        'pooler_lr': [1e-5, 1e-3],
        'cls_lr': [1e-5, 1e-3],
        'pooler_dropout': [0.0, 0.5],
        'cls_dropout': [0.0, 0.5],
        'num_layer': [1, 10]
        }
    # training ----------------------
    epochs: int = 16
    num_warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    num_workers: int = 5 # for dataloader
    cv_once: bool = False
    # cross validation --------------
    outer_cvs: int = 5
    # dataset ---------------------
    use_JA: bool = True
    use_UN: bool = True
    use_OF: bool = True # official documents
    use_II: bool = True
    use_slice: bool = False
    use_augmentation: bool = False
    augmentation_ratio: float = 0.2
    replacement_ratio: float = 0.1
    # variables --------------------
    goal_names: list[str] = ['Goal' + '{:0=2}'.format(goal+1) for goal in range(17)]
    class_number: int = 17
    sdgs_colors: list[str] = ['#E5243B','#DDA63A','#4C9F38','#C5192D','#FF3A21','#26BDE2','#FCC30B','#A21942','#FD6925','#DD1367','#FD9D24','#BF8B2E','#3F7E44','#0A97D9','#56C02B','#00689D','#19486A']
    goal_contents: list[str] = ['GOAL 01: No Poverty','GOAL 02: Zero Hunger','GOAL 03: Good Health and Well-being','GOAL 04: Quality Education','GOAL 05: Gender Equality','GOAL 06: Clean Water and Sanitation','GOAL 07: Affordable and Clean Energy','GOAL 08: Decent Work and Economic Growth','GOAL 09: Industry, Innovation and Infrastructure','GOAL 10: Reduced Inequality','GOAL 11: Sustainable Cities and Communities','GOAL 12: Responsible Consumption and Production','GOAL 13: Climate Action','GOAL 14: Life Below Water','GOAL 15: Life on Land','GOAL 16: Peace and Justice Strong Institutions','GOAL 17: Partnerships to achieve the Goal']
    # path --------------------------------
    if os.name == 'nt':
        abs_path = r'D:/Dropbox/pj07_sdgs_translator_ceis'
    elif os.name == 'posix':
        abs_path = '/mnt/sdb1/Dropbox/pj07_sdgs_translator_ceis'
    # visualize setting ------------------
    plt.style.use(os.path.join(abs_path, 'utils/myacy_white_style.mplstyle'))
    fm.fontManager.addfont(os.path.join(abs_path, 'utils/fonts/SFProFonts/Library/Fonts/SF-Pro.ttf'))
    plt.rcParams["font.family"] = 'SF Pro'
    # utils -------------------------------
    def process_args(self):
        if args.previous_result == '':
            now = datetime.now().strftime('%Y%m%d%H%M')
            save_dir = now+'_'+self.model_path.replace('/','-')
        else:
            save_dir = args.previous_result
        self.output_dir = self.make_output_dir(self.abs_path, save_dir)
    
    def make_output_dir(self, abs_path, save_dir) -> Path:
        args = [abs_path, 'results', save_dir]
        output_dir = Path(*args)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


# Data IO ------------------------------------------
def pkl_loader(pkl_filename):
    with open(pkl_filename, 'rb') as web:
        data = pickle.load(web)
    return data


def pkl_saver(object, pkl_filename):
    with open(pkl_filename, 'wb') as web:
        pickle.dump(object , web)


def log(args, metrics: dict, filename='log.csv') -> None:
    path = Path(args.output_dir / filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df: pd.DataFrame = pd.read_csv(path)
        df = pd.DataFrame(df.to_dict('records') + [metrics])
        df.to_csv(path, index=False)
    else:
        pd.DataFrame([metrics]).to_csv(path, index=False)
    tqdm.write(
        f"epoch: {metrics['epoch']}  "
        f"loss: {metrics['loss']:2.4f}  "
        f"precision: {metrics['precision']:.4f}  "
        f"recall: {metrics['recall']:.4f}  "
        f"f1: {metrics['f1']:.4f}"
    )


def clone_state_dict(net) -> dict: # ????.state_dict()とのちがいは
    return {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}


def save_json(data: dict[Any, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_readable_config(data, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = vars(data)
    data = data['_option_string_actions']
    data = {k.replace('-', ''): v.default for k, v in data.items()}
    data = {k: v if type(v) in [int, float, bool, None] else str(v) for k, v in data.items()}
    save_json(data, path)


# pre-processing dataset --------------------------------------
def process_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = text.strip() # 不要なスペースを削除
    text = re.sub(r'\s+', ' ', text) # ２つ以上連続する空白を削除
    return text


def type_modify(x):
    if type(x) == str:
        return ast.literal_eval(x)
    else:
        return list(x)


def measure_token_length(text: str, tokenizer) -> str:
    tokenized_text = tokenizer.decode(tokenizer.encode(text))
    return len(tokenized_text.split(' '))


# create dataset -------------------------------------------------
def create_dataset(args) -> pd.DataFrame:
    df = pd.DataFrame()
    if args.language == 'ja':    
        if args.use_JA:
            JA_df_path = os.path.join(args.abs_path, 'data', 'Japanese_corpus', 'japanese_corpus_20240514')
            JA_df = pkl_loader(JA_df_path)
            df = pd.concat([df, JA_df]).reset_index(drop=True)
        if args.use_UN:
            UN_JA_df_path = os.path.join(args.abs_path, 'data', 'UN_SDG_Actions_Platform', 'un_sdg_actions_platform_ja_20240122')
            UN_df = pkl_loader(UN_JA_df_path)
            df = pd.concat([df, UN_df]).reset_index(drop=True)
        if args.use_OF:
            sdg_progress_path = os.path.join(args.abs_path, 'data', 'un_sdg_progress', 'un_sdg_progress_jpn')
            sdg_progress = pkl_loader(sdg_progress_path)
            sdgs_difinition_path = os.path.join(args.abs_path, 'data', 'sdgs_difinition', 'sdgs_difinition_jpn')
            sdgs_difinition = pkl_loader(sdgs_difinition_path)
            df = pd.concat([df, sdg_progress, sdgs_difinition]).reset_index(drop=True)
        if args.use_II:
            IISD_df_path = os.path.join(args.abs_path, 'data', 'IISD_SDG_Knowledge_Hub', 'IISD_SDG_Knowledge_Hub_JA_20240428')
            IISD_df = pkl_loader(IISD_df_path)
            df = pd.concat([df, IISD_df]).reset_index(drop=True)
    elif args.language == 'en':
        if args.use_JA:
            JA_EN_df_path = os.path.join(args.abs_path, 'data', 'Japanese_corpus', 'japanese_corpus_en_20240516')
            JA_df = pkl_loader(JA_EN_df_path)
            df = pd.concat([df, JA_df]).reset_index(drop=True)
        if args.use_UN:
            UN_df_path = os.path.join(args.abs_path, 'data', 'UN_SDG_Actions_Platform', 'un_sdg_actions_platform_20240514')
            UN_df = pkl_loader(UN_df_path)
            df = pd.concat([df, UN_df]).reset_index(drop=True)
        if args.use_OF:
            sdg_progress_path = os.path.join(args.abs_path, 'data', 'un_sdg_progress', 'un_sdg_progress')
            sdg_progress = pkl_loader(sdg_progress_path)
            sdgs_difinition_path = os.path.join(args.abs_path, 'data', 'sdgs_difinition', 'sdgs_difinition')
            sdgs_difinition = pkl_loader(sdgs_difinition_path)
            df = pd.concat([df, sdg_progress, sdgs_difinition]).reset_index(drop=True)
        if args.use_II:
            IISD_df_path = os.path.join(args.abs_path, 'data', 'IISD_SDG_Knowledge_Hub', 'IISD_SDG_Knowledge_Hub_20240428')
            IISD_df = pkl_loader(IISD_df_path)
            df = pd.concat([df, IISD_df]).reset_index(drop=True)
    
    # text processing
    df['text'] = df['text'].apply(process_text)
    df['label'] = df['label'].apply(type_modify)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# data augmentation -----------------------------------------------
def get_synonyms(word: str, sense_word) -> list[str]:
    synsets = sense_word.loc[sense_word.lemma == word, 'synset'] # wordに一致するsynsets(概念)を検索する
    synset_words = set(sense_word.loc[sense_word.synset.isin(synsets), 'lemma']) # そのsynsetsに紐づく単語をすべて所得しつつ重複は消す
    if word in synset_words: # もとの単語=検索元 は削除
        synset_words.remove(word)
    return list(synset_words)


def wakati_text(text, stop_words):
    m = MeCab.Tagger(ipadic.MECAB_ARGS)
    p = m.parse(text)
    p_splits = [i.split('\t') for i in p.split('\n')][:-2]
    raw_words = [x[0] for x in p_splits]
    info = [x[1].split(',') for x in p_splits]
    lemma_words = [x[6] if x[0] in ['名詞', '動詞'] else '' for x in info]
    lemma_words = ['' if word in stop_words else word for word in lemma_words]
    return raw_words, lemma_words


def synonym_replacement(n, raw_words, lemma_words, sense_word):
    new_words = raw_words.copy()
    lemma_words_idx = [i for i, x in enumerate(lemma_words) if x != '']
    sampler = random.sample(lemma_words_idx, n) # ここは参考コードにはなかったところ。これがないと前から順にreplaceしかしなくなる
    for idx in sampler:
        raw_word = raw_words[idx]
        synonyms = get_synonyms(lemma_words[idx], sense_word) # 見出し語をもとに類義語検索
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms) # 類義語のなかからランダムに１つ選択
            new_words = [synonym if word == raw_word else word for word in new_words]
    return new_words


def create_augmented_df(args, dataframe):
    # load utilities ---------------------
    conn = sqlite3.connect(os.path.join(args.abs_path, 'utils', 'wnjpn.db')) # downloaded from https://bond-lab.github.io/wnja/ and reffer to https://qiita.com/pocket_kyoto/items/1e5d464b693a8b44eda5
    sense_word = pd.read_sql('SELECT synset,lemma FROM sense,word USING (wordid) WHERE sense.lang="jpn"', conn)
    stop_words = pd.read_csv(os.path.join(args.abs_path, 'utils', 'Japanese.txt'), header=None)[0].to_list() # wget 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    # augmentation ------------------------
    augmented_df = pd.DataFrame()
    augmentation_times = int(len(dataframe)*args.augmentation_ratio)
    print('----- start data augmentation -----')
    for time in trange(augmentation_times):
        raw_words, lemma_words = wakati_text(dataframe['text'][time], stop_words)
        num_words = len(raw_words)
        n = int(num_words * args.replacement_ratio)
        augmented_text = ''.join(synonym_replacement(n, raw_words, lemma_words, sense_word))
        augmented_df = pd.concat([augmented_df, pd.DataFrame({'text':[augmented_text], 'label':[dataframe['label'][time]]})], axis=0).reset_index(drop=True)
    return pd.concat([dataframe, augmented_df], axis=0).reset_index(drop=True)


# create dataloader -----------------------------------------
class CustomDataset(Dataset):
    def __init__(self, args, dataframe):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.data = dataframe
        self.text = self.data['text']
        self.labels = self.data['label']
        self.max_length = args.max_length
    def __len__(self):
        return len(self.text)
    def __getitem__(self, index):
        text = str(self.text[index])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens = True,
            max_length = self.max_length,
            pad_to_max_length = True,
            return_token_type_ids = True,
            truncation = True,
            # return_tensor='py' はしないほうがデータ型が使いやすくなる。なぜだろう
        )
        return BatchEncoding({
            'input_ids': torch.tensor(inputs.input_ids, dtype=torch.long), # LongTensor()にするとうまく動かなくなる
            'attention_mask': torch.tensor(inputs.attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(inputs.token_type_ids, dtype=torch.long),
            'position': torch.tensor(list(range(0, self.max_length)), dtype=torch.long),
            'labels': torch.tensor(self.labels[index], dtype=torch.long)})


def create_dataloader(args, data, shuffle=True):
    loader_params = {'batch_size': args.batch_size, 'shuffle': shuffle, 'num_workers': args.num_workers, 'pin_memory': True}
    data_set = CustomDataset(args, data)
    data_loader = DataLoader(data_set, **loader_params)
    return data_loader


class sdgs_net(nn.Module):
    def __init__(self, args):
        super(sdgs_net, self).__init__()
        self.bert = AutoModel.from_pretrained(args.model_path)
        self.bert.pooler = nn.Sequential(
            nn.Linear(in_features=self.bert.config.hidden_size, out_features=self.bert.config.hidden_size),
            nn.Dropout(args.pooler_dropout),
            nn.Tanh()
        )
        self.cls = nn.Sequential(
            nn.Linear(in_features=self.bert.config.hidden_size, out_features=args.class_number),
            nn.Dropout(args.cls_dropout)
        )
    def forward(self, input_ids, attention_mask, token_type_ids, position, labels):
        bert_output = self.bert(input_ids, attention_mask, token_type_ids, position, output_attentions=True, output_hidden_states=True)
        average_hidden_state = (bert_output.last_hidden_state*attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        net_output = self.cls(self.bert.pooler(average_hidden_state))
        return net_output, average_hidden_state, bert_output.attentions


def net_initializer(args, net):
    for param in net.parameters():
        param.requires_grad = False
    for param in net.bert.encoder.layer[net.bert.config.num_hidden_layers - args.num_layer:].parameters():
        param.requires_grad = True
    for param in net.bert.pooler.parameters():
        param.requires_grad = True
    for param in net.cls.parameters():
        param.requires_grad = True
    return net


def create_optimizer(args, net, dl) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    no_decay = {'bias', 'LayerNorm.weight'}
    optimizer_grouped_parameters = [
        # last layer --------------
        {'params': [param for name, param in net.bert.encoder.layer[net.bert.config.num_hidden_layers - args.num_layer:].named_parameters() if not name in no_decay],
         'weight_decay': args.weight_decay,
         'lr': args.encoder_lr},
        {'params': [param for name, param in net.bert.encoder.layer[net.bert.config.num_hidden_layers - args.num_layer:].named_parameters() if name in no_decay],
         'weight_decay': 0.0,
         'lr': args.encoder_lr},
         # pooler -----------------
        {'params': [param for name, param in net.bert.pooler.named_parameters() if not name in no_decay],
         'weight_decay': args.weight_decay, 
         'lr': args.pooler_lr},
        {'params': [param for name, param in net.bert.pooler.named_parameters() if name in no_decay],
         'weight_decay': 0.0, 
         'lr': args.pooler_lr},
        # cls ------------------
        {'params': [param for name, param in net.cls.named_parameters() if not name in no_decay],
         'weight_decay': args.weight_decay, 
         'lr': args.cls_lr},
        {'params': [param for name, param in net.cls.named_parameters() if name in no_decay],
         'weight_decay': 0.0,
         'lr': args.cls_lr}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-6, betas=(0.9,0.999))
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(dl)*args.epochs*args.num_warmup_ratio,
        num_training_steps=len(dl)*args.epochs
    )
    return optimizer, lr_scheduler


def create_criterion(args, dataframe):
    labels = torch.tensor(np.array(dataframe.label.tolist()))
    label_freq = labels.sum(dim=0) / len(labels)
    weights = 1 / label_freq
    weights /= torch.min(weights)
    weights = weights.to(args.device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
    return criterion


# training ------------------------------------------
def run_one_epoch(args, phase, net, dl, criterion, optimizer, lr_scheduler, scaler):
    total_loss, true_list, prob_list, pred_list = 0, [], [], []
    for batch in tqdm(dl, total=len(dl), dynamic_ncols=True, leave=True):
        with torch.set_grad_enabled(phase == 'train'):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else None):
                output, _, _ = net(**batch.to(args.device))
                labels = batch['labels'].to(args.device, dtype=torch.float)
                smooth_labels = (1-2*args.smooth_eps)*labels + args.smooth_eps
                loss = criterion(output, smooth_labels)
        if phase == 'train':
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            if scale <= scaler.get_scale():
                lr_scheduler.step()
        total_loss += loss.item() * args.batch_size
        true_list += batch.labels.tolist()
        prob = torch.sigmoid(output)
        prob_list += prob.tolist()
        pred_list += (prob > args.thred).int().tolist()
    precision, recall, f1, _ = precision_recall_fscore_support(true_list, pred_list, average='micro')
    report = pd.DataFrame(classification_report(true_list, pred_list, output_dict=True)).T.iloc[:17, 2:3]
    report.index = args.goal_names
    report = report.to_dict()
    metrics = {'loss': total_loss / len(dl), 'precision': precision, 'recall': recall, 'f1': f1, **report['f1-score']}
    result = {'total_loss': [total_loss], 'true_list': true_list, 'prob_list': prob_list, 'pred_list': pred_list}
    return net, metrics, result


def run_cv(args):
    outer_cv_results = {'total_loss': list(), 'true_list': list(), 'prob_list': list(), 'pred_list': list()}
    df = create_dataset(args)
    print(df.info())
    splitter = MultilabelStratifiedKFold(n_splits=args.outer_cvs, shuffle=True, random_state=42)
    X = df.text.values
    y = np.array(df.label.tolist())
    for cv, (train_index, val_index) in enumerate(splitter.split(X, y)):
        print(f'outer_cv={cv}/{args.outer_cvs}')
        in_time = datetime.now()
        # load dataset ---------------------------------------
        train_df = pd.DataFrame({'text': X[train_index], 'label': y[train_index].tolist()})
        val_df = pd.DataFrame({'text': X[val_index], 'label': y[val_index].tolist()})
        # data augmentation ---------------------------------
        if args.use_augmentation:
            train_df = create_augmented_df(args, train_df)
        # dataloader -----------------------------------------
        dl_dict = {'train': create_dataloader(args, train_df, shuffle=True), 
                   'val': create_dataloader(args, val_df, shuffle=False)}
        # initialize net--------------------------------------
        net = sdgs_net(args).to(args.device, non_blocking=True)
        net = net_initializer(args, net)
        # training tools --------------------------------------
        criterion = create_criterion(args, train_df)
        optimizer, lr_scheduler = create_optimizer(args, net, dl_dict['train'])
        scaler = torch.cuda.amp.GradScaler()
        # training --------------------------------------------
        for epoch in trange(args.epochs, dynamic_ncols=True):
            torch.cuda.empty_cache()
            for phase in ['train', 'val']:
                # set net mode and select criterion -----------
                if phase == 'train':
                    net.train()
                elif phase == 'val':
                    net.eval()
                # run ---------------------------------------
                net, metrics, result = run_one_epoch(args, phase, net, dl_dict[phase], criterion, optimizer, lr_scheduler, scaler)
                # final epoch -> save validation prediction result -----
                if (epoch == args.epochs-1) and (phase == 'val'):
                    for name in result.keys():
                        outer_cv_results[name].extend(result[name])
                # show metrics ----------------------------------------
                metrics = {'cv': cv, 'phase': phase, 'epoch': epoch, **metrics}
                log(args, metrics)
        # end time -----------------------------------------
        out_time = datetime.now()
        print(f'CV will finish on {out_time + (out_time-in_time)*(args.outer_cvs-cv-1)}')
        # cleaning -----------------------------------------
        del net, dl_dict
        torch.cuda.empty_cache()
        if args.cv_once:
            break
    pkl_saver(outer_cv_results, args.output_dir / 'outer_cv_results')
    classification_df = classification_report(outer_cv_results['true_list'], outer_cv_results['pred_list'], output_dict=True)
    classification_df = pd.DataFrame(classification_df).T
    classification_df.to_csv(args.output_dir / 'cv_classification_report.csv')


def objective_variable(args, train_df):

    def optuna_optimize(trial):
        # parameter set for optuna --------------------
        opt_args = args
        # objective parameters --------------------------------
        opt_args.batch_size = trial.suggest_int('batch_size', args.hypara_cand['batch_size'][0], args.hypara_cand['batch_size'][1])
        opt_args.encoder_lr = trial.suggest_loguniform('encoder_lr', args.hypara_cand['encoder_lr'][0], args.hypara_cand['encoder_lr'][1])
        opt_args.pooler_lr = trial.suggest_loguniform('pooler_lr', args.hypara_cand['pooler_lr'][0], args.hypara_cand['pooler_lr'][1])
        opt_args.cls_lr = trial.suggest_loguniform('cls_lr', args.hypara_cand['cls_lr'][0], args.hypara_cand['cls_lr'][1])
        opt_args.pooler_dropout = trial.suggest_float('pooler_dropout', args.hypara_cand['pooler_dropout'][0], args.hypara_cand['pooler_dropout'][1])
        opt_args.cls_dropout = trial.suggest_float('cls_dropout', args.hypara_cand['cls_dropout'][0], args.hypara_cand['cls_dropout'][1])
        opt_args.num_layer = trial.suggest_int('num_layer', args.hypara_cand['num_layer'][0], args.hypara_cand['num_layer'][1])
        args_dict = {'batch_size': opt_args.batch_size, 'encoder_lr': opt_args.encoder_lr, 'pooler_lr': opt_args.pooler_lr, 'cls_lr': opt_args.cls_lr,
                    'pooler_dropout': opt_args.pooler_dropout, 'cls_dropout': opt_args.cls_dropout,
                    'num_layer': opt_args.num_layer}
        # data loader ----------------------------
        opt_args.num_workers = 0
        inner_splitter = MultilabelStratifiedKFold(n_splits=args.inner_cvs, shuffle=True) # random_state=42で固定しないことでハイパラの過学習を防ぐ
        inner_X = train_df.text.values
        inner_y = np.array(train_df.label.tolist())
        inner_train_index, inner_val_index = next(inner_splitter.split(inner_X, inner_y))
        inner_train_df = pd.DataFrame({'text': inner_X[inner_train_index], 'label': inner_y[inner_train_index].tolist()})
        inner_val_df = pd.DataFrame({'text': inner_X[inner_val_index], 'label': inner_y[inner_val_index].tolist()})
        train_dl = create_dataloader(opt_args, inner_train_df.reset_index(drop=True))
        val_dl = create_dataloader(opt_args, inner_val_df.reset_index(drop=True))
        # training set initialize -------------------------------
        net = sdgs_net(opt_args).to(opt_args.device, non_blocking=True)
        net = net_initializer(opt_args, net)
        criterion = create_criterion(opt_args, inner_train_df)
        optimizer, lr_scheduler = create_optimizer(opt_args, net, train_dl)
        scaler = torch.cuda.amp.GradScaler()
        # training ---------------------------------------------
        best_val_f1 = 0
        for epoch in trange(opt_args.optuna_epochs, dynamic_ncols=True):
            torch.cuda.empty_cache()
            for phase in ['train', 'val']:
                # train and eval -----------
                if phase == 'train':
                    net.train()
                    net, metrics, _ = run_one_epoch(opt_args, phase, net, train_dl, criterion, optimizer, lr_scheduler, scaler)
                elif phase == 'val':
                    net.eval()
                    net, metrics, _ = run_one_epoch(opt_args, phase, net, val_dl, criterion, optimizer, lr_scheduler, scaler)
                    if metrics['f1'] > best_val_f1:
                        best_val_f1 = metrics['f1']
                # show metrics ------------------------------
                metrics = {'phase': phase, 'epoch': epoch, **metrics, **args_dict}
                log(args, metrics, args.output_dir / 'optuna_log.csv')
            # for pruner
            trial.report(best_val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        del inner_train_df, inner_val_df, train_dl, val_dl, net
        torch.cuda.empty_cache()
        gc.collect()
        return best_val_f1
    
    return optuna_optimize


def generate_best_model(args):
    # load dataset ---------------------------------------
    df = create_dataset(args)
    dl = create_dataloader(args, df.reset_index(drop=True), shuffle=True)
    # initialize net ------------------------------------
    net = sdgs_net(args).to(args.device, non_blocking=True)
    net = net_initializer(args, net)
    # training tools --------------------------------------
    criterion = create_criterion(args, df)
    optimizer, lr_scheduler = create_optimizer(args, net, dl)
    scaler = torch.cuda.amp.GradScaler()
    # training -------------------------------------------
    for epoch in trange(args.epochs, dynamic_ncols=True):
        torch.cuda.empty_cache()
        net.train()
        net, metrics, result = run_one_epoch(args, 'train', net, dl, criterion, optimizer, lr_scheduler, scaler)
        log(args, {'epoch': epoch, **metrics}, 'best_model_log.csv')
    best_state_dict = clone_state_dict(net)
    torch.save(best_state_dict, args.output_dir / 'best_model.pt')


# predicate ---------------------------------------
def min_max(l):
    l_min = min(l)
    l_max = max(l)
    return [(i - l_min) / (l_max - l_min) for i in l]


def predicator(args, text_list: list[str]) -> dict:
    # best net loader ---------
    best_net = sdgs_net(args).to(args.device, non_blocking=True)
    best_net = net_initializer(args, best_net)
    best_net.load_state_dict(torch.load(args.output_dir / 'best_model.pt'))
    # tokenizer loader ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # data -------------------
    analyze_data = pd.DataFrame({'text': text_list})
    analyze_data['label'] = [[0]*17 for _ in range(len(analyze_data))]
    analyze_dl = create_dataloader(args, analyze_data, shuffle=False) # shuffleしないように要注意！！！！！！！！
    # result box -------------
    results = {'tokenized_texts': list(), 'vecs': list(), 'probs': list(), 'preds': list(), 'attentions': list()}
    # predicate
    for batch in tqdm(analyze_dl, total=len(analyze_dl), dynamic_ncols=True, leave=True):
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else None):
                output, vec, attention = best_net(**batch.to(args.device))
        # tokenized text --------------------------------
        ids = batch['input_ids'].to(args.device, dtype=torch.long)
        for id in ids:
            results['tokenized_texts'].append(tokenizer.convert_ids_to_tokens(id))
        # vec ------------------------------------------
        results['vecs'].extend(vec.cpu().detach().to(torch.float32).numpy().tolist()) # float16を直接numpyに変換することはできない
        # prob and pred --------------------------------
        prob = torch.sigmoid(output)
        results['probs'].extend(prob.cpu().detach().to(torch.float32).numpy().tolist())
        results['preds'].extend((prob>args.thred).int().tolist())
        # attention ------------------------------------
        last_attention = attention[-1]
        cls_attention = last_attention[:,:,0,:]
        all_cls_attention = cls_attention.sum(dim=1) # sum all head attention
        results['attentions'].extend(all_cls_attention.cpu().detach().to(torch.float32).numpy().tolist())
    return results


def predicate_of_target(args) -> dict:
    target_dict = pd.read_csv(os.path.join(args.abs_path, 'data', 'sdgs_target.csv')).to_dict(orient='list')
    target_dict['color'] = [args.sdgs_colors[i-1] for i in target_dict['goal_code']]
    if args.language == 'ja':
        result = predicator(args, target_dict['text_ja'])
    elif args.language == 'en':
        result = predicator(args, target_dict['text_en'])
    result_of_target = {**target_dict, **result}
    pkl_saver(result_of_target, args.output_dir / 'target_dict')
    return result_of_target


def calculate_cossim(vec1, vec2):
    vec1, vec2 = torch.tensor(vec1), torch.tensor(vec2)
    vec1, vec2 = nn.functional.normalize(vec1, p=2, dim=1), nn.functional.normalize(vec2, p=2, dim=1)
    return torch.matmul(vec1, vec2.T)


def select_target_and_goal(target_dict, prob, sim, top_k=3):
    goal_target_dict = {}
    goal_prob = [(i, p) for i, p in enumerate(prob) if p > args.thred]
    for item in goal_prob:
        g, p = item[0], item[1]
        target_df = pd.DataFrame(target_dict)
        target_df['cossim'] = sim
        buff = target_df[target_df['goal_code']==(g+1)]
        buff = buff.sort_values(by='cossim', ascending=False)
        buff = buff.iloc[:top_k, :]
        texts = []
        for idx, row in buff.iterrows():
            text = f'(s={row.cossim:.0%}) {row.target_code}: {row.text_ja}'
            texts.append(text)
        goal_target_dict[f'{args.goal_contents[g]} (p={p:.0%})'] = texts
    return goal_target_dict


def make_html(args, results, file_name='attention_viz.html'):
    # calculate cossim with target ---------------------------------
    target_dict = predicate_of_target(args)
    cossim = calculate_cossim(results['probs'], target_dict['probs'])
    related_targets = []
    for i, preds in enumerate(results['preds']):
        related_targets.append(select_target_and_goal(target_dict, preds, cossim[i]))
    # misc ----------------------------------------------------------
    length = len(results['probs'])
    html = '<font face="Arial">'
    # toc ------------------------------------------------------------
    html += '<div class="toc"><ul>'
    for i in range(length):
        html += '<li><a href="#item{}">item {}</a></li>'.format(i,i)
    html += '</ul></div>'
    # contents -----------------------------------------------------
    for idx in trange(length):
        html += '<h2 id="item{}">item {}</h2><p>'.format(idx,idx)
        # text highlighted by Attention -----------------------------
        html += '<h3>Text</h3>'
        for t, a in zip(results['tokenized_texts'][idx][1:], min_max(results['attentions'][idx][1:])): # Jpn_BERT -> 1:, Eng_RoBERTa -> 2:
            if t[0] == '##': # Jpn_BERT -> ##, Eng.RoBERTa -> Ġ,
                html += ' '
            if (t != '[PAD]') & (t != '[SEP]'): # Jpn_BERT -> [PAD] or [SEP], Eng_RoBERTa -> <pad>
                html_color = '#%02X%02X%02X' % (255, int(255*(1 - a)), int(255*(1 - a)))
                html += '<span style="background-color: {}">{}</span>'.format(html_color, t.replace('##', ''))  # Jpn_BERT -> ##, Eng.RoBERTa -> Ġ, 
        # goal and target -----------------------------------------
        html += '<h3>Predicted Goals and Targets</h3>'
        goal_target = related_targets[idx]
        for goal_content, targets in goal_target.items():
            html += f'<h4><span class="highlight">{goal_content}</span></h4>'
            for target in targets:
                html += target
                html += '<br>'
            html += '<br>'
        html += '</p><hr>'
    html += '</font>'
    # save -----------------------------------------------------------
    with open(args.output_dir / file_name, 'w', encoding='UTF-8') as res:
        res.write(html)


# visualize ----------------------------------------
def viz_validation_result(args):
    log_df = pd.read_csv(args.output_dir / 'classification_report.csv')
    f1_scores = log_df['f1-score'].tolist()[:17]
    micro_avg = log_df.loc[17, 'f1-score']
    fig, ax = plt.subplots(figsize=(12,7))
    bars = ax.bar(x=args.goal_names, height=f1_scores, color=args.sdgs_colors)
    ax.set_ylim((0,1))
    ax.axhline(micro_avg, color='black', label='micro-avg = {:.3f}'.format(micro_avg))
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x()+bar.get_width() / 2., 1.05*height, '{:.3f}'.format(height), ha='center', va='center')
    ax.set_xticklabels(['{}'.format(i+1) for i in range(17)])
    ax.set_title('validation result')
    ax.set_xlabel('goal')
    ax.set_ylabel('f1-score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / 'outer_validation_result.png')
    plt.close()


def viz_f1_transition(args):
    logs = pd.read_csv(args.output_dir / 'log.csv')
    column_names = args.goal_names + ['f1']
    color_list = args.sdgs_colors + ['black']
    title_list = [f'goal_{i}' for i in range(1, 18)] + ['overall']
    # train result ------------------------------------------------
    train_mean_list = []
    train_std_list = []
    buff = logs[logs['phase']=='train']
    for name in column_names:
        means, stds = [], []
        for e in range(args.epochs):
            means.append(buff[buff['epoch']==e][name].mean())
            stds.append(buff[buff['epoch']==e][name].std())
        train_mean_list.append(means)
        train_std_list.append(stds)
    train_mean_list = np.array(train_mean_list)
    train_std_list = np.array(train_std_list)
    # val result ------------------------------------------------
    val_mean_list = []
    val_std_list = []
    buff = logs[logs['phase']=='val']
    for name in column_names:
        means, stds = [], []
        for e in range(args.epochs):
            means.append(buff[buff['epoch']==e][name].mean())
            stds.append(buff[buff['epoch']==e][name].std())
        val_mean_list.append(means)
        val_std_list.append(stds)
    val_mean_list = np.array(val_mean_list)
    val_std_list = np.array(val_std_list)
    # viz -------------------------------------------------------
    n_rows, n_cols = 3, 6
    fig, axs = plt.subplots(n_rows, n_cols, sharex='all', sharey='all', figsize=(20,10))
    fig.supylabel('f1-score')
    fig.supxlabel('epoch')
    fig.suptitle('train and validation result')
    for i in range(18):
        ax = axs[i//n_cols, i%n_cols]
        ax.plot(np.arange(args.epochs), train_mean_list[i], color=color_list[i], linestyle='-')
        ax.plot(np.arange(args.epochs), val_mean_list[i], color=color_list[i], linestyle='--')
        ax.set_title(title_list[i])
    plt.tight_layout()
    plt.savefig(args.output_dir / 'f1_transition.png')
    plt.close()




# nested-CV ================================================================
## pre-processing: splitting outer dataframe --------------------------------
args = Args()
df = create_dataset(args)


## outer train-val dividion
splitter = MultilabelStratifiedKFold(n_splits=args.outer_cvs, shuffle=True, random_state=42)
X = df.text.values
y = np.array(df.label.tolist())
train_index, val_index = next(splitter.split(X, y))
train_df = pd.DataFrame({'text': X[train_index], 'label': y[train_index].tolist()})
val_df = pd.DataFrame({'text': X[val_index], 'label': y[val_index].tolist()})
pkl_saver(train_df, os.path.join(args.abs_path, 'results', 'train_val_df_list_20240602', 'train_df'))
pkl_saver(val_df, os.path.join(args.abs_path, 'results', 'train_val_df_list_20240602', 'val_df'))



def optimize_and_run(args):
    ### data loader
    train_df = pkl_loader(os.path.join(args.abs_path, 'results', 'train_val_df_list_20240602', 'train_df'))
    val_df = pkl_loader(os.path.join(args.abs_path, 'results', 'train_val_df_list_20240602', 'val_df'))

    ### set study and run
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.HyperbandPruner(),
                                study_name='optuna_study', storage='sqlite:///{}/study_storage.db'.format(args.output_dir))
    study.enqueue_trial({'batch_size':16,
                        'encoder_lr':1.0e-4,
                        'pooler_lr':1.0e-4,
                        'cls_lr':1.0e-4,
                        'pooler_dropout':0.3,
                        'cls_dropout':0.3,
                        'num_layer':5})
    study.optimize(objective_variable(args, train_df), n_trials=args.n_trials)

    ### save optimize result
    save_json(study.best_params, args.output_dir / 'study_best_params.json')
    study.trials_dataframe().to_csv(args.output_dir / 'study_trials_df.csv')
    optuna.visualization.plot_slice(study).write_html(args.output_dir / 'study_slice_plot.html')
    optuna.visualization.plot_param_importances(study).write_html(args.output_dir / 'study_param_importances.html')
    optuna.visualization.plot_optimization_history(study).write_html(args.output_dir / 'study_optimization_history.html')
    optuna.visualization.plot_contour(study).write_html(args.output_dir / 'study_contour_plot.html')

    ### set best parameter to outer setting
    args.batch_size = study.best_params['batch_size']
    args.encoder_lr = study.best_params['encoder_lr']
    args.pooler_lr = study.best_params['pooler_lr']
    args.cls_lr = study.best_params['cls_lr']
    args.pooler_dropout = study.best_params['pooler_dropout']
    args.cls_dropout = study.best_params['cls_dropout']
    args.num_layer = study.best_params['num_layer']

    ## run outer
    outer_results = {'total_loss': list(), 'true_list': list(), 'prob_list': list(), 'pred_list': list()}
    dl_dict = {'train': create_dataloader(args, train_df, shuffle=True), 
                'val': create_dataloader(args, val_df, shuffle=False)}
    net = sdgs_net(args).to(args.device, non_blocking=True)
    net = net_initializer(args, net)
    # training tools --------------------------------------
    criterion = create_criterion(args, train_df)
    optimizer, lr_scheduler = create_optimizer(args, net, dl_dict['train'])
    scaler = torch.cuda.amp.GradScaler()
    # training --------------------------------------------
    for epoch in trange(args.epochs, dynamic_ncols=True):
        torch.cuda.empty_cache()
        for phase in ['train', 'val']:
            # set net mode and select criterion -----------
            if phase == 'train':
                net.train()
            elif phase == 'val':
                net.eval()
            # run ---------------------------------------
            net, metrics, result = run_one_epoch(args, phase, net, dl_dict[phase], criterion, optimizer, lr_scheduler, scaler)
            # final epoch -> save validation prediction result -----
            if (epoch == args.epochs-1) and (phase == 'val'):
                for name in result.keys():
                    outer_results[name].extend(result[name])
            # show metrics ----------------------------------------
            metrics = {'phase': phase, 'epoch': epoch, **metrics}
            log(args, metrics)
    # save outer-cv-result
    pkl_saver(outer_results, args.output_dir / 'outer_results')
    classification_df = classification_report(outer_results['true_list'], outer_results['pred_list'], output_dict=True)
    classification_df = pd.DataFrame(classification_df).T
    classification_df.to_csv(args.output_dir / 'classification_report.csv')


def load_best_params(args):
    study = optuna.load_study(study_name='optuna_study', storage='sqlite:///{}/study_storage.db'.format(args.output_dir))
    args.batch_size = study.best_params['batch_size']
    args.encoder_lr = study.best_params['encoder_lr']
    args.pooler_lr = study.best_params['pooler_lr']
    args.cls_lr = study.best_params['cls_lr']
    args.pooler_dropout = study.best_params['pooler_dropout']
    args.cls_dropout = study.best_params['cls_dropout']
    args.num_layer = study.best_params['num_layer']
    return args


## run inner_cv_ones
args = Args()
args.device = torch.device("cuda:0")
args.model_name = 'roberta'
args.model_path = 'roberta-large'
args.process_args()
save_readable_config(args, args.output_dir / 'config_readable.json')
optimize_and_run(args)
args = load_best_params(args)
generate_best_model(args)


args = Args()
args.device = torch.device("cuda:1")
args.model_name = 'deberta'
args.model_path = 'microsoft/deberta-v3-large'
args.process_args()
save_readable_config(args, args.output_dir / 'config_readable.json')
optimize_and_run(args)
generate_best_model(args)

"""
Trial54 failed with parameters / maybe because of batch_size:32 -> out of memory
"""
args.hypara_cand = {
        'batch_size': [4, 31], # max: 32-> 31に変更
        'encoder_lr': [1e-5, 1e-3],
        'pooler_lr': [1e-5, 1e-3],
        'cls_lr': [1e-5, 1e-3],
        'pooler_dropout': [0.0, 0.5],
        'cls_dropout': [0.0, 0.5],
        'num_layer': [1, 10]
        }


def optimize_and_run_next(args):
    ### data loader
    train_df = pkl_loader(os.path.join(args.abs_path, 'results', 'train_val_df_list_20240602', 'train_df'))
    val_df = pkl_loader(os.path.join(args.abs_path, 'results', 'train_val_df_list_20240602', 'val_df'))

    ### set study and run
    study = optuna.load_study(study_name='optuna_study', storage='sqlite:///{}/study_storage.db'.format(args.output_dir))
    study.optimize(objective_variable(args, train_df), n_trials=args.n_trials)

    ### save optimize result
    save_json(study.best_params, args.output_dir / 'study_best_params.json')
    study.trials_dataframe().to_csv(args.output_dir / 'study_trials_df.csv')
    optuna.visualization.plot_slice(study).write_html(args.output_dir / 'study_slice_plot.html')
    optuna.visualization.plot_param_importances(study).write_html(args.output_dir / 'study_param_importances.html')
    optuna.visualization.plot_optimization_history(study).write_html(args.output_dir / 'study_optimization_history.html')
    optuna.visualization.plot_contour(study).write_html(args.output_dir / 'study_contour_plot.html')

    ### set best parameter to outer setting
    args.batch_size = study.best_params['batch_size']
    args.encoder_lr = study.best_params['encoder_lr']
    args.pooler_lr = study.best_params['pooler_lr']
    args.cls_lr = study.best_params['cls_lr']
    args.pooler_dropout = study.best_params['pooler_dropout']
    args.cls_dropout = study.best_params['cls_dropout']
    args.num_layer = study.best_params['num_layer']

    ## run outer
    outer_results = {'total_loss': list(), 'true_list': list(), 'prob_list': list(), 'pred_list': list()}
    dl_dict = {'train': create_dataloader(args, train_df, shuffle=True), 
                'val': create_dataloader(args, val_df, shuffle=False)}
    net = sdgs_net(args).to(args.device, non_blocking=True)
    net = net_initializer(args, net)
    # training tools --------------------------------------
    criterion = create_criterion(args, train_df)
    optimizer, lr_scheduler = create_optimizer(args, net, dl_dict['train'])
    scaler = torch.cuda.amp.GradScaler()
    # training --------------------------------------------
    for epoch in trange(args.epochs, dynamic_ncols=True):
        torch.cuda.empty_cache()
        for phase in ['train', 'val']:
            # set net mode and select criterion -----------
            if phase == 'train':
                net.train()
            elif phase == 'val':
                net.eval()
            # run ---------------------------------------
            net, metrics, result = run_one_epoch(args, phase, net, dl_dict[phase], criterion, optimizer, lr_scheduler, scaler)
            # final epoch -> save validation prediction result -----
            if (epoch == args.epochs-1) and (phase == 'val'):
                for name in result.keys():
                    outer_results[name].extend(result[name])
            # show metrics ----------------------------------------
            metrics = {'phase': phase, 'epoch': epoch, **metrics}
            log(args, metrics)
    # save outer-cv-result
    pkl_saver(outer_results, args.output_dir / 'outer_results')
    classification_df = classification_report(outer_results['true_list'], outer_results['pred_list'], output_dict=True)
    classification_df = pd.DataFrame(classification_df).T
    classification_df.to_csv(args.output_dir / 'classification_report.csv')

optimize_and_run_next(args)

"""
Trial64 failed with parameters / maybe because of batch_size:31 -> out of memory
"""
args.device = torch.device("cuda:2")
args.hypara_cand = {
        'batch_size': [4, 30], # max: 31-> 30に変更
        'encoder_lr': [1e-5, 1e-3],
        'pooler_lr': [1e-5, 1e-3],
        'cls_lr': [1e-5, 1e-3],
        'pooler_dropout': [0.0, 0.5],
        'cls_dropout': [0.0, 0.5],
        'num_layer': [1, 10]
        }

optimize_and_run_next(args)




args = Args()
args.device = torch.device("cuda:3")
args.model_name = 'luke'
args.model_path = 'studio-ousia/luke-large-lite'
args.process_args()
save_readable_config(args, args.output_dir / 'config_readable.json')
optimize_and_run(args)
generate_best_model(args)








# validation with osdg dataset ---------------------------
def int_to_onehot(sdg):
        vector = np.zeros(17, dtype=int)
        vector[sdg -1] = 1
        return vector


def validation_with_osdg(args):
    osdg = pd.read_csv(os.path.join(args.abs_path, 'data','osdg-community-data-v2024-01-01.csv'), sep='\t')
    osdg = pd.DataFrame({'text':osdg.text, 'sdg':osdg.sdg, 'agreement':osdg.agreement})
    osdg = osdg[osdg['agreement'] > 0.99].reset_index(drop=True)
    osdg['label'] = osdg['sdg'].apply(int_to_onehot)
    osdg_predict_result = predicator(args, osdg.text.to_list())
    pkl_saver(osdg_predict_result, args.output_dir / 'osdg_predict_result')
    osdg_trues = osdg['label'].to_list()
    osdg_trues = [x[:16] for x in osdg_trues]
    osdg_preds = osdg_predict_result['preds']
    osdg_preds = [x[:16] for x in osdg_preds]
    classification_df = classification_report(osdg_trues, osdg_preds, output_dict=True)
    classification_df = pd.DataFrame(classification_df).T
    classification_df = classification_df.iloc[:17, :3]
    classification_df.to_csv(args.output_dir / 'osdg_classification_report.csv')
    # viz -----------------------------------------------
    fig, ax = plt.subplots(figsize=(7,10))
    sns.heatmap(classification_df, linewidth=0.1, ax=ax, cmap=sns.diverging_palette(20,220,n=200),
                vmin=0, vmax=1, center=0.5,
                annot=True, fmt='.3f', cbar=False, square=False, annot_kws={'size':16})
    plt.xticks(ticks=np.arange(3)+0.5, labels=['precision', 'recall', 'f1-score'], color='black', fontsize=16)
    plt.yticks(ticks=np.arange(17)+0.5, labels=args.goal_names[:-1]+['micro_avg'], rotation=0, color='black', fontsize=20)
    plt.tight_layout()
    # plt.show()
    plt.savefig(args.output_dir / 'validation_with_osdg.png', transparent=False)
    plt.close()

validation_with_osdg(args)















# training data investigation ---------------------------
## 1. count by goal
df = create_dataset(args)
label_count = [0]*17
for i, item in df.iterrows():
    vec = item.label
    for j in range(len(label_count)):
        if vec[j] == 1:
            label_count[j] += 1
label_percent = [count/len(df) for count in label_count]

## 2. viz
fig, ax = plt.subplots(figsize=(9,11))
ax.barh(y = args.goal_names, width=label_percent, color=args.sdgs_colors)
for index, value in enumerate(label_percent):
    ax.text(value+0.002, index, '{:.1%}'.format(value), va='center', ha='left', fontweight='bold', fontsize=16)
ax.set_yticklabels(labels=args.goal_names, color='black', fontsize=16)
ax.invert_yaxis()
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().tick_params(axis='y', which='both', left=False)
plt.tight_layout()
# plt.show()
plt.savefig(args.output_dir / 'training_data_distribution.png', transparent=False)
plt.close()

## 3. token num
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
df['token_length'] = df['text'].apply(lambda x: measure_token_length(x, tokenizer))
pd.DataFrame(df['token_length'].describe()).to_csv(args.output_dir / 'training_data_token_length.csv')

## 4. token num by goal
token_count = [0]*17
for idx, row in df.iterrows():
    for j in range(17):
        if row['label'][j] == 1:
            token_count[j] += row['token_length']
token_mean = np.array(token_count) / np.array(label_count)
training_data_info = pd.DataFrame({'count': label_count, 'percent': label_percent, 'token_per_document': token_mean},
                                  index=args.goal_names)
training_data_info.to_csv(args.output_dir / 'training_data_info.csv')


# get training data nums (For Japanese Model) ----------------------
ja_info = pd.read_csv(os.path.join(args.abs_path, 'data', 'Japanese_corpus', 'each_corpus_num.csv'))

lengthes = []
UN_JA_df_path = os.path.join(args.abs_path, 'data', 'UN_SDG_Actions_Platform', 'un_sdg_actions_platform_ja_20240122')
UN_df = pkl_loader(UN_JA_df_path)
lengthes.append(len(UN_df))
IISD_df_path = os.path.join(args.abs_path, 'data', 'IISD_SDG_Knowledge_Hub', 'IISD_SDG_Knowledge_Hub_JA_20240428')
IISD_df = pkl_loader(IISD_df_path)
lengthes.append(len(IISD_df))
sdg_progress_path = os.path.join(args.abs_path, 'data', 'un_sdg_progress', 'un_sdg_progress_jpn')
sdg_progress = pkl_loader(sdg_progress_path)
lengthes.append(len(sdg_progress))
sdgs_difinition_path = os.path.join(args.abs_path, 'data', 'sdgs_difinition', 'sdgs_difinition_jpn')
sdgs_difinition = pkl_loader(sdgs_difinition_path)
lengthes.append(len(sdgs_difinition))
en_info = pd.DataFrame({
    'file_name': ['UN_SDG_Actions_Platform', 'IISD_SDG_Knowledge_Hub', 'UN_SDG_Progress_report', 'SDG_difinition'],
    'num': lengthes
})

corpus_info = pd.concat([ja_info, en_info]).reset_index(drop=True)
corpus_info.to_csv(os.path.join(args.output_dir, 'corpus_info.csv'), index=False)


# Academic Article Analyzer -----------------------------------------
## 1. download
path_to_data = os.path.join(args.abs_path, 'data', 'scopus_20230828-29')
csv_files = [file for file in os.listdir(path_to_data) if file.endswith(".csv")]
aca_texts = []
aca_dfs = []
for file in csv_files:
    file_path = os.path.join(path_to_data, file)
    df = pd.read_csv(file_path)
    aca_dfs.append(df)

aca_df = pd.concat(aca_dfs, ignore_index=True)
aca_df = aca_df[aca_df['Abstract'] != '[No abstract available]'].reset_index()
aca_df['Abstract'] = aca_df['Abstract'].apply(lambda x: x[:x.find('©')])
aca_df['text'] = [str(title) + ' ' + str(abst) for title, abst in zip(aca_df['Title'], aca_df['Abstract'])]
pkl_saver(aca_df, 'aca_df')
print(aca_df.info())


## 2. predicate
aca_result = predicator(args, aca_df['text'].tolist())
pkl_saver(aca_result, 'aca_result')


## 3. goal count
def count_goal(dat):
    label_count = [0 for i in range(len(dat[0]))]
    for vec in dat:
        for j in range(len(label_count)):
            if vec[j] == 1:
                label_count[j] += 1
    return label_count

aca_goal_count = count_goal(aca_result['preds'])
aca_goal_percent = np.array(aca_goal_count) / len(aca_df)

fig, ax = plt.subplots(figsize=(12,7))
bars = ax.bar(x=args.goal_names, height=aca_goal_percent, color=args.sdgs_colors)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x()+bar.get_width() / 2., 1.05*height, '{:.3f}'.format(height), ha='center', va='center')
ax.set_ylim(0, 0.40)
ax.set_xticklabels(['{}'.format(i+1) for i in range(17)])
ax.set_title('Global-Goals in Academic Articles')
ax.set_xlabel('goal')
ax.set_ylabel('proportion')
plt.legend()
plt.tight_layout()
plt.savefig(args.output_dir / 'aca_goal_percent.png')
plt.close()


## 4. Goal count in each Year
year_list = list(range(2010,2024))
goal_counts_by_year = []
for year in year_list:
    fillter = [aca_year == year for aca_year in aca_df.Year]
    preds = [pred for i, pred in enumerate(aca_result['preds']) if fillter[i]]
    goal_count = np.array(count_goal(preds)) / len(preds)
    goal_counts_by_year.append(goal_count)

re_group = []
for g in range(17):
    goal_transition = []
    for y in range(len(year_list)):
        goal_transition.append(goal_counts_by_year[y][g])
    re_group.append(goal_transition)

fig, ax = plt.subplots(figsize=(12,7))
for g in range(len(re_group)):
    ax.plot(year_list, re_group[g], label=args.goal_names[g], color=args.sdgs_colors[g], lw=3)
ax.set_xlabel('Year')
ax.set_ylabel('proportion')
ax.legend(fontsize='large', markerscale=15, frameon=False, bbox_to_anchor=(1,1)) # bbox_to_anchor=(1,1)
# plt.title('Change in Attention of Goal')
plt.tight_layout()
# plt.show()
plt.savefig(args.output_dir / 'goal_count_in_each_year.png',transparent=False)
plt.close()



## 5. target level nexus assessment
result_of_target = predicate_of_target(args)
aca_target_cossim = calculate_cossim(aca_result['probs'], result_of_target['probs'])
print(np.median(aca_target_cossim))
print(np.percentile(aca_target_cossim, 75))
print(np.percentile(aca_target_cossim, 80))
print(np.percentile(aca_target_cossim, 85))
print(np.percentile(aca_target_cossim, 90))

aca_target_preds = (aca_target_cossim > 0.6).int()
aca_target_preds_sparse = scipy.sparse.csr_matrix(aca_target_preds) # make it sparse for caliculation efficiency
aca_target_cooccur = aca_target_preds_sparse.T.dot(aca_target_preds_sparse)
aca_target_cooccur.setdiag(0)
aca_target_cooccur.eliminate_zeros()

dense = aca_target_cooccur.toarray()
coocur_thred = np.percentile(dense, 90)
G = nx.Graph()
for i in range(169):
    G.add_node(result_of_target['target_code'][i], color=result_of_target['color'][i])
    for j in range(i+1, 169):
        weight = aca_target_cooccur[i,j]
        if weight > coocur_thred:
            G.add_edge(result_of_target['target_code'][i], result_of_target['target_code'][j], weight=weight)
isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)
G.remove_nodes_from(nx.selfloop_edges(G))

plt.figure(figsize=(12,7))
pos = nx.spring_layout(G, k=15, seed=9) # seed good candidate: 12 and k = 8
node_colors = [G.nodes[node]['color'] for node in G.nodes()]
pr = nx.pagerank(G)
node_size = [(1200*v)**1.5+100 for v in pr.values()]
edge_width = [(d["weight"]/100000)**2 for (u,v,d) in G.edges(data=True)]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8, node_size=node_size)
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.8, width=edge_width)
nx.draw_networkx_labels(G, pos, # {i: list(result_of_target['target_code'])[i] for i in range(169)},
                        font_size=10, font_color='black', font_family='SF Pro')
plt.axis('off')
plt.tight_layout()
# plt.show()
plt.savefig(args.output_dir / 'aca_target_network.png', transparent=False)
plt.close()





## 6. target-co-occurrence-heatmap
aca_target_preds = (aca_target_cossim > 0.6).int()
aca_target_preds = torch.tensor(aca_target_preds)
aca_target_cooccur = torch.matmul(aca_target_preds.t(), aca_target_preds)
aca_target_cooccur = aca_target_cooccur / len(aca_df)

mask = np.triu(np.ones_like(aca_target_cooccur))

fig, axs = plt.subplots(figsize=(18,19))
# fig.suptitle('Goal Co-occurrence in Academic Articles')
sns.heatmap(np.array(aca_target_cooccur), linewidth=0.1, cmap=sns.diverging_palette(20,220,n=200), ax=axs,
            vmin=0.0, vmax=aca_target_cooccur.max(), center=np.median(aca_target_cooccur),
            mask = mask, cbar=False
            )
plt.xticks(ticks=np.arange(169)+0.5, labels=result_of_target['target_code'], fontsize=8, color='black')
plt.yticks(ticks=np.arange(169)+0.5, labels=result_of_target['target_code'], fontsize=7, color='black')
plt.tick_params(axis='both', which='both',bottom=False, top=False, left=False, right=False)
plt.tight_layout()
# plt.show()
plt.savefig(args.output_dir / 'aca_target__co-occurrence.png', transparent=False)
plt.close()










# Classification Report Viz for PPT -----------------------------
log_df = pd.read_csv(args.output_dir / 'cv_classification_report.csv')
f1_scores = log_df['f1-score'].tolist()[:17]
micro_avg = log_df.loc[17, 'f1-score']
fig, ax = plt.subplots(figsize=(12*1.5, 8*1.5))
ax.barh(y = args.goal_names, width=f1_scores, color=args.sdgs_colors)
for index, value in enumerate(f1_scores):
    ax.text(value+0.002, index, '{:.2f}'.format(value), va='center', ha='left', fontweight='bold', fontsize=22)
ax.set_yticklabels(labels=args.goal_contents, color='black', fontsize=22)
ax.axvline(micro_avg, color='gray', label='micro-avg = {:.3f}'.format(micro_avg), linestyle='--', lw=3, alpha=0.5)
ax.invert_yaxis()
ax.set_xlim(0,1)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().tick_params(axis='y', which='both', left=False)
plt.tight_layout()
# plt.show()
plt.savefig(args.output_dir / 'classification_report_ppt_long_ver.png', transparent=False)
plt.close()


# f1-score and training_size scatter viz for PPT ---------------
goal_pathes = [os.path.join(args.abs_path, f'images/sdgs_goal/transparent_goal_icon_mini/SDG_Icons_Inverted_Transparent_WEB-{str(i).zfill(2)}.png')
               for i in range(1, 18)]

fig, ax = plt.subplots(figsize=(12*1.5, 8*1.5))
for i, (x, y) in enumerate(zip(label_count, f1_scores)):
    ab = AnnotationBbox(OffsetImage(plt.imread(goal_pathes[i]), zoom=0.08),
                        (x, y), frameon=False)
    ax.add_artist(ab)
ax.set_xlim(min(label_count)*0.9, max(label_count)*1.1)
ax.set_ylim(min(f1_scores)*0.9, max(f1_scores)*1.1)
plt.tight_layout()
# plt.show()
plt.savefig(args.output_dir / 'training_size_f1_score_scatter.png', transparent=False)
plt.close()


# f1-score and co-occurence size viz for PPT ------------------
cooccur_sizes = []
for i in range(17):
    fillter = [ls[i]==1 for ls in df.label]
    df_i = df[fillter].reset_index(drop=True)
    label_count_i = [0]*17
    for i, item in df_i.iterrows():
        vec = item.label
        for j in range(len(label_count_i)):
            if vec[j] == 1:
                label_count_i[j] += 1
    cooccur_sizes.append(sum(label_count_i)/len(df_i))

fig, ax = plt.subplots(figsize=(12*1.5, 8*1.5))
for i, (x, y) in enumerate(zip(cooccur_sizes, f1_scores)):
    ab = AnnotationBbox(OffsetImage(plt.imread(goal_pathes[i]), zoom=0.08),
                        (x, y), frameon=False)
    ax.add_artist(ab)
ax.set_xlim(min(cooccur_sizes)*0.9, max(cooccur_sizes)*1.1)
ax.set_ylim(min(f1_scores)*0.9, max(f1_scores)*1.1)
plt.tight_layout()
# plt.show()
plt.savefig(args.output_dir / 'cooccur_size_f1_score_scatter.png', transparent=False)
plt.close()



# confusion matrix viz -----------------------------------
## load cross varidation result
cv_result = pkl_loader(args.output_dir / 'outer_cv_results')
true_list = cv_result['true_list']
pred_list = cv_result['pred_list']

# create confusion matrix 
conf_mtx = []
for g in range(17):
    # get indices whose answer is Goal-i by each goal (Goal i が正解であるサンプル番号をゴールごとにリストで取得する)
    indices = []
    for idx in range(len(true_list)):
        if true_list[idx][g] == 1:
            indices.append(idx)
    # check prediction
    pre_ls = [0]*17
    for idx in indices: # load index whose answer is goal-g
        if pred_list[idx][g] != 1: # if this prediction fail to answer goal-g
            for h in range(17):
                pre_ls[h] += pred_list[idx][h] / len(indices) # count which goal was misclassidied
    conf_mtx.append(pre_ls)
conf_mtx = np.array(conf_mtx)

# viz as heatmap
mask = np.eye(17, dtype=bool) # 対角成分をマスク
fig, ax = plt.subplots(figsize=(12*1.5,7*1.5))
sns.heatmap(conf_mtx, linewidth=0.1, ax=ax, cmap='RdBu_r',
            vmin=conf_mtx.min(), vmax=conf_mtx.max(), center=np.median(conf_mtx),
            annot=True, fmt='.2f', cbar=False, square=False, mask=mask, linewidths=0)
for i in range(18):
    ax.axhline(i, color='white', lw=8)
plt.xticks(ticks=np.arange(17)+0.5, labels=['G {}'.format(i) for i in range(1, 18)], color='black', fontsize=16)
plt.yticks(ticks=np.arange(17)+0.5, labels=args.goal_contents, rotation=0, color='black', fontsize=20)
plt.tight_layout()
# plt.show()
plt.savefig(args.output_dir / 'true_pred_mtx_map.png', transparent=False)
plt.close()



# LIME ++++++++++++++++++++++++++++++++++++++++
args = Args()
args.process_args()


def predicator_for_lime(text_list):
    # data -------------------
    analyze_data = pd.DataFrame({'text': text_list})
    analyze_data['label'] = [[0]*17 for _ in range(len(analyze_data))]
    #!!!!! data loader のshuffleに注意！推論時にはFalseにしなければならない !!!!!!!!!!!!!!
    loader_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': args.num_workers, 'pin_memory': True}
    analyze_data_set = CustomDataset(args, analyze_data)
    analyze_dl = DataLoader(analyze_data_set, **loader_params)
    # result box -------------
    probs = []
    # predicate
    for batch in tqdm(analyze_dl, total=len(analyze_dl), dynamic_ncols=True, leave=True):
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else None):
                output, vec, attention = best_net(**batch.to(args.device))
        prob = torch.sigmoid(output)
        probs.append(prob.cpu().detach().to(torch.float32).numpy().tolist())        
    return np.vstack(probs)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb_color):
    return '#{:02x}{:02x}{:02x}'.format(*rgb_color)


def mix_colors(colors, weights):
    assert len(colors) == len(weights)
    rgb_colors = [hex_to_rgb(color) for color in colors]
    weighted_sum = np.dot(weights, rgb_colors) /sum(weights) # How???
    weighted_sum = np.clip(weighted_sum, 0, 255).astype(int)
    return rgb_to_hex(tuple(weighted_sum))


def run_lime(args, text_list):
    # set utils -------------------------------------------------------------
    lime_result = {'probs': [], 'dicts': [], 'tokens':[], 'preds': []}
    stop_words = pd.read_csv(os.path.join(args.abs_path, 'utils', 'Japanese.txt'), header=None)[0].to_list()
    stop_words_misc = ['「','」','、','。','(',')','・','を','で','し','へ','に','て','の','も','や']
    # iteration --------------------------------------------------------------
    for text in tqdm(text_list):
        # preprocessing ------------------------------------------------------
        text = unicodedata.normalize('NFKC', text) # これがないとLIMEは動かない==全角文字には非対応
        # predicate ----------------------------------------------------------
        prob = predicator_for_lime([text])
        prob = prob.reshape(-1)
        prob = prob.tolist()
        # run LIME -----------------------------------------------------------
        explainer = LimeTextExplainer(class_names=args.goal_names,
                                      split_expression=tokenizer.tokenize,
                                      mask_string=[tokenizer.pad_token]+stop_words+stop_words_misc,
                                      random_state=42)
        num_features = int(len(set(tokenizer.tokenize(text)))*0.15)
        exp = explainer.explain_instance(text_instance=text,
                                        classifier_fn=predicator_for_lime,
                                        labels=[i for i in range(17)],
                                        num_features=num_features,
                                        num_samples=20000)
        # make Contributional-TOKEN and Color Dictionary -------------------
        positive_contributions_list = []
        colors = []
        ## word-value dict for each predicted goal
        for idx, p in enumerate(prob):
            if p > args.thred:
                contributions = exp.as_list(label=idx)
                positive_contributions = {}
                for items in contributions:
                    token, value = items[0], items[1]
                    if value > 0:
                        positive_contributions[token] = value
                positive_contributions_list.append(positive_contributions)
                colors.append(args.sdgs_colors[idx])
        ## extend to one dict
        entire_dict = {}
        for idx, dict in enumerate(positive_contributions_list):
            for key, value in dict.items():
                if key not in entire_dict.keys():
                    lst = [0]*len(positive_contributions_list)
                else:
                    lst = entire_dict[key]
                lst[idx] = value
                entire_dict[key] = lst
        ## mix color 
        for key, value in entire_dict.items():
            entire_dict[key] = mix_colors(colors, value)
        # in
        lime_result['probs'].append(prob)
        lime_result['dicts'].append(entire_dict)
        lime_result['tokens'].append(tokenizer.tokenize(text))
        lime_result['preds'].append((np.array(prob) > args.thred).astype(int))
    return lime_result


def make_html_with_lime(args, lime_result, file_name='lime_viz.html'):
    target_dict = predicate_of_target(args)
    cossim = calculate_cossim(lime_result['probs'], target_dict['probs'])
    related_targets = []
    for i, preds in enumerate(lime_result['preds']):
        related_targets.append(select_target_and_goal(target_dict, preds, cossim[i]))
    ## make html
    length = len(lime_result['probs'])
    html = '''
    <font face="Arial">
    <head>
        <style>
            .highlight {
                background-color: #f0f0f0;
                padding: 5px;
            }
        </style>
    </head>
    '''
    # toc ------------------------------------------------------------
    html += '<div class="toc"><ul>'
    for i in range(length):
        html += '<li><a href="#item{}">Item {}</a></li>'.format(i,i+1)
    html += '</ul></div>'
    # contents -----------------------------------------------------
    for idx in trange(length):
        # load one result ------
        tokens = lime_result['tokens'][idx]
        dict = lime_result['dicts'][idx]
        prob = lime_result['probs'][idx]
        # write ---------------
        html += f'<h2 id="item{idx}">Item {idx+1}</h2><p>'
        html += '<h3>テキスト</h3>'
        for token in tokens:
            if token in dict.keys():
                html_color = dict[token] # '#%02X%02X%02X' % (255, int(255*(1 - 0.5)), int(255*(1 - 0.5)))
                html_color += '99' # 透明度を16進数で指定
                html += '<span style="background-color: {}">{}</span>'.format(html_color, token)
            else:
                html += token
        # goal and target ------------------------------------------
        html += '<h3>予測結果</h3>'
        goal_target = related_targets[idx]
        for goal_content, targets in goal_target.items():
            html += f'<h4><span class="highlight">{goal_content}</span></h4>'
            for target in targets:
                html += target
                html += '<br>'
            html += '<br>'
        html += '</p><hr>'
    html += '</font>'
    # save -----------------------------------------------------------
    with open(args.output_dir / file_name, 'w', encoding='UTF-8') as res:
        res.write(html)


best_net = sdgs_net(args).to(args.device, non_blocking=True)
best_net = net_initializer(args, best_net)
best_net.load_state_dict(torch.load(args.output_dir / 'best_model.pt'))
best_net.eval()
tokenizer = AutoTokenizer.from_pretrained(args.model_path, do_subword_tokenize=False)

text = """
大阪大学は、地域から世界全体に及ぶさまざまな課題を解決し、「生きがいを育む社会」を創造する大学として、性別、SOGI（性的指向、性自認）、障がいの有無、国籍、民族、文化的背景、年齢等の違いを超えた、真に多様性を活かせる環境作りに取り組んでいます。

その中で、大阪大学は、「知の創造、継承及び実践」を使命とし、「地域に生き世界に伸びる」をモットーに、学問の独立性と市民性を備えた世界水準の高度な教育研究を推進し、次代の社会を支え、人類の理想の実現をはかる有能な人材を社会に輩出することを教育における目的としています。

多様性こそがイノベーションの源泉の一つであると考えており、特に、女子学生や、その保護者、高校の教員などの皆さまに本学に興味を持ってもらうための取組みを「女子学生の教育体制の充実総合パッケージ」として積極的に促進してまいります。
"""
text_list = [text]
### https://www.osaka-u.ac.jp/ja/news/topics/2024/04/17001

lime_result = run_lime(args, text_list)
make_html_with_lime(args, lime_result, 'lime_viz.html')



sample = '私は歩く'
tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample))
tokenizer.encode_plus(sample)
tokenizer
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# word analysis



# Word based analysis ===================================================================
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, ColorSequenceRegistry
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.image as mping
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.font_manager as fm
from matplotlib.patheffects import withStroke
import matplotlib
from scipy.sparse import csr_matrix
from sklearn.metrics import (precision_recall_curve, auc, f1_score,
                             recall_score, 
                             silhouette_samples, silhouette_score)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import FastICA, PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict, Counter
import networkx as nx

df = create_dataset(args)

# 分かち書き
def wakatigaki_(text):
    stop_words = pd.read_csv(os.path.join(args.abs_path, 'utils', 'Japanese.txt'), header=None)[0].to_list()
    stop_words += ['する','できる', 'られる', 'なる', 'れる', 'せる']
    m = MeCab.Tagger(ipadic.MECAB_ARGS)
    p = m.parse(text)
    p_splits = [i.split('\t') for i in p.split('\n')][:-2]
    info = [x[1].split(',') for x in p_splits]
    lemma_words = []
    for x in info:
        if (x[0] in ['名詞', '動詞']) & (x[6] not in stop_words) & (x[6] not in ['', '*']):
            lemma_words.append(x[6])
    return ' '.join(lemma_words)


# tf-idf値を取得する
label_list = df.iloc[:2000, :]['label'].tolist()
text_list = df.iloc[:2000, :]['text'].apply(wakatigaki_).tolist()
vectorizer = TfidfVectorizer(smooth_idf=False, ngram_range=(1,1), binary=True)
tfidf_matrix = vectorizer.fit_transform(text_list)
feature_words = vectorizer.get_feature_names_out()
tfidf_dict_list = []
for doc_idx, doc in tqdm(enumerate(tfidf_matrix)):
    feature_index = doc.nonzero()[1] # docは次元数=語彙数のスパース行列
    tfidf_scores = zip(feature_index, [doc[0, x] for x in feature_index])
    tfidf_dict = {feature_words[i]: score for i, score in tfidf_scores}
    sorted_tfidf_dict = dict(sorted(tfidf_dict.items(), key=lambda item: item[1], reverse=True))
    top_half_count = int(len(sorted_tfidf_dict) / 2)
    top_half_tfidf = {word: score for word, score in list(sorted_tfidf_dict.items())[:top_half_count]}
    tfidf_dict_list.append(top_half_tfidf)
print(tfidf_dict_list[0])
print(text_list[0])


## wordcloud by goal ---------------------------------------------------
goal_tfidf = defaultdict(lambda: defaultdict(float))
for doc_idx, tfidf_dict in tqdm(enumerate(tfidf_dict_list)):
    categories = np.where(np.array(label_list[doc_idx]) == 1)[0]
    for category in categories:
        for word, score in tfidf_dict.items():
            goal_tfidf[category][word] += score

for category, tfidf_dict in goal_tfidf.items():
    doc_count = sum(1 for cat_vec in label_list if cat_vec[category]==1)
    for word in tfidf_dict:
        goal_tfidf[category][word] /= doc_count


for category in tqdm(goal_tfidf):
    dic = dict(goal_tfidf[category])
    cmap = LinearSegmentedColormap.from_list('custom_cmap', [args.sdgs_colors[category], 'gray'])
    wordcloud = WordCloud(width=int(1000), height=int(600), max_words=100,
                        prefer_horizontal = 1, # mode='RGBA', background_color=None, 
                        font_path='/home/ge/anaconda3/pkgs/pillow-10.1.0-py39had0adad_0/info/test/Tests/fonts/NotoSansJP-Regular.otf',
                        colormap=cmap, random_state=42)
    wordcloud.generate_from_frequencies(dic)
    filename = args.goal_names[category] + '.png'
    wordcloud.to_file(args.output_dir / 'wordcloud' / filename)


# 単語の共起ネットワークを作成する

for idx, text in enumerate(text_list):
    categories = np.where(np.array(label_list[doc_idx]) == 1)[0]
    for category in categories:
        for word, score in tfidf_dict.items():
            goal_tfidf[category][word] += score
goal_text_list = []
cooccurrences = Counter()
for text in text_list:
    words = text.split()
    for pair in itertools.combinations(sorted(set(words)), 2):
        cooccurrences[pair] += 1

np.max(np.array(cooccurrences.values()))
lst = list(cooccurrences.values())
lst = sorted(lst, reverse=True)
n = int(len(lst)*0.01)
lst[:100]

G = nx.Graph()
for (word1, word2), count in cooccurrences.items():
    if count > 100:
        G.add_node(word1)
        G.add_node(word2)
        G.add_edge(word1, word2, weight=count)

plt.figure(figsize=(20,20))
pos = nx.spring_layout(G, k=0.5)
nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', font_family='IPAexGothic')
plt.show()



goal_table = make_cluster_wordtable(goal_tfidf).sort_index()
goal_table.index = name_list
goal_table.to_csv(output_dir + '/top_words_in_each_goal.csv')






cluster_tfidf_dict = defaultdict(lambda :defaultdict(float))
for doc_idx, tfidf_dict in tqdm(enumerate(dict_list)):
    cluster_id = cluster_id_of_samples[doc_idx]
    for word, score in tfidf_dict.items():
        cluster_tfidf_dict[cluster_id][word] += score

for cluster_id, tfidf_dict in cluster_tfidf_dict.items():
    doc_count = sum(1 for n in set(cluster_id_of_samples) if n == cluster_id)
    for word in tfidf_dict:
        cluster_tfidf_dict[cluster_id][word] /= doc_count


## Caliculate each Articles TF-IDF using Title+Keywords ---------------------------------
if os.path.exists('tfidf_dict_list'):
    tfidf_dict_list = pkl_loader('tfidf_dict_list')
else:
    title_keywords = [title+' '+key for title, key in zip(aca_df['Title'].fillna(''), aca_df['Author Keywords'].fillna(''))]
    buff = pd.DataFrame({'label':aca_preds_plus, 'text':title_keywords})
    text_list = buff['text'].apply(one_text_preprocessing)

    vectorizer = TfidfVectorizer(smooth_idf=False, ngram_range=(1,1), binary=True, )
    tfidf_matrix = vectorizer.fit_transform(text_list)
    feature_words = vectorizer.get_feature_names_out()

    tfidf_dict_list = []
    for doc_idx, doc in tqdm(enumerate(tfidf_matrix)):
        feature_index = doc.nonzero()[1]
        tfidf_scores = zip(feature_index, [doc[0, x] for x in feature_index])
        tfidf_dict = {feature_words[i]: score for i, score in tfidf_scores}
        sorted_tfidf_dict = dict(sorted(tfidf_dict.items(), key=lambda item: item[1], reverse=True))
        top_half_count = int(len(sorted_tfidf_dict) / 2)
        top_half_tfidf = {word: score for word, score in list(sorted_tfidf_dict.items())[:top_half_count]}
        tfidf_dict_list.append(top_half_tfidf)

    pkl_saver(tfidf_dict_list, 'tfidf_dict_list')


## utils functions ----------------------------------------------------------------
def make_cluster_tfidf_dict(dict_list, cluster_id_of_samples):
    cluster_tfidf_dict = defaultdict(lambda :defaultdict(float))
    for doc_idx, tfidf_dict in tqdm(enumerate(dict_list)):
        cluster_id = cluster_id_of_samples[doc_idx]
        for word, score in tfidf_dict.items():
            cluster_tfidf_dict[cluster_id][word] += score
    
    for cluster_id, tfidf_dict in cluster_tfidf_dict.items():
        doc_count = sum(1 for n in set(cluster_id_of_samples) if n == cluster_id)
        for word in tfidf_dict:
            cluster_tfidf_dict[cluster_id][word] /= doc_count
    return cluster_tfidf_dict


def make_cluster_wordtable(cluster_tfidf_dict):
    sorted_dbscan_tfidf = []
    for cluster_id in cluster_tfidf_dict:
        dic = cluster_tfidf_dict[cluster_id]
        sorted_dic = sorted(dic.items(), key=lambda x:x[1], reverse=True)[:100]
        sorted_keys = [item[0] for item in sorted_dic]
        sorted_dbscan_tfidf.append(sorted_keys)
    table = pd.DataFrame(sorted_dbscan_tfidf,
                         index=cluster_tfidf_dict.keys(), 
                         columns=['top_{}'.format(i+1) for i in range(len(sorted_dbscan_tfidf[0]))])
    return table


def make_cluster_wordclouds(cluster_tfidf_dict, cmap=None, save_dir=None):
    if cmap == None:
        cmap = LinearSegmentedColormap.from_list('custom_cmap', [color_code_dict['blue'], color_code_dict['gray']])
    wordclouds = []
    for cluster_id in tqdm(cluster_tfidf_dict):
        dic = dict(cluster_tfidf_dict[cluster_id])
        wordcloud = WordCloud(width=int(1000), height=int(600), max_words=50,
                            mode='RGBA', background_color=None, prefer_horizontal = 1,
                            font_path=os.path.join(abs_path, 'utils/fonts/SFProFonts/Library/Fonts/SF-Pro.ttf'),
                            colormap=cmap, random_state=42)
        wordcloud.generate_from_frequencies(dic)
        wordclouds.append(wordcloud)
        if save_dir != None:
            filename = save_dir + '/cluster_id={}.png'.format(str(cluster_id))
            wordcloud.to_file(filename)
    return wordclouds




























