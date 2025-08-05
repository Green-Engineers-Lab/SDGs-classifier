# library
## basic ----------------------------
import os
import pickle
import pandas as pd
import numpy as np
import warnings
import json
import random
from tqdm import tqdm, trange
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import unicodedata
import ast
import re
import gc
## sklearn ------------------------
from sklearn.metrics import precision_recall_fscore_support, classification_report, f1_score
## decomposition and clustering
## natural language ---------------
from nltk.corpus import stopwords
import string
import spacy
# Load spacy model and stopwords
# This requires the user to run: python -m spacy download en_core_web_sm
spacy_nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))
## visualization -----------------
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors
## machine learning ---------------
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
## deep learning -------------------
import optuna
import torch
from torch import nn
import torchinfo
from contextlib import redirect_stdout
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers.tokenization_utils import BatchEncoding
from transformers.optimization import get_linear_schedule_with_warmup
from huggingface_hub import hf_hub_download
import transformers
import torch._dynamo
transformers.__version__ # 2025/02/17 4.47.1


# Intializing envs ========================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
warnings.simplefilter('ignore')
np.set_printoptions(precision=3)
torch._dynamo.config.suppress_errors = True


# args =======================================================
class Args():
    # notes ---------------------------------------
    file_name: str = ''
    note: str = ''
    previous_result = ''
    # envs -----------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed: int = 42
    # model ---------------------------------
    model_name: str = ''
    model_path: str = ''
    # hyper parameter -------------------
    batch_size: int = 16
    encoder_lr: float = 1e-4
    pooler_lr: float = 1e-4
    cls_lr: float = 1e-4
    pooler_dropout: float = 0.3
    gat_layer_lr: float = 1e-4
    gat_out_lr: float = 1e-4
    max_length: int = 512
    num_layer: int = 5
    smooth_eps: float = 0.1
    thred: float = 0.5
    # optuna ----------------------
    optuna_epochs: int = 8
    n_trials: int = 32
    hypara_cand: dict = {
        'batch_size': [8, 16, 32, 48],
        'encoder_lr': [1e-5, 1e-3],
        'pooler_lr': [1e-5, 1e-3],
        'cls_lr': [1e-5, 1e-3],
        'pooler_dropout': [0.0, 0.5],
        'num_layer': [1, 10]
        }
    # training ----------------------
    epochs: int = 8
    num_warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    num_workers: int = 0 # for dataloader
    # variables --------------------
    goal_names: list[str] = [f'Goal {i+1}' for i in range(17)]
    class_number: int = 17
    sdgs_colors: list[str] = ['#E5243B','#DDA63A','#4C9F38','#C5192D','#FF3A21','#26BDE2','#FCC30B','#A21942','#FD6925','#DD1367','#FD9D24','#BF8B2E','#3F7E44','#0A97D9','#56C02B','#00689D','#19486A']
    goal_contents: list[str] = ['Goal 1: No Poverty','Goal 2: Zero Hunger','Goal 3: Good Health and Well-being','Goal 4: Quality Education','Goal 5: Gender Equality','Goal 6: Clean Water and Sanitation','Goal 7: Affordable and Clean Energy','Goal 8: Decent Work and Economic Growth','Goal 9: Industry, Innovation and Infrastructure','Goal 10: Reduced Inequalities','Goal 11: Sustainable Cities and Communities','Goal 12: Responsible Consumption and Production','Goal 13: Climate Action','Goal 14: Life Below Water','Goal 15: Life on Land','Goal 16: Peace, Justice and Strong Institutions','Goal 17: Partnerships for the Goals']
    # path --------------------------------
    abs_path = Path(__file__).parent
    corpus_path = abs_path / 'data' / 'corpus.pkl' # Assuming corpus is in a 'data' subfolder
    # Add a stable path for the pretrained model
    pretrained_model_dir = abs_path / 'pretrained_model'  
    # Hugging Face Hub settings for automatic model download
    hf_repo_id: str = "GE-Lab/SDGs-classifier"
    hf_model_filename: str = "best_model.pt"
    # utils -------------------------------
    def process_args(self):
        if args.previous_result == '':
            now = datetime.now().strftime('%Y%m%d%H%M')
            save_dir = now + '_' + self.file_name
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
        return pickle.load(web)


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


def clone_state_dict(net) -> dict:
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
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def type_modify(x):
    if type(x) == str:
        return ast.literal_eval(x)
    else:
        return list(x)


# create dataset -------------------------------------------------
def create_dataset(args) -> pd.DataFrame:
    # load sdg corpus
    df = pkl_loader(args.corpus_path)
    df['text'] = df['text'].apply(process_text)
    df['label'] = df['label'].apply(type_modify)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def dataset_splitter(df, n_splits):
    splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = df.text.values
    y = np.array(df.label.tolist())
    train_index, test_index = next(splitter.split(X, y))
    train_df = pd.DataFrame({'text': X[train_index], 'label': y[train_index].tolist()})
    test_df = pd.DataFrame({'text': X[test_index], 'label': y[test_index].tolist()})
    return train_df, test_df


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
            padding='max_length',
            return_token_type_ids = True,
            truncation = True,
        )
        return BatchEncoding({
            'input_ids': torch.tensor(inputs.input_ids, dtype=torch.long),
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
        self.dropout = nn.Dropout(args.pooler_dropout)
        self.pooler = nn.Sequential(nn.Linear(in_features=self.bert.config.hidden_size, out_features=self.bert.config.hidden_size))
        self.tanh = nn.Tanh()
        self.cls = nn.Linear(in_features=self.bert.config.hidden_size, out_features=args.class_number)
    def forward(self, input_ids, attention_mask, token_type_ids, position, labels):
        bert_output = self.bert(input_ids, attention_mask, token_type_ids, position, output_attentions=True, output_hidden_states=True)
        average_hidden_state = (bert_output.last_hidden_state*attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        pooler_output = self.tanh(self.pooler(self.dropout(average_hidden_state)))
        logits = self.cls(pooler_output)
        return logits, average_hidden_state, bert_output.attentions


def net_initializer(args, net):
    for param in net.parameters():
        param.requires_grad = True
    for param in net.bert.encoder.layer[:net.bert.config.num_hidden_layers - args.num_layer].parameters():
        param.requires_grad = False
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
        {'params': [param for name, param in net.pooler.named_parameters() if not name in no_decay],
        'weight_decay': args.weight_decay, 
        'lr': args.pooler_lr},
        {'params': [param for name, param in net.pooler.named_parameters() if name in no_decay],
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
    return torch.nn.BCEWithLogitsLoss(pos_weight=weights)


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
        total_loss += loss.item() * batch['input_ids'].size(0)
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


def objective_variable(args, train_df, val_df):
    def optuna_optimize(trial):
        # parameter set for optuna --------------------
        opt_args = args
        # objective parameters --------------------------------
        opt_args.batch_size = trial.suggest_categorical('batch_size', args.hypara_cand['batch_size'])
        opt_args.encoder_lr = trial.suggest_loguniform('encoder_lr', args.hypara_cand['encoder_lr'][0], args.hypara_cand['encoder_lr'][1])
        opt_args.pooler_lr = trial.suggest_loguniform('pooler_lr', args.hypara_cand['pooler_lr'][0], args.hypara_cand['pooler_lr'][1])
        opt_args.cls_lr = trial.suggest_loguniform('cls_lr', args.hypara_cand['cls_lr'][0], args.hypara_cand['cls_lr'][1])
        opt_args.pooler_dropout = trial.suggest_float('pooler_dropout', args.hypara_cand['pooler_dropout'][0], args.hypara_cand['pooler_dropout'][1])
        opt_args.num_layer = trial.suggest_int('num_layer', args.hypara_cand['num_layer'][0], args.hypara_cand['num_layer'][1])
        args_dict = {'batch_size': opt_args.batch_size, 'encoder_lr': opt_args.encoder_lr, 'pooler_lr': opt_args.pooler_lr, 'cls_lr': opt_args.cls_lr,
                    'pooler_dropout': opt_args.pooler_dropout, 'num_layer': opt_args.num_layer}
        # data loader ----------------------------
        opt_args.num_workers = 0
        train_dl = create_dataloader(opt_args, train_df.reset_index(drop=True))
        val_dl = create_dataloader(opt_args, val_df.reset_index(drop=True))
        # training set initialize -------------------------------
        net = sdgs_net(opt_args).to(opt_args.device, non_blocking=True)
        net = net_initializer(opt_args, net)
        criterion = create_criterion(opt_args, train_df)
        optimizer, lr_scheduler = create_optimizer(opt_args, net, train_dl)
        scaler = torch.cuda.amp.GradScaler()
        # training ---------------------------------------------
        best_val_f1 = 0
        try:
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
                    log(args, metrics, args.output_dir / 'optimize_run/optuna_log.csv')
                # for pruner
                trial.report(best_val_f1, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        except:
            pass
        del net
        torch.cuda.empty_cache()
        gc.collect()
        return best_val_f1
    
    return optuna_optimize


def optimize_and_run(args, train_df, val_df, test_df):
    ### set study and run
    if not os.path.exists(os.path.join(args.output_dir, 'optimize_run')):
        os.makedirs(os.path.join(args.output_dir, 'optimize_run'))
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.HyperbandPruner(),
                                study_name='optuna_study', storage='sqlite:///{}/study_storage.db'.format(os.path.join(args.output_dir, 'optimize_run')))
    study.enqueue_trial({'batch_size':16,
                    'encoder_lr':1.0e-4,
                    'pooler_lr':1.0e-4,
                    'cls_lr':1.0e-4,
                    'pooler_dropout':0.3,
                    'num_layer':5})
    study.optimize(objective_variable(args, train_df, val_df), n_trials=args.n_trials)

    ### save optimize result
    save_json(study.best_params, args.output_dir / 'optimize_run/study_best_params.json')
    study.trials_dataframe().to_csv(args.output_dir / 'optimize_run/study_trials_df.csv')
    optuna.visualization.plot_slice(study).write_html(args.output_dir / 'optimize_run/study_slice_plot.html')
    optuna.visualization.plot_param_importances(study).write_html(args.output_dir / 'optimize_run/study_param_importances.html')
    optuna.visualization.plot_optimization_history(study).write_html(args.output_dir / 'optimize_run/study_optimization_history.html')
    optuna.visualization.plot_contour(study).write_html(args.output_dir / 'optimize_run/study_contour_plot.html')

    ### set best parameter to outer setting
    args.batch_size = study.best_params['batch_size']
    args.encoder_lr = study.best_params['encoder_lr']
    args.pooler_lr = study.best_params['pooler_lr']
    args.cls_lr = study.best_params['cls_lr']
    args.pooler_dropout = study.best_params['pooler_dropout']
    args.num_layer = study.best_params['num_layer']
    ## run outer
    outer_results = {'total_loss': list(), 'true_list': list(), 'prob_list': list(), 'pred_list': list()}
    train_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    dl_dict = {'train': create_dataloader(args, train_df, shuffle=True), 
                'val': create_dataloader(args, test_df, shuffle=False)}
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
            log(args, metrics, args.output_dir / 'optimize_run/run_log.csv')
    # save outer-cv-result
    pkl_saver(outer_results, args.output_dir / 'optimize_run/outer_results')
    classification_df = classification_report(outer_results['true_list'], outer_results['pred_list'], output_dict=True)
    classification_df = pd.DataFrame(classification_df).T
    classification_df.to_csv(args.output_dir / 'optimize_run/classification_report.csv')


def run_without_optimize(args, train_df, test_df):
    ### set study and run
    if not os.path.exists(os.path.join(args.output_dir, 'run')):
        os.makedirs(os.path.join(args.output_dir, 'run'))
    ## run outer
    outer_results = {'total_loss': list(), 'true_list': list(), 'prob_list': list(), 'pred_list': list()}
    dl_dict = {'train': create_dataloader(args, train_df, shuffle=True), 
                'val': create_dataloader(args, test_df, shuffle=False)}
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
            log(args, metrics, args.output_dir / 'run/run_log.csv')
    # save outer-cv-result
    pkl_saver(outer_results, args.output_dir / 'run/outer_results')
    classification_df = classification_report(outer_results['true_list'], outer_results['pred_list'], output_dict=True)
    classification_df = pd.DataFrame(classification_df).T
    classification_df.to_csv(args.output_dir / 'run/classification_report.csv')


def generate_best_model(args, df):
    # load dataset ---------------------------------------
    df = pd.DataFrame({'text':df.text, 'label':df.label})
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
def download_model_from_hf(args):
    # Create the local directory for the pretrained model if it doesn't exist
    args.pretrained_model_dir.mkdir(parents=True, exist_ok=True)
    local_model_path = args.pretrained_model_dir / args.hf_model_filename    
    if not local_model_path.exists():
        print(f"Model file not found locally. Downloading from {args.hf_repo_id}...")
        hf_hub_download(
            repo_id=args.hf_repo_id,
            filename=args.hf_model_filename,
            local_dir=args.pretrained_model_dir,
            local_dir_use_symlinks=False
        )
        print("Download complete.")        
    return local_model_path


def predicator(args, text_list: list[str]) -> dict:
    best_net = sdgs_net(args).to(args.device, non_blocking=True)
    best_net = net_initializer(args, best_net)
    # Download the model from HF Hub (if not present) and get the local path
    model_path = download_model_from_hf(args)    
    # Load the state dict from the downloaded model file
    best_net.load_state_dict(torch.load(model_path, map_location=args.device))
    # tokenizer loader ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # data -------------------
    analyze_data = pd.DataFrame({'text': text_list})
    analyze_data['label'] = [[0]*17 for _ in range(len(analyze_data))]
    analyze_dl = create_dataloader(args, analyze_data, shuffle=False) 
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
        results['vecs'].extend(vec.cpu().detach().to(torch.float32).numpy().tolist()) 
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


# utils for analysis ------------------------------------------
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


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Main Execution Block
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def run_prediction_example():
    """
    This is the main function to demonstrate how to use the pre-trained model for prediction.
    """
    # 1. Set up arguments
    args = Args()
    args.file_name = 'sdgs_classifier_prediction'
    args.model_name = 'luke'
    # NOTE: The model_path should be the BASE model used for fine-tuning.
    # The fine-tuned weights will be loaded by the predicator function.
    args.model_path = 'studio-ousia/luke-large-lite'
    args.process_args()
    # NOTE: The fine-tuned model weights ('best_model.pt') will be automatically 
    # downloaded from the Hugging Face Hub (GE-Lab/SDGs-classifier) on the first run 
    # and saved to the 'pretrained_model' directory
     
    # 2. Define texts to predict
    texts_to_predict = [
        'Renewable energy is a solution of climate change',
        'Ending poverty is foundation for human development and sustainable future'
    ]

    # 3. Run prediction
    print("Running prediction...")
    results = predicator(args, texts_to_predict)
    
    # 4. Display results
    print("\n--- Prediction Results ---")
    for i, text in enumerate(texts_to_predict):
        print(f"\nText: '{text}'")
        predicted_goals = [args.goal_contents[j] for j, pred in enumerate(results['preds'][i]) if pred == 1]
        print(f"  Predicted SDGs: {predicted_goals}")
        
    print("\nPrediction complete. Detailed results (vectors, probabilities, etc.) are in the 'results' dictionary.")


def run_training_pipeline():
    """
    This function contains the full pipeline to train the model from scratch.
    NOTE: To run this, you must first prepare the training corpus `corpus.pkl`
    by following the steps in the Supplementary Information S4 of our paper.
    """
    # 1. Set result folder
    args = Args()
    args.file_name = 'sdg_classifier_training'
    args.process_args()

    # 2. Set language and model
    args.language = 'en'
    args.model_name = 'luke'
    args.model_path = 'studio-ousia/luke-large-lite'
    args.epochs = 32

    # 3. Train and Validate
    print("Step 3: Creating dataset from corpus.pkl...")
    df = create_dataset(args)
    train_df, val_df = dataset_splitter(df, n_splits=5)
    print("Running training and validation...")
    run_without_optimize(args, train_df, val_df)

    # 4. Create best model
    print("Step 4: Generating the best model using the full dataset...")
    df = create_dataset(args)
    generate_best_model(args, df)
    print("Training pipeline complete.")


if __name__ == '__main__':
    # By default, this script runs a simple prediction example.
    run_prediction_example()
    
    # To run the full training pipeline, uncomment the line below.
    # WARNING: This requires the pre-compiled training corpus and will take a long time.
    # run_training_pipeline()