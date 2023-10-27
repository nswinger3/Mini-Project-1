from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from os import listdir
from os.path import isfile, join
import pandas as pd
import torch
import numpy as np
from scipy.stats import sem, norm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from datetime import datetime
import time
from collections import OrderedDict
import sys
import evaluate
from datasets import Dataset, DatasetDict, concatenate_datasets
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm


def get_subreddits(v):
    condition_subreddits = ['EDAnonymous',
                            'addiction',
                            'adhd',
                            'alcoholism',
                            'anxiety',
                            'autism',
                            'bipolarreddit',
                            'bpd',
                            'depression',
                            'ptsd',
                            'schizophrenia']

    mh_subreddits = ['COVID19_support',
                     'EDAnonymous',
                     'addiction',
                     'adhd',
                     'alcoholism',
                     'anxiety',
                     'autism',
                     'bipolarreddit',
                     'bpd',
                     'depression',
                     'healthanxiety'
                     'lonely',
                     'mental_health',
                     'ptsd',
                     'schizophrenia',
                     'socialanxiety',
                     'suicide watch']

    non_mh_subreddits = ['conspiracy',
                         'divorce',
                         'fitness',
                         'guns',
                         'jokes',
                         'legaladvice',
                         'meditation',
                         'parenting',
                         'personalfinance',
                         'relationships'
                         'teaching']

    full_reddit = mh_subreddits + non_mh_subreddits

    if v == 'c':
        return condition_subreddits
    if v == 'mh':
        return mh_subreddits
    if v == 'nmh':
        return non_mh_subreddits
    else:
        return full_reddit


def load_data(global_file_path, subsample, subsample_size, subreddit, clean_data=True, file_name=None):
    # subsamples balanced across subreddits with subsample_size type int, unbalanced with float
    path = join(global_file_path, 'reddit_mental_health_dataset_nswinger')
    sub_names = {'c': 'condition', 'mh': 'mental_health', 'nmh': 'non_mental_health', 'all': 'full_reddit'}

    if subsample:
        save_file = join(global_file_path, f'{sub_names[subreddit]}_df_{subsample_size}.pkl')
    else:
        save_file = join(global_file_path, f'{sub_names[subreddit]}_df.pkl')

    if file_name is None:
        try:
            df = pd.read_pickle(save_file)
        except FileNotFoundError:
            all_files = [f for f in listdir(path) if isfile(join(path, f)) and any(n == f.split('_')[0] for n in get_subreddits(subreddit))]

            if subsample:
                max_subsample_size = get_max_subsample_size(path, all_files)
                if subsample_size > max_subsample_size:
                    print(f'Warning: Selected subsample size is too large, continuing with subsamples of size {max_subsample_size}...\n')
                    subsample_size = max_subsample_size
                    save_file = join(global_file_path, f'{sub_names[subreddit]}_df_{subsample_size}.pkl')

            progress_bar = tqdm(range(len(all_files)))
            progress_bar.set_description('Loading Data')
            all_files.sort()
            df = pd.DataFrame()
            for f in all_files:
                df2 = pd.read_csv(join(path, f), engine='python', on_bad_lines='skip')
                if clean_data:
                    df2 = clean(df2)
                if subsample:
                    df2 = subsample_df(df2, subsample_size)
                df = pd.concat([df, df2])
                progress_bar.update(1)

            progress_bar.close()
            print()
            if 'covid19_total' in df.columns:
                df = df.drop(columns=['covid19_total'])
            df.to_pickle(save_file)
    else:
        df = pd.read_csv(join(path, file_name), engine='python', on_bad_lines='skip')
        if clean_data:
            df = clean(df)
        if subsample:
            max_subsample_size = get_max_subsample_size(path, file_name)
            if subsample_size >= max_subsample_size:
                print(f'Warning: Selected subsample size is too large, continuing with subsamples of size {max_subsample_size}...\n')
                subsample_size = max_subsample_size
        df = subsample_df(df, subsample_size)

    return df


def subsample_df(df, subsample):
    if type(subsample) == float:
        subsample = int(df.shape[0] * subsample)
    df = df.reset_index(drop=True)
    df2 = df.loc[np.random.choice(df.index, subsample, replace=False)]
    return df2


def clean(df):
    # remove author duplicates and shuffle so we dont keep only first posts in time
    reddit_data = df.sample(frac=1)  # shuffle
    reddit_data = reddit_data.drop_duplicates(subset='author', keep='first')
    reddit_data = reddit_data[
        ~reddit_data.author.str.contains('|'.join(['bot', 'BOT', 'Bot']))]  # There is at least one bot per subreddit
    reddit_data = reddit_data[
        ~reddit_data.post.str.contains('|'.join(['quote', 'QUOTE', 'Quote']))]  # Remove posts in case quotes are long
    reddit_data = reddit_data.reset_index(drop=True)
    return reddit_data


def get_max_subsample_size(path, fs):
    min_len = 1000000
    for f in fs:
        df = clean(pd.read_csv(join(path, f), engine='python', on_bad_lines='skip'))
        if len(df) < min_len:
            min_len = len(df)
    return min_len


def get_preprocessed_for_tokenization(df, seed):
    df_subset = df[['subreddit', 'post']]
    train_df, test_df = train_test_split(df_subset, test_size=0.2, random_state=seed)
    train_dataset = Dataset.from_pandas(train_df).rename_column('subreddit', 'labels').rename_column('post', 'text').remove_columns(["__index_level_0__"])
    test_dataset = Dataset.from_pandas(test_df).rename_column('subreddit', 'labels').rename_column('post', 'text').remove_columns(["__index_level_0__"])
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    return dataset


def get_features(df):
    features = list(df.columns)
    features = [n for n in features if n not in ['subreddit', 'author', 'date', 'post']]
    return df[features].values, df['subreddit'].values.tolist()


def load_BERT_embeddings(global_file_path, dataset, batch_size, subreddit, train_model):
    try:
        start = time.time()
        total_posts = sum(len(dataset[split]['text']) for split in dataset)
        embeddings = np.load(join(global_file_path, f'BERT_embeddings-{subreddit}-batch{batch_size}-train{train_model}-emb{total_posts}x768.npy'))
        stop = time.time()
        print(f'Pretrained BERT embeddings loaded, run time: {stop-start:.2f} s...\n   embedding set has shape [{len(embeddings)} x {len(embeddings[0])}]\n')
    except FileNotFoundError:
        print(f'Embedding save file not found, generating from scratch...\n')

        print(f'Loading BERT model and tokenizer...\n')
        start = time.time()
        model, tokenizer = load_model(global_file_path, subreddit)
        stop = time.time()
        print(f'BERT model and tokenizer loaded, run time: {stop-start:.2f} s...\n')

        embeddings = generate_embeddings(global_file_path, tokenizer, model, dataset, batch_size, subreddit, train_model)

    return embeddings, dataset['train']['labels'] + dataset['test']['labels']


def load_model(global_file_path, subreddit):
    try:
        pytorch_lm = torch.load(join(global_file_path, f'model_{subreddit}.pth'))
    except FileNotFoundError:
        pytorch_lm = AutoModelForSequenceClassification.from_pretrained('mnaylor/psychbert-cased', from_flax=True, output_hidden_states=True, num_labels=len(get_subreddits(subreddit)))
        torch.save(pytorch_lm, join(global_file_path, f'model_{subreddit}.pth'))
    pytorch_tk = AutoTokenizer.from_pretrained("mnaylor/psychbert-cased")
    return pytorch_lm, pytorch_tk


def tokenize(tokenizer, dataset, batch_size):
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, )

    print(f'Tokenizing posts...\n')
    start = time.time()
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    combined_dataset = concatenate_datasets([tokenized_datasets["train"], tokenized_datasets["test"]])

    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=batch_size)
    full_dataloader = DataLoader(combined_dataset, batch_size=batch_size)
    stop = time.time()
    print(f'\nTokenization complete, run time: {stop-start:.2f} s...\n')

    return full_dataloader, train_dataloader, eval_dataloader


def train_m(model, train_dataloader):
    print(f'Training model...\n')
    start = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    progress_bar = tqdm(range(num_training_steps))
    progress_bar.set_description('Training Model')

    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    progress_bar.close()
    stop = time.time()
    print(f'\nModel training complete, run time: {stop-start:.2f} s...\n')
    return model


def evaluate_m(model, eval_dataloader):
    print(f'\nEvaluating model...\n')
    start = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    model.eval()

    progress_bar = tqdm(range(len(eval_dataloader)))
    progress_bar.set_description('Evaluating Model')
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        acc_metric.add_batch(predictions=predictions, references=batch["labels"])
        f1_metric.add_batch(predictions=predictions, references=batch["labels"])
        progress_bar.update(1)

    progress_bar.close()
    stop = time.time()
    print(f'\nModel evaluation complete, run time: {stop-start:.2f} s...')
    print(f'   evalution accuracy metric: {acc_metric.compute()}\n   evaluation f1 metric: {f1_metric.compute(average="macro")}\n')
    return model


def generate_embeddings(global_file_path, tokenizer, model, dataset, batch_size, subreddit, train_model):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    full_dataloader, train_dataloader, eval_dataloader = tokenize(tokenizer, dataset, batch_size)
    if train_model:
        model = train_m(model, train_dataloader)
        model = evaluate_m(model, eval_dataloader)
    else:
        model.to(device)
        model.eval()

    print(f'\nGenerating embeddings...\n')
    start = time.time()

    progress_bar = tqdm(range(len(full_dataloader)))
    progress_bar.set_description('Generating Embeddings')
    embeddings = []
    for batch in full_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            hidden_states = outputs.hidden_states
            token_vecs = hidden_states[-2]
            sentence_embedding = torch.mean(token_vecs, dim=1)
            embeddings.extend(sentence_embedding.cpu().numpy())

        progress_bar.update(1)

    progress_bar.close()
    stop = time.time()
    print(f'BERT embeddings generated, run time: {stop-start:.2f} s...\n   embedding set has shape [{len(embeddings)} x {len(embeddings[0])}]\n')
    np.save(join(global_file_path, f'BERT_embeddings-{subreddit}-batch{batch_size}-train{train_model}-emb{len(embeddings)}x{len(embeddings[0])}'), embeddings)

    return embeddings


def do_pca(X_train, X_test, n_pca_components, evrs, evs):
    if n_pca_components is not None and n_pca_components >= len(X_train[0]):
        n_pca_components = None

    pca = PCA(n_components=n_pca_components)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    explained_variance_ratio = pca.explained_variance_ratio_
    explained_variance = pca.explained_variance_
    evrs.append(explained_variance_ratio)
    evs.append(explained_variance)

    return X_train, X_test, evrs, evs


def eval_pca(global_file_path, evrs, evs, n_eig_view, vec_name, n_trials):
    evr_stats = get_stats(evrs, batch=True)
    ev_stats = get_stats(evs, batch=True)

    if n_eig_view > evr_stats['mu'].size:
        n_eig_view = evr_stats['mu'].size
    print(f'\nPCA on {vec_name} complete...')
    print(f"   proportion of explained variance by {n_eig_view} components: {evr_stats['mu'][:n_eig_view].sum():.2f}")
    print(f"   eigenvalues of first {n_eig_view} components: {np.round(ev_stats['mu'][:n_eig_view],2)}\n")

    print(f'\n results saved at {plot_PCA(global_file_path, evr_stats, ev_stats, vec_name, n_trials)}\n')


def classify(global_file_path, X, y, n_trials, perform_pca, n_pca_components, n_eig_view, vec_name):
    f1s = []
    accs = []

    if perform_pca:
        print(f'Performing PCA on {vec_name} embeddings...')
        evrs, evs = [], []

    progress_bar = tqdm(range(n_trials))
    progress_bar.set_description(f'Classifying {vec_name}')
    for n in range(n_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        if perform_pca:
            X_train, X_test, evrs, evs = do_pca(X_train, X_test, n_pca_components, evrs, evs)

        clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = clf.score(X_test, y_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        f1s.append(f1)
        accs.append(accuracy)

        progress_bar.update(1)

    progress_bar.close()

    if perform_pca:
        eval_pca(global_file_path, evrs, evs, n_eig_view, vec_name, n_trials)

    return accs, f1s


def get_stats(x, batch=False):
    assert type(x) is list, 'x must be a list'

    if batch:
        mu = np.mean(x, axis=0)
        sigma = np.std(x, axis=0, ddof=1)
        se = sem(x)
        ci = [norm.interval(0.95, loc=m, scale=s) for m, s in zip(mu, se)]
    else:
        mu = np.mean(x)
        sigma = np.std(x, ddof=1)
        se = sem(x)
        ci = norm.interval(0.95, loc=mu, scale=se)
    return {'mu':mu, 'sigma': sigma, 'se': se, 'ci': ci}


def plot_classification_results(global_file_path, BERT_accs, BERT_f1s, FEAT_accs, FEAT_f1s):
    assert len(BERT_accs) == len(BERT_f1s) == len(FEAT_accs) == len(FEAT_f1s), 'plot lists are not same length'

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    stats = OrderedDict({'BERT accuracy': get_stats(BERT_accs),
                         'feature accuracy': get_stats(FEAT_accs),
                         'BERT f1': get_stats(BERT_f1s),
                         'feature f1': get_stats(FEAT_f1s)})

    x = np.ones(len(BERT_accs))

    fig, ax = plt.subplots()

    # plot all trials
    ax.plot(x, BERT_accs, marker='.', ms=4, color='#2a623d', linestyle='', label='BERT accuracies')
    ax.plot(x * 2, FEAT_accs, marker='.', ms=4, color='#222f5b', linestyle='', label='Feature accuracies')
    ax.plot(x * 3, BERT_f1s, marker='.', ms=4, color='#1a472a', linestyle='', label='BERT f1s')
    ax.plot(x * 4, FEAT_f1s, marker='.', ms=4, color='#0e1a40', linestyle='', label='Feature f1s')

    # plot means and error bars
    for i, stat in enumerate(stats.values()):
        x, y = i + 1.1, stat['mu']
        ybot, ytop = [y - stat['ci'][0]], [stat['ci'][1] - y]
        ax.errorbar(x, y, yerr=(ybot, ytop), fmt='_r', ecolor='#00a6fb', label='95% CI' if i==0 else None)

        x += 0.1
        ax.errorbar(x, y, yerr=stat['se'], fmt='_r', ecolor='#0582ca', label='SEM' if i==0 else None)

        x += 0.1
        ax.errorbar(x, y, yerr=stat['sigma'], fmt='_r', ecolor='#006494', label='σ' if i==0 else None)


    ax.set_xticks((1.15, 2.15, 3.15, 4.15))
    ax.set_xticklabels(stats.keys())
    ax.set_xlabel('Accuracy Metric                                                                    F1 Metric      ')

    ax.set_yticks(np.linspace(0,1,11))
    ax.set_yticklabels((0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0))
    ax.set_ylabel('Scores')
    ax.set_title('Classification Results')
    ax.legend()

    results_file = join(global_file_path, f'results_{datetime.now().strftime("%m-%d-%y %X")}.png')
    plt.tight_layout()
    plt.savefig(f'{results_file}')
    plt.show()

    return results_file


def plot_PCA(global_file_path, evr_stats, ev_stats, vec_name, num_trials):
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, axs = plt.subplots(2, 1)
    num_comps = ev_stats['mu'].size
    x = np.arange(1, evr_stats['mu'].size + 1)

    axs[0].plot(np.insert(x, 0, 0), np.insert(np.cumsum(evr_stats['mu']), 0, 0))
    axs[0].axhline(y=.8, color='r', linestyle='--')

    x_ticks = np.linspace(0, tens_ceil(num_comps), num=11, dtype=int)
    axs[0].set_xticks(x_ticks)
    x_labels = np.char.mod('%s', x_ticks)
    # x_labels[0] = ''
    axs[0].set_xticklabels(x_labels)

    axs[0].set_xlim(-0.5, tens_ceil(num_comps) + .5)
    axs[0].set_xlabel('# of Components')

    axs[0].set_yticks(np.arange(0, 1.1, 0.1))
    axs[0].set_yticklabels(np.arange(0, 110, 10, dtype=int))
    axs[0].set_ylim(0, 1.05)
    axs[0].set_ylabel(f'μ % Variance Explained\n({num_trials} trials)')
    axs[0].set_title('PCA Proportion of Variance Explained')

    axs[1].bar(x, ev_stats['mu'], width=1, color='#31945a')
    axs[1].set_xticks(x_ticks)
    axs[1].set_xticklabels(x_labels)
    axs[1].set_xlim(0, tens_ceil(num_comps) + .5)
    axs[1].set_xlabel('Component #')
    axs[1].set_ylabel(f'μ Total Variance Explained\n({num_trials} trials)')

    axs[1].set_title('PCA Scree')

    fig.suptitle(f'{vec_name} PCA Results')

    results_file = join(global_file_path, f'pca_{datetime.now().strftime("%m-%d-%y %X")}.png')
    plt.tight_layout()
    plt.savefig(f'{results_file}')
    plt.show()

    return results_file


def tens_ceil(x):
    return int(np.ceil(x / 10.0) * 10)
