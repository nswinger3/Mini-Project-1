from transformers import AutoModelForMaskedLM, AutoTokenizer
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
import matplotlib.pyplot as plt
from datetime import datetime
import time
from collections import OrderedDict


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


def load_model(global_file_path):
    try:
        pytorch_lm = torch.load(join(global_file_path, 'model.pth'))
    except FileNotFoundError:
        pytorch_lm = AutoModelForMaskedLM.from_pretrained('mnaylor/psychbert-cased', from_flax=True,
                                                          output_hidden_states=True)
        torch.save(pytorch_lm, join(global_file_path, 'model.pth'))
    pytorch_tk = AutoTokenizer.from_pretrained("mnaylor/psychbert-cased")
    return pytorch_lm, pytorch_tk


def get_max_subsample_size(path, fs):
    min_len = 1000000
    for f in fs:
        df = clean(pd.read_csv(join(path, f), engine='python', on_bad_lines='skip'))
        if len(df) < min_len:
            min_len = len(df)
    return min_len


def load_data(global_file_path, file_name=None, subsample=True, subsample_size=10, subreddit='c', clean_data=True):
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

            all_files.sort()
            df = pd.DataFrame()
            for f in all_files:
                df2 = pd.read_csv(join(path, f), engine='python', on_bad_lines='skip')
                if clean_data:
                    df2 = clean(df2)
                if subsample:
                    df2 = subsample_df(df2, subsample_size)
                df = pd.concat([df, df2])
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


def tokenize(global_file_path, tokenizer, model, posts, max_batch_size, max_tokens, subreddit):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    num_batches = int(np.ceil(len(posts) / max_batch_size))
    post_batches = np.array_split(posts, num_batches)

    model.eval()
    model.to(device)

    embeddings = []

    print(f'Tokenizing posts and feeding to pretrained BERT model...\n')
    start = time.time()
    for i, post_batch in enumerate(post_batches):
        encoded_input = tokenizer(post_batch.tolist(), padding=True, truncation=True, max_length=max_tokens, return_tensors="pt")

        # Move tokenized data to GPU just before forwarding through the model
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            output = model(**encoded_input)
            hidden_states = output.hidden_states

            token_vecs = hidden_states[-2]
            sentence_embedding = torch.mean(token_vecs, dim=1)
            embeddings.extend(sentence_embedding.cpu().numpy())

    stop = time.time()
    print(f'BERT embeddings generated, run time: {stop-start:.2f} s...\n   embedding set has shape [{len(embeddings)} x {len(embeddings[0])}]\n')
    np.save(join(global_file_path, f'BERT_embeddings-{subreddit}-mb{max_batch_size}-mt{max_tokens}-{len(embeddings)}x{len(embeddings[0])}'), embeddings)

    return embeddings


def get_embedding(sentence, posts, embeddings):
    index = posts.index(sentence)
    return embeddings[index]


def load_BERT_embeddings(global_file_path, posts, max_batch_size=5, max_tokens=512, subreddit='c'):
    try:
        start = time.time()
        embeddings = np.load(join(global_file_path, f'BERT_embeddings-{subreddit}-mb{max_batch_size}-mt{max_tokens}-{len(posts)}x768.npy'))
        stop = time.time()
        print(f'Pretrained BERT embeddings loaded, run time: {stop-start:.2f} s...\n   embedding set has shape [{len(embeddings)} x {len(embeddings[0])}]\n')
    except FileNotFoundError:
        print(f'Embedding save file not found, generating from scratch...\n')

        print(f'Loading BERT model and tokenizer...\n')
        start = time.time()
        psychBERT_model, tokenizer = load_model(global_file_path)
        stop = time.time()
        print(f'BERT model and tokenizer loaded, run time: {stop-start:.2f} s...\n')

        embeddings = tokenize(global_file_path, tokenizer, psychBERT_model, posts, max_batch_size, max_tokens, subreddit)

    return embeddings


def classify(global_file_path, X, y, n_trials=50, perform_pca=False, n_pca_components=150, n_eig_view=20, vec_name='', pca_only=False):
    if pca_only:
        perform_pca = True

    f1s = []
    accs = []

    if perform_pca:
        print(f'Performing PCA on {vec_name} embeddings...')
        evrs, evs = [], []
    for n in range(n_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        if perform_pca:
            if n_pca_components is not None and n_pca_components >= len(X_train[0]):
                n_pca_components = None

            pca = PCA(n_components=n_pca_components)

            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

            explained_variance_ratio = pca.explained_variance_ratio_
            explained_variance = pca.explained_variance_
            evrs.append(explained_variance_ratio)
            evs.append(explained_variance)

        if not pca_only:
            clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = clf.score(X_test, y_test)
            f1 = f1_score(y_test, y_pred, average='macro')
            f1s.append(f1)
            accs.append(accuracy)

    if perform_pca:
        evr_stats = get_stats(evrs, batch=True)
        ev_stats = get_stats(evs, batch=True)

        if n_eig_view > evr_stats['mu'].size:
            n_eig_view = evr_stats['mu'].size
        print(f"   proportion of explained variance by {n_eig_view} components: {evr_stats['mu'][:n_eig_view].sum():.2f}")
        print(f"   eigenvalues of first {n_eig_view} components: {np.round(ev_stats['mu'][:n_eig_view],2)}\n")

        plot_PCA(global_file_path, evr_stats, ev_stats, vec_name, n_trials)

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
    ax.plot(x, BERT_accs, marker=',', color='#ecf469', linestyle='', label='BERT accuracies')
    ax.plot(x * 2, FEAT_accs, marker=',', color='#b2dd53', linestyle='', label='Feature accuracies')
    ax.plot(x * 3, BERT_f1s, marker=',', color='#53c14f', linestyle='', label='BERT f1s')
    ax.plot(x * 4, FEAT_f1s, marker=',', color='#31945a', linestyle='', label='Feature f1s')

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
    ax.set_xlabel('Accuracy Metric                                F1 Metric')

    ax.set_yticks(np.linspace(0,1,11))
    ax.set_yticklabels(np.linspace(0,1,11))
    ax.set_ylabel('Scores')
    ax.set_title('Classification Results')
    ax.legend()

    results_file = join(global_file_path, f'results_{datetime.now().strftime("%m-%d-%y %X")}.png')
    plt.tight_layout()
    plt.savefig(f'{results_file}')
    plt.show()


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


def tens_ceil(x):
    return int(np.ceil(x / 10.0) * 10)
