from util import *

'''Dataset: Low, D. M., Rumker, L., Talker, T., Torous, J., Cecchi, G., & Ghosh, S. S. Natural Language Processing Reveals Vulnerable Mental Health Support Groups and Heightened Health Anxiety on Reddit during COVID-19: An Observational Study. Journal of medical Internet research. doi: 10.2196/22635
Model: V. Vajre, M. Naylor, U. Kamath and A. Shehu, "PsychBERT: A Mental Health Language Model for Social Media Mental Health Behavioral Analysis," 2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), Houston, TX, USA, 2021, pp. 1077-1082, doi: 10.1109/BIBM52615.2021.9669469.'''

if __name__ == '__main__':
    global_file_path = '/Users/nswinger/CodingProjects/MiniProject1' # directory location
    subreddit = 'c' # picks which subreddits are loaded from full data set, options: 'c', 'mh', 'nmh', 'all'
    subsample_size = 453 # number of posts from each subreddit, note: int for balanced, float for unbalanced
    max_batch_size=5 # number of posts per batch when tokenizing and modeling
    max_tokens=512 # max number of tokens per post
    pca_only = False # won't do any classification when False,
    num_trials = 100 # number of random trials used by classify for significance testing
    perform_pca_bert=False # performs PCA on BERT embeddings when True, uses n_pca_components, note: auto set to True if pca_only is True
    perform_pca_feat=False # performs PCA on feature embeddings when True, uses n_pca_components, note: auto set to True if pca_only is True
    n_pca_components_bert=None # determines the number of components for PCA dimensionality reduction for BERT, note: if None or if >= original # of dimensions, no reduction
    n_pca_components_feat=None # determines the number of components for PCA dimensionality reduction for features, note: if None or if >= original # of dimensions, no reduction
    n_eig_view=20 # determines number of viewed eigenvalues in printed output, no bearing on reduction: note: if > n_pca_components, defaults to n_pca_components
    np.random.seed(0) # set random seed for subsampling df

    print(f'Beginning data loading...\n')
    start = time.time()
    reddit_data = load_data(global_file_path, subsample_size=subsample_size, subreddit=subreddit)
    stop = time.time()
    print(f'Data loading complete, run time: {stop-start:.2f} s...\n   data has shape [{reddit_data.shape[0]} x {reddit_data.shape[1]}]\n')

    posts = reddit_data.post.values.tolist()

    print(f'Loading pretrained BERT embeddings...\n')
    psychBERT_embeddings = load_BERT_embeddings(global_file_path, posts, max_batch_size=max_batch_size, max_tokens=max_tokens, subreddit=subreddit)

    X_bert = psychBERT_embeddings

    print(f'Beginning feature processing...\n')
    start = time.time()
    features = list(reddit_data.columns)
    features = [n for n in features if n not in ['subreddit', 'author', 'date', 'post']]
    X_feat = reddit_data[features].values
    stop = time.time()
    print(f'Feature processing complete, run time: {stop-start:.2f} s...\n   features have shape [{len(X_feat)} x {len(X_feat[0])}]\n')

    y = reddit_data.subreddit.values

    if not pca_only:
        print(f'Beginning classification...\n')

    start_b = time.time()
    accs_bert, f1s_bert = classify(global_file_path, X_bert, y, n_trials=num_trials, perform_pca=perform_pca_bert, n_pca_components=n_pca_components_bert, vec_name='BERT', n_eig_view=n_eig_view, pca_only=pca_only)
    stop_b = time.time()

    start_f = time.time()
    accs_feat, f1s_feat = classify(global_file_path, X_feat, y, n_trials=num_trials, perform_pca=perform_pca_feat, n_pca_components=n_pca_components_feat, vec_name='Features', n_eig_view=n_eig_view, pca_only=pca_only)
    stop_f = time.time()

    if not pca_only:
        acc_bert = get_stats(accs_bert)
        f1_bert = get_stats(f1s_bert)
        acc_feat = get_stats(accs_feat)
        f1_feat = get_stats(f1s_feat)

        print(f'Classification results over {num_trials} trials:')
        print(f"    BERT:    Runtime: {stop_b-start_b:.2f} s    ---    mean accuracy: {acc_bert['mu']:.2f}, σ: {acc_bert['sigma']:.2f}    ---    mean f1: {f1_bert['mu']:.2f}, σ: {f1_bert['sigma']:.2f}")
        print(f"    Features:    Runtime: {stop_f-start_f:.2f} s    ---    mean accuracy: {acc_feat['mu']:.2f}, σ: {acc_feat['sigma']:.2f}    ---    mean f1: {f1_feat['mu']:.2f}, σ: {f1_feat['sigma']:.2f}")

        plot_classification_results(global_file_path, accs_bert, f1s_bert, accs_feat, f1s_feat)
