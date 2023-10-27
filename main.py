from util import *

'''Dataset: Low, D. M., Rumker, L., Talker, T., Torous, J., Cecchi, G., & Ghosh, S. S. Natural Language Processing Reveals Vulnerable Mental Health Support Groups and Heightened Health Anxiety on Reddit during COVID-19: An Observational Study. Journal of medical Internet research. doi: 10.2196/22635
Model: V. Vajre, M. Naylor, U. Kamath and A. Shehu, "PsychBERT: A Mental Health Language Model for Social Media Mental Health Behavioral Analysis," 2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), Houston, TX, USA, 2021, pp. 1077-1082, doi: 10.1109/BIBM52615.2021.9669469.'''

if __name__ == '__main__':
    global_file_path = '/data' # directory location
    subreddit = 'c' # picks which subreddits are loaded from full data set, options: 'c', 'mh', 'nmh', 'all'
    subsample = True # If false, full datasets are used, else uses subsample_size for subsampling
    subsample_size = 453 # number of posts from each subreddit, note: int for balanced, float for unbalanced
    train_model = True # Trains model if true, else skips training and embeddings are generated and evaluated as such
    batch_size = 15 # number of posts per batch when training and evaluating model
    num_trials = 20 # number of random trials used by classify for significance testing
    perform_pca_bert=True # performs PCA on BERT embeddings when True, uses n_pca_components, note: auto set to True if pca_only is True
    perform_pca_feat=True # performs PCA on feature embeddings when True, uses n_pca_components, note: auto set to True if pca_only is True
    n_pca_components_bert=None # determines the number of components for PCA dimensionality reduction for BERT, note: if None or if >= original # of dimensions, no reduction
    n_pca_components_feat=None # determines the number of components for PCA dimensionality reduction for features, note: if None or if >= original # of dimensions, no reduction
    n_eig_view=20 # determines number of viewed eigenvalues in printed output, no bearing on reduction: note: if > n_pca_components, defaults to n_pca_components
    rand_feat = False # If true, uses randomly generated features instead of given text features, note: meant for code testing purposes
    seed = 42 # seed for random seed


    np.random.seed(seed) # set random seed for subsampling df

    with open(join(global_file_path, f"terminal_output-{datetime.now().strftime('%m-%d-%y %X')}.out"), 'w') as f:
        sys.stdout = f

        print(f'Parameters of run:')
        print(f'    subreddit = {subreddit}')
        print(f'    subsample = {subsample}')
        print(f'    subsample_size = {subsample_size}')
        print(f'    train_model = {train_model}')
        print(f'    batch_size = {batch_size}')
        print(f'    num_trials = {num_trials}')
        print(f'    perform_pca_bert = {perform_pca_bert}')
        print(f'    perform_pca_feat = {perform_pca_feat}')
        print(f'    n_pca_componenents_bert = {n_pca_components_bert}')
        print(f'    n_pca_componenents_feat = {n_pca_components_feat}')
        print(f'    n_eig_view = {n_eig_view}')
        print(f'    rand_feat = {rand_feat}')
        print(f'    seed = {seed}\n_____________________________________________________________\n')

        print(f'Beginning data loading...\n')
        start = time.time()
        reddit_data = load_data(global_file_path, subsample, subsample_size, subreddit)
        stop = time.time()
        print(f'Data loading complete, run time: {stop-start:.2f} s...\n   data has shape [{reddit_data.shape[0]} x {reddit_data.shape[1]}]\n')

        le = preprocessing.LabelEncoder()
        reddit_data.loc[:, 'subreddit'] = le.fit_transform(reddit_data['subreddit'])
        token_data = get_preprocessed_for_tokenization(reddit_data, seed)

        print(f'Loading pretrained BERT embeddings...\n')
        X_bert, y_bert = load_BERT_embeddings(global_file_path, token_data, batch_size, subreddit, train_model)

        print(f'Beginning feature processing...\n')
        start = time.time()
        X_feat, y_feat = get_features(reddit_data)
        stop = time.time()
        print(f'Feature processing complete, run time: {stop-start:.2f} s...\n   features have shape [{len(X_feat)} x {len(X_feat[0])}]\n')

        if rand_feat:
            X_feat = np.random.rand(len(X_feat), len(X_feat[0])) # testing classification in random case instead of features


        print(f'Beginning classification...\n')

        start_b = time.time()
        accs_bert, f1s_bert = classify(global_file_path, X_bert, y_bert, n_trials=num_trials, perform_pca=perform_pca_bert, n_pca_components=n_pca_components_bert, vec_name='BERT', n_eig_view=n_eig_view)
        stop_b = time.time()

        start_f = time.time()
        accs_feat, f1s_feat = classify(global_file_path, X_feat, y_feat, n_trials=num_trials, perform_pca=perform_pca_feat, n_pca_components=n_pca_components_feat, vec_name='Features', n_eig_view=n_eig_view)
        stop_f = time.time()

        acc_bert = get_stats(accs_bert)
        f1_bert = get_stats(f1s_bert)
        acc_feat = get_stats(accs_feat)
        f1_feat = get_stats(f1s_feat)

        print(f'Classification results over {num_trials} trials:')
        print(f"    BERT:        Runtime: {stop_b-start_b:.2f} s    ---    mean accuracy: {acc_bert['mu']:.2f}, σ: {acc_bert['sigma']:.2f}    ---    mean f1: {f1_bert['mu']:.2f}, σ: {f1_bert['sigma']:.2f}")
        print(f"    Features:    Runtime: {stop_f-start_f:.2f} s    ---    mean accuracy: {acc_feat['mu']:.2f}, σ: {acc_feat['sigma']:.2f}    ---    mean f1: {f1_feat['mu']:.2f}, σ: {f1_feat['sigma']:.2f}\n")

        print(f'\n    Results saved at {plot_classification_results(global_file_path, accs_bert, f1s_bert, accs_feat, f1s_feat)}\n')
