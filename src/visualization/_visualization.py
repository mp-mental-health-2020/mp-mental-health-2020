import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def plot_duration_histogram(chunks):
    """
    plots the histogram of the durations of all chunks
    Parameters
    ----------
    chunks: array of chunks - make sure to call with the chunks of only one hand, not the dictionary containing both hands

    Returns
    -------

    """
    durations = np.array([len(c) / 50 for c in chunks])
    print("Mean {:1.2f} +/- {:1.2f}".format(durations.mean(), durations.std()))
    plt.hist(durations, bins=range(0, 40))
    plt.xlabel("duration in s")
    plt.ylabel("# of samples")
    plt.legend()
    plt.show()


def swarm_plot_top_features(data):
    sns.set(style="whitegrid", palette="muted")
    data = pd.melt(data, id_vars=["index", "class"], var_name="features")
    print(data.head())
    plt.figure(figsize=(25,10))
    sns.swarmplot(x="features", y="value", hue="class", data=data)


def pca_2d(X, y, targets, colors):
    from sklearn.decomposition import PCA
    import pandas as pd
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, y.reset_index(drop=True)], axis=1)
    finalDf.columns = ['principal component 1', 'principal component 2', 'target']

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color)
    ax.legend(targets)
    ax.grid()


def pca_3d(X, y, targets= ['OCD activity', 'null class'], colors = ['r', 'b']):
    from sklearn.decomposition import PCA
    import pandas as pd
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2', 'principal component 3'])
    finalDf = pd.concat([principalDf, y.reset_index(drop=True)], axis=1)
    finalDf.columns = ['principal component 1', 'principal component 2', 'principal component 3', 'target']

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_zlabel('Principal Component 3', fontsize=15)
    ax.set_title('3 component PCA', fontsize=20)

    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , finalDf.loc[indicesToKeep, 'principal component 3']
                   , c=color)
    ax.legend(targets)
    ax.grid()

def sne_2d(X, y, targets= ['OCD activity', 'null class'], colors = ['r', 'b'], n_iter=5000, perplexity=25):
    from sklearn.manifold import TSNE
    import pandas as pd
    sne = TSNE(n_components=2, n_iter=n_iter, perplexity=perplexity)
    sne_components = sne.fit_transform(X)
    sne_df = pd.DataFrame(data=sne_components
                          , columns=['principal component 1', 'principal component 2'])
    sne_finalDf = pd.concat([sne_df, y.reset_index(drop=True)], axis=1)
    sne_finalDf.columns = ['principal component 1', 'principal component 2', 'target']

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    for target, color in zip(targets, colors):
        indicesToKeep = sne_finalDf['target'] == target
        ax.scatter(sne_finalDf.loc[indicesToKeep, 'principal component 1']
                   , sne_finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color)
    ax.legend(targets)
    ax.grid()

def sne_3d(X, y, targets= ['OCD activity', 'null class'], colors = ['r', 'b'], n_iter=5000, perplexity=25):
    from sklearn.manifold import TSNE
    import pandas as pd
    sne = TSNE(n_components=3, n_iter=n_iter, perplexity=perplexity)
    sne_components = sne.fit_transform(X)
    sne_df = pd.DataFrame(data=sne_components
                          , columns=['principal component 1', 'principal component 2', 'principal component 3'])
    sne_finalDf = pd.concat([sne_df, y.reset_index(drop=True)], axis=1)
    sne_finalDf.columns = ['principal component 1', 'principal component 2', 'principal component 3', 'target']

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_zlabel('Principal Component 3', fontsize=15)
    ax.set_title('3 component SNE', fontsize=20)

    for target, color in zip(targets, colors):
        indicesToKeep = sne_finalDf['target'] == target
        ax.scatter(sne_finalDf.loc[indicesToKeep, 'principal component 1']
                   , sne_finalDf.loc[indicesToKeep, 'principal component 2']
                   , sne_finalDf.loc[indicesToKeep, 'principal component 3']
                   , c=color)
    ax.legend(targets)
    ax.grid()