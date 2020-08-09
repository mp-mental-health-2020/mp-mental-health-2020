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
    fig = plt.figure(figsize=(8, 8))
    plt.hist(durations, bins=range(0, 40))
    plt.xlabel("duration in s")
    plt.ylabel("# of samples")
    plt.legend()
    plt.show()
    return fig


def swarm_plot_top_features(data):
    sns.set(style="whitegrid", palette="muted")
    data = pd.melt(data, id_vars=["index", "class"], var_name="features")
    print(data.head())
    fig = plt.figure(figsize=(25,10))
    sns.swarmplot(x="features", y="value", hue="class", data=data)
    return fig


def pca_2d(X, y, targets, colors):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)
    principal_df = pd.DataFrame(data=principal_components
                               , columns=['principal component 1', 'principal component 2'])
    final_df = pd.concat([principal_df, y.reset_index(drop=True)], axis=1)
    final_df.columns = ['principal component 1', 'principal component 2', 'target']

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    for target, color in zip(targets, colors):
        indices_to_keep = final_df['target'] == target
        ax.scatter(final_df.loc[indices_to_keep, 'principal component 1']
                   , final_df.loc[indices_to_keep, 'principal component 2']
                   , c=color)
    ax.legend(targets)
    ax.grid()
    return fig


def pca_null_clf(X, X_null, n_components=2):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    principal_components_ocd = pca.fit_transform(X)
    columns = ['principal component 1', 'principal component 2'] if n_components==2 else ['principal component 1', 'principal component 2', 'principal component 3']
    principal_df_ocd = pd.DataFrame(data=principal_components_ocd
                               , columns=columns)
    principal_components_null = pca.transform(X_null)
    principal_df_null = pd.DataFrame(data=principal_components_null
                               , columns=columns)
    if 2 <= n_components <= 3:
        fig = plt.figure(figsize=(8, 8))
        if n_components == 2:
            ax = fig.add_subplot(1, 1, 1)
        else:
            # to plot this in jupyter interactively you need to add the following line to the notebook
            # %matplotlib qt
            from mpl_toolkits import mplot3d
            ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        if n_components == 3:
            ax.set_zlabel('Principal Component 3', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        if n_components == 2:
            ax.scatter(principal_df_ocd['principal component 1'], principal_df_ocd['principal component 2'], color="b")
            ax.scatter(principal_df_null['principal component 1'], principal_df_null['principal component 2'], color="r")
        else:
            ax.scatter(principal_df_ocd['principal component 1'], principal_df_ocd['principal component 2'], principal_df_ocd['principal component 3'], color="b")
            ax.scatter(principal_df_null['principal component 1'], principal_df_null['principal component 2'], principal_df_null['principal component 3'], color="r")
        plt.show()

    print("PCA variance")
    print(pca.explained_variance_ratio_)

    return principal_components_ocd, principal_components_null


def pca_3d(X, y, targets= ['OCD activity', 'null class'], colors = ['r', 'b']):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(X)
    principal_df = pd.DataFrame(data=principal_components
                               , columns=['principal component 1', 'principal component 2', 'principal component 3'])
    final_df = pd.concat([principal_df, y.reset_index(drop=True)], axis=1)
    final_df.columns = ['principal component 1', 'principal component 2', 'principal component 3', 'target']

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_zlabel('Principal Component 3', fontsize=15)
    ax.set_title('3 component PCA', fontsize=20)

    for target, color in zip(targets, colors):
        indices_to_keep = final_df['target'] == target
        ax.scatter(final_df.loc[indices_to_keep, 'principal component 1']
                   , final_df.loc[indices_to_keep, 'principal component 2']
                   , final_df.loc[indices_to_keep, 'principal component 3']
                   , c=color)
    ax.legend(targets)
    ax.grid()
    return fig

def sne_2d(X, y, targets= ['OCD activity', 'null class'], colors = ['r', 'b'], n_iter=5000, perplexity=25):
    from sklearn.manifold import TSNE
    sne = TSNE(n_components=2, n_iter=n_iter, perplexity=perplexity)
    sne_components = sne.fit_transform(X)
    sne_df = pd.DataFrame(data=sne_components
                          , columns=['principal component 1', 'principal component 2'])
    sne_final_df = pd.concat([sne_df, y.reset_index(drop=True)], axis=1)
    sne_final_df.columns = ['principal component 1', 'principal component 2', 'target']

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component SNE', fontsize=20)

    for target, color in zip(targets, colors):
        indices_to_keep = sne_final_df['target'] == target
        ax.scatter(sne_final_df.loc[indices_to_keep, 'principal component 1']
                   , sne_final_df.loc[indices_to_keep, 'principal component 2']
                   , c=color)
    ax.legend(targets)
    ax.grid()
    return fig

def sne_3d(X, y, targets= ['OCD activity', 'null class'], colors = ['r', 'b'], n_iter=5000, perplexity=25):
    from sklearn.manifold import TSNE
    sne = TSNE(n_components=3, n_iter=n_iter, perplexity=perplexity)
    sne_components = sne.fit_transform(X)
    sne_df = pd.DataFrame(data=sne_components
                          , columns=['principal component 1', 'principal component 2', 'principal component 3'])
    sne_final_df = pd.concat([sne_df, y.reset_index(drop=True)], axis=1)
    sne_final_df.columns = ['principal component 1', 'principal component 2', 'principal component 3', 'target']

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_zlabel('Principal Component 3', fontsize=15)
    ax.set_title('3 component SNE', fontsize=20)

    for target, color in zip(targets, colors):
        indices_to_keep = sne_final_df['target'] == target
        ax.scatter(sne_final_df.loc[indices_to_keep, 'principal component 1']
                   , sne_final_df.loc[indices_to_keep, 'principal component 2']
                   , sne_final_df.loc[indices_to_keep, 'principal component 3']
                   , c=color)
    ax.legend(targets)
    ax.grid()
    return fig

def expand_columns(df, list_columns):
    df_result = df.copy(deep = True)
    for col in list_columns:
        colvalues = df_result[col].unique()
        for colvalue in colvalues:
            newcol_name = "{} is {}".format(col, colvalue)
            df_result.loc[df_result[col] == colvalue, newcol_name] = 1
            df_result.loc[df_result[col] != colvalue, newcol_name] = 0
    df_result.drop(list_columns, inplace=True, axis=1)
    return df_result

import seaborn as sn

def visualize_final_results():
    data = pd.read_csv("../../all_results.csv", sep="|", index_col=False)
    config_options = data["experiment"].str.split("_", expand=True,)
    classType = data["experiment"].str.split("[A-Z]", n=1, expand=True,)[0]
    config_options = pd.concat([classType, config_options], axis = 1)
    config_options.columns = range(config_options.shape[1])
    config_options[1] = data["experiment"].str.split("[A-Z]", n=1, expand=True,)[1]
    indoor = data["experiment"].str.split("ind", n=1, expand=True,)[1]
    config_options = pd.concat([indoor, config_options], axis = 1)
    config_options.columns = range(config_options.shape[1])
    config_options[2] = config_options[2].str.split("ind", n=1, expand=True,)[0]
    config_options.columns = ['proximity used', 'classification type binary', 'recordings multi', 'fingerprinting used', 'features minimal', 'window size', 'null class used']
    data = pd.concat([config_options, data], axis = 1).drop(columns=["experiment"]).replace("fingerprintingFalse", False).replace("fingerprintingTrue", True).replace("fingerprintingTrue", True).replace("nullClassIncludedTrue", True).replace("nullClassIncludedFalse", False).replace("oorFalse.*", False, regex=True).replace("oorTrue.*", True, regex=True).replace("riane", False).replace("na-2-Ariane-Julian-Wiktoria.*", True, regex=True).replace("binary", True).replace("multi", False).replace("featuresMinimalFCParameters", True).replace("featuresEfficientFCParameters", False).replace("windowSize100", "100").replace("windowSize200", "200")
    data["null class used"] = data["null class used"].fillna("Binary")
    data['proximity used'].loc[(data['proximity used']) & (data['fingerprinting used'])] = False
    data = data[['classification type binary', 'null class used', 'recordings multi', 'features minimal', 'proximity used', 'fingerprinting used', 'window size', 'Logistic_Regression']]
    data = expand_columns(data, ['window size'])
    data = data.drop("window size is windowSize50", axis=1)
    data_binary = data.loc[data['classification type binary']].drop(["null class used", "classification type binary"], axis = 1)
    data_multi = data.loc[data['classification type binary'] == False].drop("classification type binary", axis = 1)

    #print(data_multi["Logistic_Regression"].sum()/len(data_multi["Logistic_Regression"]))
    y = data_multi[["Logistic_Regression"]]
    X = data_multi.drop("Logistic_Regression", axis = 1)
    import statsmodels.api as sm
    X = sm.add_constant(X)
    results = sm.OLS(y, X.astype(float), missing="drop").fit()
    print(results.params)

    #corrMatrix = data_multi.corr()["Logistic_Regression"]
    sn.heatmap(pd.DataFrame(results.params).drop("const", axis = 0), annot=True) #.drop("Logistic_Regression", axis = 0)
    plt.show()

def test_final_results_visualization():
    visualize_final_results()