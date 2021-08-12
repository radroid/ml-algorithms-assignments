import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, learning_curve, RepeatedKFold
from sklearn.metrics import classification_report, confusion_matrix, auc


def plot_conf_mat(*all_conf_mat, **format_settings):
    """Plots the confusion matrix using Seaborn Heatmap.
    
    Args:
        all_conf_mat (args as ndarray or pandas dataframe): The confusion matrix generated using sklearn's method.
        format_settings (keywords arguements): 
            titles_list (list of str): Titles to be given to each of the plots.
            cmaps_list (list of str): cmap value to decide the color for each of the plots 
                                     (refer to Matplotlib for possible values).
            num_of_rows (int): number of rows of the plots.
            num_of_cols (int): number of columns of the plots.
        
    Returns: None
    """
    num_of_plots = len(all_conf_mat)
    num_of_rows, num_of_cols = get_ideal_plot_dim(num_of_plots)

    all_titles = format_settings.get("titles_list", ["Confusion Matrix"]*num_of_plots)
    all_cmaps = format_settings.get("cmaps_list", ['Blues']*num_of_plots)
    nrows = format_settings.get("nrows", num_of_rows)
    ncols = format_settings.get("ncols", num_of_cols)
    figsize = format_settings.get("figsize", (12, 4))
    
    fig, axes = plt.subplots(nrows=nrows,
                             ncols=ncols,
                             figsize=figsize)
    if not type(axes) == np.ndarray:
        axes = [axes]
    
    for ax, conf_mat, title, cmap in zip(axes, all_conf_mat, all_titles, all_cmaps):
        sns.heatmap(conf_mat, 
                    annot=True,
                    annot_kws={'size': 12},
                    cbar=False,
                    cmap=cmap,
                    ax=ax)

        ax.set_title(title, fontdict={'size': 16, 'weight': 'bold'}, pad=30)
        ax.set_xlabel('Predicted Label', fontdict={'size': 14, 'weight': 'bold'}, labelpad=20)
        ax.set_ylabel('True Label', fontdict={'size': 14, 'weight': 'bold'}, labelpad=20)
        
        loc_x = ax.get_xticks()
        loc_y = ax.get_yticks()
        loc_y -= 0.3

        ax.set_xticklabels(['Edible', 'Poisonuous'], fontdict={"fontsize": 14})
        ax.set_yticks(loc_y)
        ax.set_yticklabels(['Edible', 'Poisonuous'], fontdict={"fontsize": 14}, rotation=90)
        
    plt.tight_layout(rect=[0, 0.03, 1.1, 0.97])


def get_ideal_plot_dim(total_plots):
    """Calculated the best number of rows and columns to create plots clearly.
    
    Args:
        total_plots (int): Total number of plots to be created.
    
    Returns:
        nrows (int): Number of rows to be plot.
        ncols (int): Number of columns to be plot.
    """
    nrows = 1
    ncols = 3
    if total_plots <= 3:
        ncols = total_plots
        return nrows, ncols
    else:
        nrows = round(total_plots / 3) + 2
        return nrows, ncols


def metric_evaluation(model, X, y, cv: int = 5, print_results: bool = True):
    """
    The function evalutes the model on five metrics using cross validation scores for each and plots them.
    The six metrics are:
    1. Accuracy
    2. Weighted Precision
    3. Weighted Recall
    4. Weighted f1 score
    6. Area under the ROC curve
    
    Notes:
        The higher the value of the metric, the better. 
    
    Args:
        model (object): An instantiated object of a sklearn classifier.
        X (np.ndarray or pd.DataFrame): Contains the features from the dataset.
        y (np.ndarray or pd.DataFrame): Contains the target or response varible from the dataset.
        cv (int): number of folds in the cross validation score.
        print_results (bool): prints the performance dataframe if True. Default True.
    
    Returns:
        performance_df (pd.DataFrame): A dictionary containing the four performance metrics.
    """
    
    # Setting up random seed to ensure all models are evaluated on same data splits
    np.random.seed(100)

    # Creating a list of metrics
    metrics_list = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    performance_dict = {}

    for metric in metrics_list:
        metric_score = cross_val_score(model, X, y, cv=cv, scoring=metric)
        performance_dict.update({metric: metric_score})
        
    performance_df = pd.DataFrame(performance_dict).round(3).T
    performance_df.columns = [f"Fold {n}" for n in range(1, len(metric_score)+1)]

    # Instantiating figure
    fig, ax = plt.subplots(figsize=(10,7))

    # plotting data
    performance_df.T.plot.bar(ax=ax)

    # Formatting title and axes
    ax.set_title("Cross Validation scores of the Model", 
                fontdict={"fontsize":20, "fontweight":'bold'},
                pad=30)

    ax.set_xlabel("Cross Validation Fold number", fontsize=14, fontweight='bold', labelpad=20)
    ax.set_ylabel("Cross Validated scores of the Metric", 
                fontsize=14, fontweight='bold', labelpad=20)

    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12)

    # Formatting legend
    leg = ax.legend(fontsize=12, loc=(1.02, 0.81), frameon=True)
    leg.get_frame().set_color("#F2F2F2")
    leg.get_frame().set_edgecolor("#000000")
    leg.set_title("Estimator", prop={"size": 14, "weight": 'bold'})

    # Displaying Results
    if print_results:
        print("######### Averaged Cross Validation Scores ##########\n"
              "Accuracy score:            {accuracy: 0.2%}\n"
              "Weighted Precision score:  {precision_weighted: 0.2%}\n"
              "Weighted Recall score:     {recall_weighted: 0.2%}\n"
              "Weighted F1 score:         {f1_weighted: 0.2%}\n"
              "ROC Area Under Curve:      {roc_auc: .3f}".format(**dict(performance_df.T.mean())))
    
    # Print statement to know completion, incase of function being called multiple times in a cell.
    print("Model evaluation complete.") 

    return performance_df


def plot_learning_curves(model, X, y):
    """
    The function plots the learning curve for the input model.

    Args:
        model (object): An instantiated object of a sklearn classifier.
        X (np.ndarray or pd.DataFrame): Contains the features from the dataset.
        y (np.ndarray or pd.DataFrame): Contains the target or response varible from the dataset.
    """
    # Setting up random seed to ensure all models are evaluated on same data splits
    np.random.seed(100)

    train_sizes, train_scores, test_scores = learning_curve(estimator=model,
                                                            X=X, 
                                                            y=y,
                                                            train_sizes= np.linspace(0.1, 1.0, 10),
                                                            cv=10,
                                                            scoring='recall_weighted',random_state=100)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean,color='blue', marker='o', 
             markersize=5, label='training recall')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
             label='validation recall')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std,
                     alpha=0.15, color='green')
    plt.grid(True)
    plt.xlabel('Number of training samples')
    plt.ylabel('Recall')
    plt.legend(loc='best')
    plt.show()
    

def plot_box_plot(model, X, y, cv: int = 5, model_name: str = "Model", scoring: str = "recall_weighted",
                  models: list = None):
    """The function plots a box plot for the scoring parameter defined.

    Args:
        model (object): An instantiated object of a sklearn classifier.
        X (np.ndarray or pd.DataFrame): Contains the features from the dataset.
        y (np.ndarray or pd.DataFrame): Contains the target or response varible from the dataset.
        model_name (str, optional): Name of the model used. If None, uses `Decision Tree`. Defaults to None.
        scoring (str, optional): The scoring parameter to be used. Defaults to "recall_weighted".
        models (list, optional): A list of `model`s. Defaults to [].
    """
    # Setting up random seed to ensure all models are evaluated on same data splits
    np.random.seed(100)

    if models is None:
        models = []
        models.append((model_name, model))

    results =[]
    names=[]
    scoring ='recall_weighted'
    metric = scoring.replace('_', ' ').capitalize()
    print(f'Model Evaluation - {metric}')
    for name, model in models:
        # rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)
        cv_results = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        print('{} {:.2f} +/- {:.2f}'.format(name,cv_results.mean(),cv_results.std()))
    print('\n')

    fig = plt.figure(figsize=(5,5))
    fig.suptitle('Boxplot View')
    ax = fig.add_subplot(111)
    sns.boxplot(data=results)
    ax.set_xticklabels(names)
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.show()
    pass


def full_model_evaluation(model, X, y, cv: int = 5, model_name: str = None):
    """
    The function does the following 4 things:
    1. Plot the learning curve.
    2. Plot a boxplot for weighted recall.
    3. Plot the cross validation scores of 5 metrics.
    """
    # Setting up random seed to ensure all models are evaluated on same data splits
    np.random.seed(100)

    if model_name is None:
        model_name = "Decision Tree"

    # Plot learning curve
    print(f'{model_name} Learning Curve')
    plot_learning_curves(model, X, y)
    
    # Model Evaluation - Boxplot
    plot_box_plot(model, X, y, cv=cv, model_name=model_name)
    
    # Evaluate the performance
    metric_evaluation(model, X, y)
