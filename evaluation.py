import numpy as np
import pandas as pd
import os
import os.path as osp
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix 
import copy

print("imported")
def get_empty_reports():
    return { "labels": None, "results": None}

class Report(dict):
    def __init__(self, results = None, labels = None, class_labels = "infer", threshold = 0):
        self.results = results
        self.labels = labels
        self.threshold = threshold
        self.class_labels = class_labels
        for key, value in self.__dict__.items():
            self[key] = value
        
        
    def metrics(self):
        assert self.results is not None, "No results to evaluate."
        assert self.labels is not None, "No labels to evaluate."
        self.class_labels = range(self.labels[0]) if self.class_labels == "infer" else self.class_labels
        return compute_metrics_for_threshold(self["results"], self["labels"], self.threshold, self.class_labels)

    def overall_metrics(self):
        return get_overall_metrics(self.metrics(), self.threshold)
        
    def show_results(self, normalize_confusion = False):
        show_results(self.metrics(), normalize_confusion, self.threshold)
    
    def save_metrics(self, path, figures = False):
        metrics = self.metrics() if metrics is None else metrics 
        save_metrics(metrics, path, figures = figures)
         
    def save_labels_results(self, path, labels_name = "labels.npy", results_name = "results.npy"):
        np.save(osp.join(path, results_name), self.results)
        np.save(osp.join(path, labels_name), self.labels)
    
    def load_labels_results(self, path,labels_name = "labels.npy", results_name = "results.npy"):
        self.results = np.load(osp.join(path, results_name))
        self.labels = np.load(osp.join(path, labels_name))
    
    def __str__(self):
        return f"Report object, threshold set at {self.threshold} "
    
    def __repr__(self):
        return f"Report object, threshold set at {self.threshold} "
        


def get_overall_metrics(metrics, threshold = 0.5, dec = 3):
    """
    Return a latex friendly version of the accuracy recall and f1_score retrieved from a metrics dictionnary.
    """
    index = ["accuracy", "recall", "f1_score" ]
    values = [round(metrics[i], dec) for i in index]
    df = pd.DataFrame(values, index=index)
    df.columns = [""]
    return df

def save_metrics(metrics, path, figures = False):
    save_latek(osp.join(path, "prfs_per_class.tex"),
               df_prfs(metrics).to_latex())

    save_latek(osp.join(path, "prfs_macro.tex"),
              get_overall_metrics(metrics).to_latex())

    mat = metrics["confusion_matrix"]
    save_latek(osp.join(path, "confusion.tex"), mat.to_latex())
    save_latek(osp.join(path, "normalized_confusion.tex"), normalize_matrix(mat).to_latex())
    if figures :
        plot_confusion_matrix(normalize_matrix(mat))
        save_fig(osp.join(path, "normalized_confusion.pdf"))
        plt.cla()
        plot_confusion_matrix(mat);
        save_fig(osp.join(path, "confusion.pdf"))
    

def save_latek(path, latek):
    assert path[-4:] == ".tex", "your filename should have the extension .tex"
    with open(path, "w+") as f:
        f.write(latek)
        
def metrics(reports):
    last_epoch_reports = get_last_results(reports)
    metrics_at05 = list(last_epoch_reports["metrics_at_threshold"].values())[-1]
    return metrics_at05

def get_last_results(reports):
    return list(reports["results_at_epoch"].values())[-1]

def show_results(metrics, normalize_confusion = False, threshold = None):
    print(f"showing results with agreements of {threshold*100}%")
    plot_confusion_matrix(metrics["confusion_matrix"], normalize_confusion)
    display(df_prfs(metrics))

    for metric in ["accuracy","recall","f1_score"]:
        print(f"""\nthe {metric} is {metrics[metric].round(2)}""")

            
def df_prfs(metrics):
    return pd.DataFrame(metrics["precision_recall_fscore_support"],
                  columns = metrics["confusion_matrix"].columns,
                  index =["precision","recall","f1_score", "support"]).round(2)

def compute_metrics_for_threshold(results, labels, agreement_threshold, confusion_labels):
    labels_thresholded = threshold_array(labels, agreement_threshold)

    labels_argmax = np.argmax(labels[labels_thresholded], 1)
    results_argmax = np.argmax(results[labels_thresholded], 1)
    
    accuracy = sklearn.metrics.accuracy_score(labels_argmax, results_argmax)
    confusion_matrix = create_confusion_matrix(labels_argmax, results_argmax, confusion_labels)
    
    metrics_at_threshold = {              
              "agreement_threshold": agreement_threshold,
              "confusion_matrix" : confusion_matrix,
                "precision_recall_fscore_support" : sklearn.metrics.precision_recall_fscore_support(labels_argmax, results_argmax, labels = range(len(confusion_labels)), zero_division = 0),
               "recall" : sklearn.metrics.recall_score(labels_argmax, results_argmax, average="macro", zero_division = 0),
                "f1_score" :  sklearn.metrics.f1_score(labels_argmax, results_argmax, average="macro", zero_division = 0),
                "accuracy" : accuracy
    }
    
    return metrics_at_threshold


def create_confusion_matrix(gt, pred, labels, normalize = False):
    matrix = confusion_matrix(gt, pred, labels = list(range(len(labels))))
    matrix = pd.DataFrame(data = matrix, columns = labels, index = labels)
    if normalize:
        matrix = normalize_matrix(matrix)
    return matrix

def plot_confusion_matrix(confusion_matrix, normalize = False):
    plt.clf()
    if normalize:
        confusion_matrix = normalize_matrix(confusion_matrix)
    plt.figure(figsize=[len(confusion_matrix.columns)*1.4]*2);
    #cmap = sns.diverging_palette(220,5, n = 10, center = "dark")
    cmap = sns.light_palette(sns.crayons["Wild Strawberry"], as_cmap=True)
    fmt = ".2f" if normalize else ".2f" #Format the plot annotations
    
    sns.heatmap(confusion_matrix, annot=True, square = True, cbar = False, fmt = fmt, robust = True, cmap = cmap);
    plt.title("Predicted");
    plt.xlabel('Predicted');
    plt.ylabel("Actual");
    plt.tick_params(axis='both', which='major', labelsize=12, labelbottom = True, bottom=True, top = False, labeltop=True);
    return

def normalize_matrix(matrix, axis = 0, dec = 2):
    matrix = copy.copy(matrix)/matrix.sum(axis)
    matrix *= 100
    matrix = matrix.round(dec)
    matrix = matrix.fillna(0)
    return matrix


def threshold_array(array, threshold, strict = True):
    """
    return : The boolean sequence where the max of the array is above a threshold
    """
    if strict: 
        threshold = 0.999 if threshold == 1 else threshold
        return array.max(1) > threshold
    return array.max(1) >= threshold