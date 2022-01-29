import numpy as np
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch import nn
import sklearn
from IPython.display import clear_output
from tqdm.notebook import tqdm as tqdm_notebook
import seaborn as sns
from IPython.display import display
import plotly.express as px
import copy
import os
import sys
import pickle
import random
from IPython.display import clear_output


model_data_save_path = "../../../results/model_data/"
MAX_REPORTS = 10


def get_empty_reports():
    return { "labels": None, "results_at_epoch": {}}

global device 
device = "cuda" if torch.cuda.is_available() else "cpu"

def train_n_epochs(model, loss_fn, optimizer, epochs, loader, device):
    model = model.train().to(device)
    epoch_loss = 0
    result_tracker = []
    for epoch in range(epochs):
        for data in tqdm_notebook(loader):
            inputs = data["image"].to(device)
            targets = data["label"].to(device)
            
            optimizer.zero_grad()
            
            prediction = model(inputs)

            loss = loss_fn(prediction, targets)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        result_tracker.append(epoch_loss)
        epoch_loss = 0
    return result_tracker





def copy_model_data(model_data):
    """
    deprecated
    """
    return Model_data(model = copy.deepcopy(model_data.model),
                     loss_fn = model_data.loss_fn,
                    optimizer_fn = model_data.optimizer_fn,
                     lr = model_data.lr,
                     data_loaders = model_data.data_loaders,
                     device = device)



class Model_data :
    def __init__(self, model, loss_fn, optimizer_fn, lr, data_loaders, device):
        self._model = model
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self._lr = lr
        self.data_loaders = data_loaders
        self.device = device
        self.epochs_trained_on = 0
        self.reports = get_empty_reports()
        self.classification_accuracy = []
        self.validation_losses = [np.inf]
        self.epoch_losses = []
        self.set_optimizer()
        self.new_id()
    
    @property
    def model(self):
        return self._model
    
    @property
    def lr(self):
        return self._lr
    
    @lr.setter
    def lr(self, new_lr):
        self._lr = new_lr
        self.set_optimizer()
        
    @model.setter
    def model(self, new_model):
        self._model = new_model
        self.set_optimizer()
    
#    def __call__(self, args):
        
    def new_id(self):
        self.id = hex(random.randint(0, sys.maxsize))
           
    def set_optimizer(self):
        self.optimizer = self.optimizer_fn(self.model.parameters(), lr=self.lr)
        print("Optimizer successfully updated. :)")
        
        
    def test_network(self, test_epochs = 50):
        """

        """
        for batch_to_overfit in self.data_loaders["train"]:
            break
            
        losses = train_n_epochs(model = self.model,
                       loss_fn = self.loss_fn,
                       optimizer=self.optimizer,
                       epochs=test_epochs,
                       loader=[batch_to_overfit],
                        device = self.device)
        
        results, labels = evaluate_on_dataset_classifier(self.model, {"test" : [batch_to_overfit]}, "test", device)
        tresh = threshold_array(labels.detach().cpu().numpy(), 0.5)
        results = results[tresh]
        labels = labels[tresh]
        accuracy = sklearn.metrics.accuracy_score(np.argmax(labels,1), np.argmax(results,1))
        assert  accuracy == 1, f"The model was not able to overfit on a single batch, something is wrong. He got an accuracy of {accuracy}"
        clear_output()
        print("Your model is running correctly. Rejoice ! :-)")
        
    
    def compute_accuracy(self, subset = "test"):
        """
        The accuracy is not thresholded
        """
        results, labels = evaluate_on_dataset_classifier(self.model, self.data_loaders, subset, device)
        accuracy = sklearn.metrics.accuracy_score(np.argmax(labels.cpu().detach().numpy(),1), np.argmax(results.cpu().detach(),1))
        return accuracy
        
    def train_n_epochs(self, n_epochs):
        print(f"The model was trained for {self.epochs_trained_on} epochs, training for {n_epochs} more!")
        self.epoch_losses += train_n_epochs(model = self.model,
                                            loss_fn = self.loss_fn,
                                            optimizer = self.optimizer,
                                            epochs = n_epochs,
                                            loader = self.data_loaders["train"],
                                            device = self.device)
        print(f"Success, training losses are {self.epoch_losses[-n_epochs:]}")
        self.epochs_trained_on += n_epochs
        
    def get_last_metrics(self):
        return get_last_metrics(self.reports)
    
    def get_last_results(self):
        return get_last_results(self.reports)
    
    def get_last_arf1(self):
        get_overall_metrics(self.get_last_metrics())
    
    def export_results(self, path):
        export_results(self.get_last_metrics(), path)

    def save(self, results_path):
        """
        Not yet tested,
        Missing the loading part
        """
        models_path = results_path + f"models/"
        reports_path = results_path + f"reports/"
        
        if "models" not in os.listdir(results_path):
            os.mkdir(models_path)
            print(f"Models file created")

        if "reports" not in os.listdir(results_path):
            os.mkdir(reports_path)
            print(f"reports file created")
        
        self.save_model(models_path)
        self.save_reports(reports_path)
    
    def save_model(self, models_path):
        torch.save(self.model.state_dict(), models_path + f"{self.id}.pt")
        
    def save_reports(self, reports_path):
        with open(reports_path + str(self.id)+".bin", "wb") as f:
            pickle.dump(self.reports, f)
             
    
                
                
def get_overall_metrics(metrics, dec = 3):
    """
    Return a latex friendly version of the accuracy recall and f1_score retrieved from a metrics dictionnary.
    """
    index = ["accuracy", "recall", "f1_score" ]
    values = [round(metrics[i], dec) for i in index]
    df = pd.DataFrame(values, index=index)
    df.columns = [""]
    return df

def export_results(metrics, path):
    save_latek(osp.join(path, "ptfs_per_class.tex"),
               ct.df_prfs(metrics).to_latex()) #macro recall, f1_score, accuracy
    
    save_latek(osp.join(path, "ptfs_overall.tex"),
              get_overall_metrics(metrics).to_latex()) #metrics
    
    mat = metrics["confusion_matrix"]
    save_latek(osp.join(path, "confusion.tex"), mat.to_latex()) #confusion -> Values
    save_latek(osp.join(path, "normalized_confusion.tex"), ct.normalize_matrix(mat).to_latex()) #confusion -> Normed_Values
    ct.plot_confusion_matrix(ct.normalize_matrix(mat))
    save_fig(osp.join(path, "normalized_confusion.pdf"))
    plt.cla()
    ct.plot_confusion_matrix(mat)
    dt.save_fig(osp.join(path, "confusion.pdf"))
    

    
def save_latek(path, latek):
    assert path[-4:] == ".tex", "your filename should have the right extension"
    with open(path, "w+") as f:
        f.write(latek)
        
def get_last_metrics(reports):
    last_epoch_reports = get_last_results(reports)
    metrics_at05 = list(last_epoch_reports["metrics_at_threshold"].values())[-1]
    return metrics_at05

def get_last_results(reports):
    return list(reports["results_at_epoch"].values())[-1]

#Transformations for the pickle data_loader
def trim(trim_size):
    def batch_trimmer(batch_to_trim):
        batch_to_trim["image"] = batch_to_trim["image"][:,:-trim_size]
        return batch_to_trim
    return batch_trimmer

def max_pool(n_classes=9):
    def batch_max_pooler(batch_to_maxpool):
        batch_to_maxpool["label"] = nn.functional.one_hot(batch_to_maxpool["label"].argmax(1), n_classes)
        return batch_to_maxpool
    return batch_max_pooler

@torch.no_grad()
def evaluate_on_dataset_classifier(model, data_loaders, subset, device):
    model.eval()
    results = torch.tensor([])
    all_labels = torch.tensor([])
    for batch in tqdm_notebook(data_loaders[subset]):
        img = batch['image']
        labels = batch['label'] # emotion_distribution
        logits = model(img).cpu()
        results = torch.cat([results, logits])
        all_labels = torch.cat([all_labels, labels])
    return results, all_labels

def threshold_array(array, threshold):
    """
    return : The boolean sequence where the max of the array is above a threshold
    """

    return (array.max(1) >= threshold)

def create_report(model_data, confusion_labels, normalize_confusion = False, show_fig = True, agreement_threshold = 0.5):
    """
    A function to create a report on the current model performance. And store it inside the model data.
    Should be updated to return the report.

    Parameters
    ----------
    model_data : a Model_data object
    second : confusion
    third : {'value', 'other'}, optional
        the 3rd param, by default 'False'

    Returns
    -------
    a nested dict object with 
    report : {result at epoch : {0 : {results : result_0, metric_at_threshold : {threshold 0.1 : report_01, threshold 0.2 :report_02}} 
                                1: {threshold 0.1 : report_11, threshold 0.2 :report_12} }
    }
    

    """
    #get test results
    if model_data.epochs_trained_on not in model_data.reports["results_at_epoch"].keys():
        results, labels = evaluate_on_dataset_classifier(model_data.model, model_data.data_loaders, "test", device)
        results = results.numpy()
        labels = labels.numpy()
        model_data.reports["results_at_epoch"][model_data.epochs_trained_on]= {"results":results, "metrics_at_threshold":{}}
           
    else :
        results = model_data.reports["results_at_epoch"][model_data.epochs_trained_on]["results"]
        labels = model_data.reports["labels"]
        
    
    
    model_data.reports["labels"] = labels
    metrics_for_thresholds = model_data.reports["results_at_epoch"][model_data.epochs_trained_on]["metrics_at_threshold"]
    
    # get metrics
    if agreement_threshold not in metrics_for_thresholds.keys():
        metrics = compute_metrics_for_threshold(results = results,
                                                labels = labels,
                                                agreement_threshold = agreement_threshold,
                                                confusion_labels = confusion_labels)      
    else:
        metrics = metrics_for_thresholds[agreement_threshold]
        
    metrics_for_thresholds[agreement_threshold] = metrics
    model_data.classification_accuracy.append(metrics["accuracy"])
    # Avoid saving too many results
    if len(model_data.reports["results_at_epoch"].keys()) >= MAX_REPORTS:
        print(f"Max thresholds count {MAX_REPORTS} reached printing report without saving.")
        show_results(model_data = model_data, epoch = model_data.epochs_trained_on, threshold = agreement_threshold)
        del(metrics_for_thresholds[agreement_threshold])
        
    if show_fig:
        show_results(model_data = model_data,
                     epoch = model_data.epochs_trained_on,
                     normalize_confusion = normalize_confusion,
                     threshold = agreement_threshold)

    return 

def show_results(model_data, epoch, normalize_confusion = False, threshold = None):
        try :
            results = model_data.reports["results_at_epoch"][epoch]
            if not threshold : threshold = list(results["metrics_at_threshold"].keys())[-1]
            metrics = results["metrics_at_threshold"][threshold]
            
            print(f"showing results with agreements of {threshold*100}% at epoch n°{epoch}\n")
            plot_confusion_matrix(metrics["confusion_matrix"], normalize_confusion)
            
            display(df_prfs(metrics))
            
            for metric in list(metrics.keys())[-3:]:
                print(f"""\nthe {metric} is {metrics[metric].round(2)}""")
            
        except (KeyError, NameError) as e :
            print(e)
            print(f"results for threshold {threshold} not found")

            
def df_prfs(metrics):
    return pd.DataFrame(metrics["precision_recall_fscore_support"],
                  columns = metrics["confusion_matrix"].columns,
                  index =["precision","recall","f1_score", "support"]).round(2)

def save_report():
    try :
        save_dir = path_to_reports + f"{model_data.id}"
        os.mkdir(save_dir)
        print(f"file for id {model_data.id} created")
    except FileExistsError :
        print(f"file for id {model_data.id} already exists")

def compute_metrics_for_threshold(results, labels, agreement_threshold, confusion_labels):
    labels_thresholded = threshold_array(labels, agreement_threshold)

    labels_argmax = np.argmax(labels[labels_thresholded], 1)
    results_argmax = np.argmax(results[labels_thresholded], 1)
    
    accuracy = sklearn.metrics.accuracy_score(labels_argmax, results_argmax)
    confusion_matrix = create_confusion_matrix(labels_argmax, results_argmax, confusion_labels)
    
    metrics_at_threshold = {              
              "agreement_threshold": agreement_threshold,
              "labels_thresholded" : labels_thresholded,
              "confusion_matrix" : confusion_matrix,
                "precision_recall_fscore_support" : sklearn.metrics.precision_recall_fscore_support(labels_argmax, results_argmax),
               "recall" : sklearn.metrics.recall_score(labels_argmax, results_argmax, average="macro"),
                "f1_score" :  sklearn.metrics.f1_score(labels_argmax, results_argmax, average="macro"),
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
    plt.figure(figsize=[len(confusion_matrix.columns)*1.4]*2)
    #cmap = sns.diverging_palette(220,5, n = 10, center = "dark")
    cmap = sns.light_palette(sns.crayons["Wild Strawberry"], as_cmap=True)
    fmt = ".2f" if normalize else ".2f" #Format the plot annotations
    
    sns.heatmap(confusion_matrix, annot=True, square = True, cbar = False, fmt = fmt, robust = True, cmap = cmap);
    plt.title("Predicted")
    plt.xlabel('Predicted')
    plt.ylabel("Actual")
    plt.tick_params(axis='both', which='major', labelsize=12, labelbottom = False, bottom=True, top = False, labeltop=True)
    sns.despine()
    
@torch.no_grad()
def eval_loss(model_data, subset = "val"):
    eval_set = model_data.data_loaders[subset]
    epoch_loss = 0
    for data in eval_set:
        inputs = data["image"].to(device)
        targets = data["label"].to(device)
        prediction = model_data.model(inputs)
        loss = model_data.loss_fn(prediction, targets)
        epoch_loss += loss.item()
    return epoch_loss
    

def train_eval(model_data, epochs_per_lr, lrs, run_check = False):
    for lr in lrs:
        for epoch in range(epochs_per_lr):
            model_data.lr = lr
            previous_state = copy.copy(model_data.model.state_dict())
            model_data.train_n_epochs(1)
            val_loss = eval_loss(model_data)
            print(f"val loss is {val_loss}")
            if val_loss > model_data.validation_losses[-1] :
                model_data.model.load_state_dict(previous_state)
                print(f"stoped at learning rate {lr}")
                break     
            model_data.validation_losses.append(val_loss)
    return
  

def normalize_matrix(matrix, axis = 0, dec = 2):
    matrix = copy.copy(matrix)/matrix.sum(axis)
    matrix *= 100
    matrix = matrix.round(dec)
    matrix = matrix.fillna(0)
    return matrix