"""
Evaluate performance of a Pytorch model
"""
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from tabulate import tabulate
import os

class ClassifierValidator():
    """
    This class is used to evaluate the performance of a classifier.
    
    Parameters
    ----------
        classifier: torch.nn.Module
            The classifier to evaluate.
        test_set: torch.utils.DataLoader
            The test set to evaluate the classifier on.
        device: torch.device
            The device to use for the classifier.
        val_set: list 
            A list of dictionaries containing the images that the classifier 
             has been validated on.

    Attributes
    ----------
        classifier: torch.nn.Module
            The classifier to evaluate.
        device: torch.device
            The device to use for the classifier.
        to_validate: torch.utils.DataLoader
            The set of images to validate the classifier on.
        validated_set: list
            A list of dictionaries containing the images that the classifier
            has been validated on.
        validation_categories: list
            A list of the categories that the classifier has been validated on.
        results_by_category: dict
            A dictionary of the results by category.
        false_pos: int
            The number of false positives.
        false_neg: int
            The number of false negatives.
        true_pos: int
            The number of true positives.
        true_neg: int
            The number of true negatives.
    """
    
    def __init__(self, classifier, test_set, device=None, val_set = None):
        self.classifier = classifier
        try:
            self.classifier.eval()
        except:
            print('Classifier is not a Pytorch model')
            pass
        self.device = device
        self.to_validate = test_set
        if val_set is not None:
            self.validated_set = val_set
        else:
            self.validated_set = {} 
        self.validation_categories = set()
        self.validated_masks = {}
        self.validated_refs = {}
        self.validated_class_refs = {}
        for img in self.to_validate:
            self.validation_categories.add(img['category'][0])
            self.validated_set[img['category'][0]] = []
            self.validated_masks[img['category'][0]] = None 
            self.validated_class_refs[img['category'][0]] = None 
            self.validated_refs[img['category'][0]] = None
        self.false_pos, self.true_pos, self.false_neg, self.true_neg = 0, 0, 0, 0
        self.results_by_category = {}
        
    def validate_classifier(self, verbose=False):
        """
        Evaluates the classifier on the validation set and saves output to self.validated_set.

        Arguments
        ---------
            verbose: bool
                If True, prints the progress of the validation.
        """
        with torch.no_grad():
            if verbose:
                print("Running classifier on validation set...")

            for img in self.to_validate:
                if self.device is not None:
                    classified_mask = self.classifier(img['img'].to(self.device))
                else:
                    classified_mask = self.classifier(img['img'])

                self.validated_set[img['category'][0]].append(0)
                if self.validated_masks[img['category'][0]] is None:
                    self.validated_masks[img['category'][0]] = classified_mask
                    self.validated_refs[img['category'][0]] = img['ref']
                    try:
                        self.validated_class_refs[img['category'][0]] = img['classmask']
                    except:
                        self.validated_class_refs[img['category'][0]] = img['ref']
                else:
                    self.validated_refs[img['category'][0]] = torch.cat((self.validated_refs[img['category'][0]], img['ref']))
                    try:
                        self.validated_class_refs[img['category'][0]] = torch.cat((self.validated_class_refs[img['category'][0]], img['classmask']))
                    except:
                        self.validated_class_refs[img['category'][0]] = torch.cat((self.validated_class_refs[img['category'][0]], img['ref']))
                    self.validated_masks[img['category'][0]] = torch.cat((self.validated_masks[img['category'][0]], classified_mask))

    def evaluate_classifier(self, cloud_threshold=0.5, save_on=True, verbose=False, buffer=0, get_fp_classes=False):
        """
        Evaluates the classifier given a cloud threshold.

        Arguments
        ---------
            cloud_threshold: float
                The threshold for the classifier to classify as cloud.
            save_on: bool
                If True, saves the results to self.results_by_category.
            verbose: bool
                If True, prints the progress of the evaluation.

        Returns
        -------
            false_pos: int
                The number of false positives.
            false_neg: int
                The number of false negatives.
            true_pos: int
                The number of true positives.
            true_neg: int
                The number of true negatives.
            results_by_category: dict
                A dictionary of the results by category.
        """
        #Validate classifier on val set to get classifier output on test set
        if len(self.validated_set[list(self.validated_set.keys())[0]]) == 0:
            self.validate_classifier()

        if verbose:
            print(f"Comparing classifier results to the reference mask with cloud_threshold={cloud_threshold}...")

        #Loop over test set
        false_pos, true_pos, false_neg, true_neg = 0, 0, 0, 0
        results_by_category = {}
        for category_name, category in self.validated_set.items():
            (fp, fn, tp, tn), fp_classes, all_classes = ClassifierValidator.compare_to_mask(self.validated_refs[category_name], self.validated_masks[category_name], self.validated_class_refs[category_name], cloud_threshold, buffer=buffer, get_fp_classes=get_fp_classes)
            results_by_category[category_name] = (fp, fn, tp, tn)
            false_pos += fp
            false_neg += fn
            true_pos += tp
            true_neg += tn
 
        #Save results
        if save_on:
            self.false_pos, self.true_pos, self.false_neg, self.true_neg = false_pos, true_pos, false_neg, true_neg
            self.results_by_category = results_by_category

        return false_pos, false_neg, true_pos, true_neg, results_by_category, fp_classes, all_classes

    @staticmethod
    def compare_to_mask_multiclass(ref, classifier_result, class_ref, cloud_threshold=0.5, buffer=0, fp_classes=[(0, 1, 5), (2, 3, 4, 5, 6), (0, 1, 2, 3, 4, 6)], get_fp_classes=False):
        """
        Compare to reference for multiclass input.
        """
        fp, fn, tp, tn = 0, 0, 0, 0
        fp_classes_out = []#{fp_class:0 for fp_class in fp_classes}
        all_classes = []#{gen_class:0 for gen_class in fp_classes}

        #Vectorized for speed
        class_mask = torch.argmax(classifier_result, dim=1)
        b, h, w, n_classes = ref.shape[0], ref.shape[2], ref.shape[3], ref.shape[4]

        class_metrics = []
        ref_mask = torch.argmax(ref, dim=4).reshape((b, h, w))
        if buffer != 0:
            #Convolve with 2*buffer+1 x 2*buffer+1 kernel of all ones to create buffered masks
            conv2d = torch.nn.Conv2d(in_channels=n_classes, out_channels=n_classes, kernel_size=2*buffer+1, padding=buffer, bias=False)
            conv2d.weight = torch.nn.Parameter(torch.ones((n_classes, n_classes, 2*buffer+1, 2*buffer+1))) # You can set anything you want.
            model = torch.nn.Sequential(conv2d)
            #buffered_classifier_mask = model(class_mask.to(torch.float32))
            buffered_ref_mask = model(ref.reshape(b, n_classes, h, w).to(torch.float32))
            zeros = torch.zeros_like(buffered_ref_mask)

            #Buffered masks are 1 if any pixel in the kernel surrounding the pixel is 1
            fp_ref_mask = (buffered_ref_mask > zeros)*1.0
            fn_ref_mask = (buffered_ref_mask == n_classes**2)*1.0
        else:
            fp_ref_mask = ref.reshape((b, n_classes, h, w))
            fn_ref_mask = ref.reshape((b, n_classes, h, w))

        for n_class in range(n_classes):
            class_fp_ref_mask = fp_ref_mask[:, n_class, :, :]
            class_fn_ref_mask = fn_ref_mask[:, n_class, :, :]
            fp = torch.sum((class_mask == n_class)*1.0 + (class_fp_ref_mask != 1.0)*1.0 == 2).item()
            fn = torch.sum((class_mask != n_class)*1.0 + (class_fn_ref_mask == 1.0)*1.0 == 2).item()
            tp = (torch.sum(class_mask == n_class) - fp).item()
            tn = (torch.sum(class_mask != n_class) - fn).item()
            class_metrics.append((fp, fn, tp, tn))
            if get_fp_classes:
                n_fp_classes = {}
                n_all_classes = {}
                for fp_class in fp_classes[n_class]:
                    n_fp_classes[fp_class] = torch.sum(((class_mask == n_class)*1.0 + (class_fp_ref_mask != 1.0)*1.0 + (class_ref*255 == fp_class)*1.0)==3).item()

                    n_all_classes[fp_class] = torch.sum((class_fp_ref_mask != 1.0)*1.0 + (class_ref*255 == fp_class)*1.0==2).item()
                fp_classes_out.append(n_fp_classes)
                all_classes.append(n_all_classes)
            else:
                n_fp_classes = {}
                n_all_classes = {}
                for fp_class in fp_classes[n_class]:
                    n_fp_classes[fp_class] = 0
                    n_all_classes[fp_class] = 0
                fp_classes_out.append(n_fp_classes)
                all_classes.append(n_all_classes)

        return class_metrics[0], fp_classes_out[0], all_classes[0]


    @staticmethod
    def compare_to_mask(ref, classifier_result, class_ref,cloud_threshold=0.5, buffer=0, fp_classes=(0, 1, 2, 3, 4, 6), get_fp_classes=False):
        """
        Compares the classifier result to the reference mask.

        Arguments
        ---------
            ref: torch.Tensor
                The reference mask.
            classifier_result: torch.Tensor
                The classifier result.
            cloud_threshold: float
                The threshold for the classifier to classify as cloud.

        Returns
        -------
            metrics: tuple
                A tuple of the form (false_pos, false_neg, true_pos, true_neg)
        """
        fp, fn, tp, tn = 0, 0, 0, 0
        fp_classes = {fp_class:0 for fp_class in fp_classes}
        all_classes = {gen_class:0 for gen_class in fp_classes}

        #Vectorized for speed
        class_mask = classifier_result >= cloud_threshold
        neg_mask = classifier_result < cloud_threshold
        if buffer == 0:
            fp = torch.sum(class_mask > ref).item() #converts tensor to int
            if get_fp_classes:
                for fp_class in fp_classes.keys():
                    fp_classes[fp_class] = torch.sum(((class_mask > ref)*1.0 + (class_ref*255 == fp_class)*1.0)==2).item()

                    all_classes[fp_class] = torch.sum(((ref ==0)*1.0 + (class_ref*255 == fp_class)*1.0)==2).item()
            tp = (torch.sum(class_mask)-fp).item()
            fn = (torch.sum(class_mask < ref)).item()
            tn = (torch.sum(neg_mask) - fn).item()
        else:
            #Convolve with 2*buffer+1 x 2*buffer+1 kernel of all ones to create buffered masks
            conv2d = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2*buffer+1, padding=buffer, bias=False)
            conv2d.weight = torch.nn.Parameter(torch.ones((1, 1, 2*buffer+1, 2*buffer+1))) # You can set anything you want.
            model = torch.nn.Sequential(conv2d)
            buffered_classifier_mask = model(class_mask.to(torch.float32))
            buffered_ref_mask = model(ref.to(torch.float32))
            zeros = torch.zeros_like(buffered_classifier_mask)

            #Buffered masks are 1 if any pixel in the kernel surrounding the pixel is 1
            buffered_classifier_mask = buffered_classifier_mask > zeros
            buffered_ref_mask = buffered_ref_mask > zeros

            #FP if in classifier mask but not in buffered ref mask
            fp = torch.sum(class_mask > buffered_ref_mask).item() #converts tensor to int
            if get_fp_classes:
                for fp_class in fp_classes.keys():
                    fp_classes[fp_class] = torch.sum(((class_mask > buffered_ref_mask)*1.0 + (class_ref*255 == fp_class)*1.0)==2).item()
                    all_classes[fp_class] = torch.sum(((buffered_ref_mask < 1)*1.0 + (class_ref*255 == fp_class)*1.0)==2).item()

            #FN if in ref mask but not in buffered classifier mask
            fn = torch.sum(ref > buffered_classifier_mask).item()

            #TP if in classifier mask and buffered ref mask
            tp = (torch.sum(class_mask)-fp).item()

            #TN if in neg mask and buffered neg mask
            tn = (torch.sum(neg_mask) - fn).item()

        return (fp, fn, tp, tn), fp_classes, all_classes
    
    def generate_confusion_matrix(self, cloud_threshold=0.5, category=None, title="", from_saved=False, plot_on=False, verbose=False, buffer=0):
        """
        Generates a confusion matrix for the classifier given a cloud threshold.

        Arguments
        ---------
            cloud_threshold: float
                The threshold for the classifier to classify as cloud.
            category: str
                The category to generate the confusion matrix for.
            title: str
                The title of the confusion matrix plot.
            from_saved: bool
                If True, uses the results from the last evaluation.
            plot_on: bool
                If True, plots the confusion matrix.
            verbose: bool
                If True, prints the progress of the evaluation.

        Returns
        -------
            confusion_matrix: np.array
                The confusion matrix.
            fig: matplotlib.pyplot.figure
                The figure of the confusion matrix.
            ax: matplotlib.pyplot.axes
                The axes of the confusion matrix.
        """
        #Get results
        if from_saved:
            fp, fn, tp, tn, results_by_category = self.false_pos, self.false_neg, self.true_pos, self.true_neg, self.results_by_category
        else:
            fp, fn, tp, tn, results_by_category, _, _ = self.evaluate_classifier(cloud_threshold, buffer=buffer)

        #Narrow down on category if needed
        if category in results_by_category.keys():
            if verbose:
                print(f"Plotting confusion matrix for data in the {category} category...")
            (fp, fn, tp, tn) = results_by_category[category]
        elif verbose:
            print("Plotting confusion matrix for all data...")
            
        #Generate confusion matrix
        confusion_table = np.array([[tn, fp],
                   [fn, tp]])
        fig, ax = plot_confusion_matrix(conf_mat=confusion_table,
                                        show_absolute=False,
                                        show_normed=True,
                                        colorbar=True,
                                        class_names=["Not cloud", "Cloud"])
        #Plot if needed
        if plot_on:
            plt.title(title)
            plt.subplots_adjust(bottom=0.23)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            plt.show()

        #Return fig, ax for further use in addition to table
        return confusion_table, fig, ax

    def generate_fp_class_table(self, cloud_threshold=0.5, buffer=0, fp_names=('Cloud Shadow', 'Cloud Shadow over Water', 'Water', 'Ice/Snow', 'Land', 'Flooded')):
        #Evaluate classifier with appropriate threshold
        fp, fn, tp, tn, _, fp_classes, all_classes = self.evaluate_classifier(cloud_threshold, save_on=True, buffer=buffer, get_fp_classes=True)
        fp_classes_keys_sorted = sorted(fp_classes.keys())
        fps = []
        alls = []
        ratios = []
        sum_fp_classes = np.sum(np.array(list(fp_classes.values())))
        sum_all_classes = np.sum(np.array(list(all_classes.values())))
        for key in fp_classes_keys_sorted:
            fps.append(fp_classes[key]*100.0/sum_fp_classes)
            alls.append(all_classes[key]*100.0/sum_all_classes)
        for i, fp_cat in enumerate(fps):
            ratios.append(fp_cat/alls[i])

        #Generate LaTeX table
        #Generate header
        latex_str = "\\begin{table}\n\
        \centering\n\
        \caption{Insert caption here}\n\
        \label{table:insert_label_here}\n\
        \\begin{tabular}{@{}ccccccc@{}}\n\
        \\toprule\n\
        \\textbf{Category} & \\textbf{Cloud Shadow} & \\textbf{Cloud Shadow over Water} & \\textbf{Water} & \\textbf{Ice/Snow} & \\textbf{Land} & \\textbf{Flooded} \\\ \n \midrule\n"

        #Fill in data
        latex_str += "\% of False Positives &"
        for pct in ratios:
            latex_str += f"{pct:.2f} &"
        latex_str = latex_str[:-2]
        latex_str += "\\\ \n \midrule \n"
        latex_str += "\% of Total Non-Cloud Pixels &"
        for pct in alls:
            latex_str += f"{pct:.2f}\% &"
        latex_str = latex_str[:-2]
        latex_str += "\\\ \n \\bottomrule\n\
        \end{tabular}\n\
        \end{table}"

        return tabulate([ratios], headers=list(fp_names), tablefmt='orgtbl'), latex_str

                    
    def generate_comparison_table(self, cloud_threshold=0.5, buffer=0):
        """
        Generates a table of the results of the classifier.

        Arguments 
        ---------
            cloud_threshold: float
                The threshold for the classifier to classify as cloud.

        Returns
        -------
            table: str
                A string representation of the table.
        """
        #Evaluate classifier with appropriate threshold
        self.evaluate_classifier(cloud_threshold, save_on=True, buffer=buffer)

        #For each category, parse results
        cats = []
        total_fp, total_fn, total_tp, total_tn = 0, 0, 0, 0
        for cat_result in self.results_by_category.items():
            cat, (fp, fn, tp, tn) = cat_result 
            total_fp += fp
            total_fn += fn
            total_tp += tp
            total_tn += tn
            acc = (tn+tp)/(1.0*(fp+fn+tp+tn))
            sensitivity = 0
            if tp+fn > 0:
                sensitivity = (tp/(tp+fn))
            specificity = (tn/(tn+fp)) if tn+fp > 0 else 0
            recall = sensitivity
            precision = (tp/(tp+fp)) if tp+fp > 0 else 0
            f1 = (2*precision*recall)/(precision+recall) if precision + recall > 0 else 0
            iou = (precision*recall)/(precision+recall-precision*recall) if (precision + recall - precision*recall) > 0 else 0
            bal_acc = (sensitivity+specificity)/2.0
            res_cat = [cat, acc, bal_acc, sensitivity, specificity, precision, recall, f1, iou]
            cats.append(res_cat)

        #Parse total results
        total_acc = (total_tn+total_tp)/(1.0*(total_fp+total_fn+total_tp+total_tn))
        total_sensitivity = 0
        if total_tp+total_fn > 0:
            total_sensitivity = (total_tp/(total_tp+total_fn))
        total_specificity = (total_tn/(total_tn+total_fp))
        total_bal_acc = (total_sensitivity+total_specificity)/2.0
        total_recall = total_sensitivity
        total_precision = (total_tp/(total_tp+total_fp)) if total_tp+total_fp > 0 else 0
        total_f1 = (2*total_precision*total_recall)/(total_precision+total_recall) if total_precision + total_recall > 0 else 0
        total_iou = (total_precision*total_recall)/(total_precision+total_recall-total_precision*total_recall) if (total_precision + total_recall - total_precision*total_recall) > 0 else 0
        total_res = ["Total", total_acc, total_bal_acc, total_sensitivity, total_specificity, total_precision, total_recall, total_f1, total_iou]
        cats.append(total_res)

        #Generate LaTeX table
        #Generate header
        latex_str = "\\begin{table}\n\
        \centering\n\
        \caption{Insert caption here}\n\
        \label{table:insert_label_here}\n\
        \\begin{tabular}{@{}lrrrrrrrr@{}}\n\
        \\toprule\n\
        \\textbf{Scene Type} & \\textbf{Acc.} & \\textbf{Bal. Acc.} & \\textbf{Sens.} & \\textbf{Spec.} & \\textbf{Prec.} & \\textbf{Recall}  & \n\
        \\textbf{$\\textrm{F}_1$} & \\textbf{IoU} \\\ \n \midrule\n"

        #Fill in data
        for i in range(len(cats)-1):
            [cat, acc, bal_acc, sensitivity, specificity, precision, recall, f1, iou] = cats[i] 
            acc, bal_acc, sensitivity, specificity, precision, recall = 100*acc, 100*bal_acc, 100*sensitivity, 100*specificity, 100*precision, 100*recall
            latex_str += f"{cat.capitalize()} & {acc:.2f}\% & {bal_acc:.2f}\% & {sensitivity:.2f}\% & {specificity:.2f}\% & {precision:.2f}\% & {recall:.2f}\% & {f1:.4f} & {iou:.4f} \\\ \n"

        #Fill in totals and footer
        totals = cats[-1]
        [total_cat, total_acc, total_bal_acc, total_sensitivity, total_specificity, total_precision, total_recall, total_f1, total_iou] = totals
        total_acc, total_bal_acc, total_sensitivity, total_specificity, total_precision, total_recall = 100*total_acc, 100*total_bal_acc, 100*total_sensitivity, 100*total_specificity, 100*total_precision, 100*total_recall
        latex_str += "\midrule \n"
        latex_str += f"\\textbf{{Overall}} & \\textbf{{{total_acc:.2f}\%}} & \\textbf{{{total_bal_acc:.2f}\%}} & \\textbf{{{total_sensitivity:.2f}\%}} & \\textbf{{{total_specificity:.2f}\%}} & \\textbf{{{total_precision:.2f}\%}} & \\textbf{{{total_recall:.2f}\%}} & \\textbf{{{total_f1:.4f}}} & \\textbf{{{total_iou:.4f}}} \\\ \n"
        latex_str += "\\bottomrule\n\
        \end{tabular}\n\
        \end{table}"

        return tabulate(cats, headers=['Category', 'Acc', 'Bal Acc', 'Sens', 'Spec', 'Prec', 'Recall', 'F1', 'IoU'], tablefmt='orgtbl'), latex_str
        
    def generate_roc_curve(self, roc_thresholds=[0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0], plot_on=False, title="", classifier_label="", adaptive_threshold=True, buffer=0):
        """
        Generates a ROC curve for the classifier.

        Arguments
        ---------
            roc_thresholds: list
                The thresholds to use for the ROC curve.
            plot_on: bool
                If True, plots the ROC curve.
            title: str
                The title of the ROC curve plot.
            classifier_label: str
                The label of the classifier in the plot legend.

        Returns
        -------
            roc_fprates: list
                The false positive rates for the ROC curve.
            roc_tprates: list
                The true positive rates for the ROC curve.
        """
        #Evaluate classifier at each threshold
        thold_fps, thold_fns, thold_tps, thold_tns = [], [], [], []
        if not adaptive_threshold:
            for thold in roc_thresholds:
                fp, fn, tp, tn, _, _, _ = self.evaluate_classifier(thold, save_on=False, buffer=buffer)
                thold_fps.append(fp)
                thold_fns.append(fn)
                thold_tps.append(tp)
                thold_tns.append(tn)
                #Calculate false and true positive rates at each threshold
                roc_tprates, roc_fprates  = [], []
                for i in range(len(thold_fps)):
                    roc_fprates.append(thold_fps[i]/(thold_fps[i]+thold_tns[i]))
                for i in range(len(thold_fns)):
                    roc_tprates.append(thold_tps[i]/(thold_fns[i]+thold_tps[i]))
        else:
            tholds = [0.0, 1.0]
            adaptive_gap_size = 0.01
            min_step_size = 0.001
            roc_fprates = []
            roc_tprates = []
            fps, fns, tps, tns = [], [], [], []
            for thold in tholds:
                fp, fn, tp, tn, _, _, _ = self.evaluate_classifier(thold, save_on=False, buffer=buffer)
                fps.append(fp)
                fns.append(fn)
                tps.append(tp)
                tns.append(tn)

                fprate = fp/(fp+tn) if fp+tn > 0 else 0
                tprate = tp/(tp+fn) if tp+fn > 0 else 0
                roc_fprates.append(fprate)
                roc_tprates.append(tprate)
            gaps = []
            for i in range(len(roc_fprates)-1):
                gap = roc_fprates[i]-roc_fprates[i+1]
                gaps.append(gap)
            add_tholds = []
            for i in range(len(gaps)):
                if gaps[i] > adaptive_gap_size and np.abs(tholds[i]-tholds[i+1]) > min_step_size:
                    add_tholds.append(i + len(add_tholds))
            while len(add_tholds) > 0:
                for ind in add_tholds:
                    thold = (tholds[ind] + tholds[ind+1])/2
                    tholds.insert(ind+1, thold)
                    fp, fn, tp, tn, _, _, _= self.evaluate_classifier(thold, save_on=False, buffer=buffer)
                    fps.insert(ind+1, fp)
                    fns.insert(ind+1, fn)
                    tps.insert(ind+1, tp)
                    tns.insert(ind+1, tn)

                    roc_fprates.insert(ind+1, fp/(fp+tn))
                    roc_tprates.insert(ind+1, tp/(tp+fn))
                new_gaps = []
                for i in range(len(roc_fprates)-1):
                    gap = roc_fprates[i]-roc_fprates[i+1]
                    new_gaps.append(gap)
                gaps = new_gaps
                add_tholds = []
                for i in range(len(gaps)):
                    if gaps[i] > adaptive_gap_size and np.abs(tholds[i]-tholds[i+1]) > min_step_size:
                        add_tholds.append(i + len(add_tholds))
            gaps = []
            for i in range(len(roc_tprates)-1):
                gap = roc_tprates[i]-roc_tprates[i+1]
                gaps.append(gap)
            add_tholds = []
            for i in range(len(gaps)):
                if gaps[i] > adaptive_gap_size and np.abs(tholds[i]-tholds[i+1]) > min_step_size:
                    add_tholds.append(i + len(add_tholds))
            while len(add_tholds) > 0:
                for ind in add_tholds:
                    thold = (tholds[ind] + tholds[ind+1])/2
                    tholds.insert(ind+1, thold)
                    fp, fn, tp, tn, _, _, _ = self.evaluate_classifier(thold, save_on=False, buffer=buffer)
                    fps.insert(ind+1, fp)
                    fns.insert(ind+1, fn)
                    tps.insert(ind+1, tp)
                    tns.insert(ind+1, tn)

                    roc_fprates.insert(ind+1, fp/(fp+tn))
                    roc_tprates.insert(ind+1, tp/(tp+fn))
                new_gaps = []
                for i in range(len(roc_fprates)-1):
                    gap = roc_tprates[i]-roc_tprates[i+1]
                    new_gaps.append(gap)
                gaps = new_gaps
                add_tholds = []
                for i in range(len(gaps)):
                    if gaps[i] > adaptive_gap_size and np.abs(tholds[i]-tholds[i+1]) > min_step_size:
                        add_tholds.append(i + len(add_tholds))
        #roc_fprates.insert(0, 0)
        #roc_tprates.insert(0, 0)
        roc_fprates.append(0)
        roc_tprates.append(0)

        AUC = 0
        for i in range(1, len(roc_fprates)):
            AUC += (-roc_fprates[i]+roc_fprates[i-1])*(roc_tprates[i-1]+roc_tprates[i])/2.0
        thold = self.get_best_threshold(fps, fns, tps, tns, tholds)

        #Plot if needed
        if plot_on:
            plt.figure()
            ref = [0, 1]
            plt.plot(roc_fprates, roc_tprates)
            plt.plot(ref, ref, 'k--')
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title(title)
            plt.legend([classifier_label, "Reference (y=x)"])
            plt.tight_layout()
            plt.grid()
            plt.show()

        return roc_fprates, roc_tprates, AUC, thold

    def thold_score(self, fp, fn, tp, tn):
        sensitivity = 0
        if tp+fn > 0:
            sensitivity = (tp/(tp+fn))
        specificity = (tn/(tn+fp)) if tn+fp > 0 else 0
        recall = sensitivity
        precision = (tp/(tp+fp)) if tp+fp > 0 else 0
        f1 = (2*precision*recall)/(precision+recall) if precision + recall > 0 else 0
        return 1-f1


    def get_best_threshold(self, fprates, fnrates, tprates, tnrates, tholds):
        scores = []
        for i, fp in enumerate(fprates):
            score = self.thold_score(fprates[i], fnrates[i], tprates[i], tnrates[i])
            scores.append(score)
        ind = scores.index(min(scores))
        return tholds[ind]



    @staticmethod
    def compare_to_mask_buffer(ref, classifier_result, buffer, cloud_threshold=0.5):
        """
        Compares the classifier result to the reference mask.
        Arguments
        ---------
            ref: torch.Tensor
                The reference mask.
            classifier_result: torch.Tensor
                The classifier result.
            cloud_threshold: float
                The threshold for the classifier to classify as cloud.
        Returns
        -------
            metrics: tuple
                A tuple of the form (false_pos, false_neg, true_pos, true_neg)
        """
        fp, fn, tp, tn = 0, 0, 0, 0

        #Vectorized for speed
        class_mask = classifier_result >= cloud_threshold
        neg_mask = classifier_result < cloud_threshold
        neg_ref_mask = ref == 0

        #Convolve with 2*buffer+1 x 2*buffer+1 kernel of all ones to create buffered masks
        conv2d = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2*buffer+1, padding=buffer, bias=False)
        conv2d.weight = torch.nn.Parameter(torch.ones((1, 1, 2*buffer+1, 2*buffer+1))) # You can set anything you want.
        model = torch.nn.Sequential(conv2d)
        buffered_classifier_mask = model(class_mask.to(torch.float32))
        buffered_ref_mask = model(ref.to(torch.float32))
        buffered_neg_mask = model(neg_mask.to(torch.float32))
        buffered_neg_ref_mask = model(neg_ref_mask.to(torch.float32))
        zeros = torch.zeros_like(buffered_classifier_mask)

        #Buffered masks are 1 if any pixel in the kernel surrounding the pixel is 1
        buffered_classifier_mask = buffered_classifier_mask > zeros
        buffered_ref_mask = buffered_ref_mask > zeros
        buffered_neg_mask = buffered_neg_mask > zeros
        buffered_neg_ref_mask = buffered_neg_ref_mask > zeros

        #FP if in classifier mask but not in buffered ref mask
        fp = torch.sum(class_mask > buffered_ref_mask).item() #converts tensor to int

        #FN if in ref mask but not in buffered classifier mask
        fn = torch.sum(ref > buffered_classifier_mask).item()

        #TP if in classifier mask and buffered ref mask
        tp = (torch.sum(class_mask)-fp).item()

        #TN if in neg mask and buffered neg mask
        tn = (torch.sum(neg_mask) - fn).item()

        error_mask = (class_mask > buffered_ref_mask).to(torch.float32) - (ref > buffered_classifier_mask).to(torch.float32)
        un_buffered_error_mask = (class_mask > ref).to(torch.float32) - (ref > class_mask).to(torch.float32)

        buffered_mask_pretty = (1/3)*(class_mask.to(torch.float32)*3 + ((buffered_classifier_mask.to(torch.float32) - class_mask.to(torch.float32))>0).to(torch.float32) + ((class_mask.to(torch.float32) - buffered_neg_mask.to(torch.float32))>0).to(torch.float32))
        buffered_ref_pretty = (1/3)*(ref.to(torch.float32)*3 + ((buffered_ref_mask.to(torch.float32) - ref.to(torch.float32))>0).to(torch.float32) + ((ref.to(torch.float32) - buffered_neg_ref_mask.to(torch.float32))>0).to(torch.float32))

        unbuffer_fp = torch.sum(class_mask > ref).item() #converts tensor to int
        unbuffer_tp = (torch.sum(class_mask)-fp).item()
        unbuffer_fn = (torch.sum(class_mask < ref)).item()
        unbuffer_tn = (torch.sum(neg_mask) - fn).item()


        return (fp, fn, tp, tn), (unbuffer_fp, unbuffer_fn, unbuffer_tp, unbuffer_tn), error_mask, un_buffered_error_mask, buffered_mask_pretty, buffered_ref_pretty
 
