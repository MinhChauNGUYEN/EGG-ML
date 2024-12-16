
from speechbrain.utils.metric_stats import MetricStats
from collections import defaultdict
import torch

def acc_metric(predictions, targets, peakdet, oqs):
    assert(len(predictions) == len(targets) == len(peakdet) == len(oqs))
    
    # tensor([2, 2, 2, 2]) [[0], [0], [0], [0]]
    acc5 = [float(i in tgt) for i, tgt in zip(predictions, targets)]
    
    # List is important here: tensor(3) in [3] is True, tensor(3) in {3} is False
    targets3 = [ list(set([t for target in item_targets for t in CreakStats.cat_map[target]]))  for item_targets in targets ]
    acc3 = [float(i in tgt) for i, tgt in zip(predictions, targets3)]
    
    acc2 = []
    pred2 = predictions > 0
    targets2 = [ list(set([ t > 0 for t in item_targets])) for item_targets in targets ]
    acc2 = [float(i in tgt) for i, tgt in zip(pred2, targets2)]
    
    p2 = [float(i in tgt) for i, tgt in zip(pred2, targets2) if i]
    r2 = [float(i in tgt) for i, tgt in zip(pred2, targets2) if True in tgt]

    oq_preds = torch.tensor([0 if classe == 0 else peak[classe-1] for classe, peak in zip(predictions, peakdet)], device=oqs.device)

    # ~ print(oqs)
    # ~ print(oq_preds)
    # ~ exit()
    return (torch.tensor(acc5, dtype=torch.float), 
            torch.tensor(acc3, dtype=torch.float),
            torch.tensor(acc2, dtype=torch.float),
            torch.tensor(p2, dtype=torch.float),
            torch.tensor(r2, dtype=torch.float),
            eval_regressor(oq_preds / 100, oqs))

def eval_binary(preds, golds):
    preds = preds.squeeze().float()
    acc = (preds == golds).float()

    p2 = [float(p == g) for p, g in zip(preds, golds) if p == 1]
    r2 = [float(p == g) for p, g in zip(preds, golds) if g == 1]
    return acc, p2, r2

def eval_regressor(preds, golds):
    preds = preds.squeeze()
    return torch.abs(preds - golds)

def avg_round(l):
    return round(float(sum(l) / len(l)) * 100, 2)

class CreakStats(MetricStats):

    # Neutralizing distinction between classes 1 and 2 (dérivé, dérivé + smoothing
    # and classes 3 and  4: barycenter / barycenter + smoothing
    cat_map = {0: [0], 1:[1,2], 2:[1,2], 3:[3,4], 4:[3,4]}

    def __init__(self):
        super(CreakStats, self).__init__(acc_metric)

    def clear(self):
        """Creates empty container for storage, removing existing stats."""
        
        self.scores = defaultdict(list)
        # ~ # For the 5 parts classifiers
        # ~ self.scores_acc5 = []
        # ~ self.scores_acc3 = []
        # ~ self.scores_acc2 = []
        # ~ self.scores_p2 = []
        # ~ self.scores_r2 = []

        # ~ # For binary classifier
        # ~ self.scores_binary_acc = []
        # ~ self.scores_binary_p2 = []
        # ~ self.scores_binary_r2 = []
        # ~ # For regressor
        # ~ self.scores_regression = []

        # For all
        self.ids = []
        self.raw_preds = []  # predictiosn 5 way classifier
        self.raw_tgts = []
        self.logits = []
        
        self.summary = {}
        self.confusion = [defaultdict(int) for _ in range(5)]
        self.header_confusion = set()
        
        self.predictions = [0 for i in range(5)]

    def append(self, ids, *args, **kwargs):
        """Store a particular set of metric scores.

        Arguments
        ---------
        ids : list
            List of ids corresponding to utterances.
        *args, **kwargs
            Arguments to pass to the metric function.
        """
        
        self.ids.extend(ids)
        
        predictions = kwargs["predictions"]
        outputs = kwargs["outputs"]
        raw_labels = outputs["raw_labels"]
        
        # outputs: ['nary_labels', 'raw_labels', 'oqs', 'binary_labels']
        # predictions: ['y5', 'y2', 'y']

        assert(len(ids) == len(raw_labels))

        maxes, argmaxes = torch.max(predictions["y5"], dim=1)

        self.raw_preds.extend(list(argmaxes.cpu().numpy()))
        self.raw_tgts.extend(raw_labels)
        self.logits.extend(predictions["y5"])
        assert(len(self.ids) == len(self.raw_preds))
        assert(len(self.ids) == len(self.raw_tgts))

        # Batch evaluation
        #if self.batch_eval:
        scores5, scores3, scores2, p2, r2, reg = self.metric(argmaxes, raw_labels, outputs["peakdet"], outputs["oqs"])
        
        for pred, tgt in zip(argmaxes.cpu().numpy(), raw_labels):
            self.predictions[pred] += 1
            self.confusion[pred][tuple(tgt)] += 1
            self.header_confusion.add(tuple(tgt))

        self.scores["nary_acc5"].extend(scores5)
        self.scores["nary_acc3"].extend(scores3)
        self.scores["nary_acc2"].extend(scores2)
        self.scores["nary_p2"].extend(p2)
        self.scores["nary_r2"].extend(r2)
        self.scores["mae"].extend(reg)
        
        binary_predictions = (predictions["y2"] > 0).squeeze().float()
        scores_bin, pbin, rbin = eval_binary(binary_predictions, outputs["binary_labels"])
        
        self.scores["binary_acc"].extend(scores_bin)
        self.scores["binary_p2"].extend(pbin)
        self.scores["binary_r2"].extend(rbin)
        
        # For regressor
        self.scores["regression"].extend(eval_regressor(predictions["y"].squeeze() * binary_predictions, outputs["oqs"]))


    def summarize(self, field=None):
        """Summarize the metric scores, returning relevant stats.

        Arguments
        ---------
        field : str
            If provided, only returns selected statistic. If not,
            returns all computed statistics.

        Returns
        -------
        float or dict
            Returns a float if ``field`` is provided, otherwise
            returns a dictionary containing all computed stats.
        """
        #min_index = torch.argmin(torch.tensor(self.scores))
        #max_index = torch.argmax(torch.tensor(self.scores))
        N = sum(self.predictions)
        
        confusion, confusion_percents = self.confusion_to_table()
        p2 = avg_round(self.scores["nary_p2"])
        r2 = avg_round(self.scores["nary_r2"])
        f2 = round(2 * p2 * r2 / (p2 + r2), 2)

        p2_bin = avg_round(self.scores["binary_p2"])
        r2_bin = avg_round(self.scores["binary_r2"])
        f2_bin = round(2 * p2_bin * r2_bin / (p2_bin + r2_bin), 2)

        self.summary = {
            "nacc5": avg_round(self.scores["nary_acc5"]),
            "nacc3": avg_round(self.scores["nary_acc3"]),
            "nacc2": avg_round(self.scores["nary_acc2"]),
            "np2" : p2,
            "nr2" : r2,
            "nf2" : f2,
            "mae": avg_round(self.scores["mae"]),
            "pred": self.predictions,
            "pred_%": [round(i*100/N, 2) for i in self.predictions],
            "bacc": avg_round(self.scores["binary_acc"]),
            "bp2": p2_bin,
            "br2": r2_bin,
            "bf2": f2_bin,
            "reg": avg_round(self.scores["regression"]),
            #"min_score": float(self.scores[min_index]),
            #"min_id": self.ids[min_index],
            #"max_score": float(self.scores[max_index]),
            #"max_id": self.ids[max_index],
            "confusion": confusion,
            "confusion_%": confusion_percents,
        }

        if field is not None:
            return self.summary[field]
        else:
            return self.summary
    
    def export(self, filename):
        
        with open(filename, "w", encoding="utf8") as f:
            header = "\t".join("id gold pred score_0 score_1 score_2 score_3 score_4".split(" "))
            f.write(f"{header}\n")
            for ids, gold, pred, logits in zip(self.ids, self.raw_tgts, self.raw_preds, self.logits):
                gold = "/".join(map(str, gold))
                logits = "\t".join(map(lambda x: str(round(x.item(), 4)), logits))
                f.write(f"{ids}\t{gold}\t{pred}\t{logits}\n")

    def confusion_to_table(self):
        
        header = sorted(self.header_confusion)
        header_str = ["pred/gold"] + ["".join(map(str, l))  for l in header]
        
        sums = [0 for _ in header]
        for i in range(5):
            for j, k in enumerate(header):
                sums[j] += self.confusion[i][k]

        matrix = []
        matrix_norm = []
        for i in range(5):
            vals = [self.confusion[i][k] for k in header]
            matrix.append([i] + vals)
            matrix_norm.append([i] + [round(v / sums[j]*100, 2) for j, v in enumerate(vals)])
        
        return [header_str] + matrix, [header_str] + matrix_norm

    def write_stats(self, filestream, verbose=False):
        """Write all relevant statistics to file.

        Arguments
        ---------
        filestream : file-like object
            A stream for the stats to be written to.
        verbose : bool
            Whether to also print the stats to stdout.
        """
        if not self.summary:
            self.summarize()
    
        message = ""
        for k, v in sorted(self.summary.items()):
            message += f'{k}={v}\n'
        filestream.write(message)
        if verbose:
            print(message)

