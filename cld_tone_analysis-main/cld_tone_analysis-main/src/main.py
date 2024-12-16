import os
import json
import logging
import sys
import subprocess
from collections import defaultdict
from tabulate import tabulate
import torchaudio
from torchaudio.functional import compute_deltas
import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from pandas import read_excel

from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.lobes.features import MFCC, Fbank
from speechbrain.nnet.losses import nll_loss
from speechbrain.utils.metric_stats import MetricStats

from metrics import CreakStats

# ~ from speechbrain.utils.data_utils import get_all_files, download_file
# ~ from speechbrain.dataio.dataio import read_audio
# ~ from speechbrain.processing.speech_augmentation import Resample

##https://github.com/speechbrain/speechbrain/blob/develop/recipes/Fisher-Callhome-Spanish/fisher_callhome_prepare.py
#`creating a DynamiItemDataset instance from JSON or CSV annotation is immediate



@sb.utils.data_pipeline.takes("file_path_egg")
@sb.utils.data_pipeline.provides("signal")
def audio_pipeline(file_path_egg):
    sig = sb.dataio.dataio.read_audio(file_path_egg)
    return sig

@sb.utils.data_pipeline.takes("file_path_wav")
@sb.utils.data_pipeline.provides("signal")
def audio_pipeline_wav(file_path_wav):
    sig = sb.dataio.dataio.read_audio(file_path_wav)
    return sig

def scale_f0(f):
    # log did not work
    return f / 100


class CreakBrain(sb.Brain):

    def compute_forward(self, batch, stage):

        if self.hparams.use_egg:
            batch = batch.to(self.device)
            x = self.modules.features(batch.signal.data)

            #segments = batch["segments"]
            examples = []

            #print(len(segments), [len(s) for s in segments])
            #loss = 0

            for i, seg_batch in enumerate(batch["segments"]):
                for segment in seg_batch:
                    examples.append(self.extract_frames(i, segment, x))
            
            lengths = [ex["length"] for ex in examples]
            batched_signal = [ex["signal"] for ex in examples]
            batched_signal = pad_sequence(batched_signal, batch_first=True)
            batched_signal = self.modules.encoder(batched_signal)
            packed_batched_signal = pack_padded_sequence(batched_signal, lengths, batch_first=True, enforce_sorted=False)

            x, (_, _) = self.modules.seq(packed_batched_signal)
            x, _ = pad_packed_sequence(x, batch_first=True)
            
            x = self.collect_input(x, examples)
            x = self.modules.FFN(x)
            
            y_annot = self.modules.clf_annot(x)
            y_regression = self.modules.regression_oq(x)
            y_binary = self.modules.clf_binary(x)

            #y_creak = self.modules.clf_creak(x)
            #predictions.append((y_annot, y_creak))
            #return y_annot
            return {"y5": y_annot, "y2": y_binary, "y": y_regression}

            # ~ predictions = []
            # ~ for ex in examples:
                # ~ x = self.modules.encoder(ex["signal"])
                # ~ x, (_,_) = self.modules.seq(x)
                # ~ x = self.collect_input(x, ex)
                
                # ~ y_annot = self.modules.clf_annot(x)
                # ~ #y_creak = self.modules.clf_creak(x)
                # ~ #predictions.append((y_annot, y_creak))
                # ~ predictions.append(y_annot)

        else:
            #  TODO: update that
            x = self.fake_features
            examples = []
            for i, seg_batch in enumerate(batch["segments"]):
                for segment in seg_batch:
                    examples.append(self.extract_frames(i, segment, x))
            
            predictions = []
            x = torch.stack([torch.tensor(oq, device=self.device) for ex in examples for oq in ex["oq"]])
            if self.hparams.use_F0:
                F0 = scale_f0(torch.cat([torch.tensor(f0, device=self.device) for ex in examples for f0 in ex["f0"]]).reshape(-1, 1))
                x = torch.cat([x, F0], dim=1)
            #y_annot = self.modules.clf_annot(x)
            y_annot = self.modules.clf_annot(x)
            y_regression = self.modules.regression_oq(x)
            y_binary = self.modules.clf_binary(x)

            return {"y5": y_annot, "y2": y_binary, "y": y_regression}


    def compute_objectives(self, predictions, batch, stage):
        outputs = self.collect_output(batch)
        
        loss = F.binary_cross_entropy_with_logits(predictions["y5"], outputs["nary_labels"])
        
        loss_reg = F.mse_loss(predictions["y"].squeeze(), outputs["oqs"])
        
        loss_binary = F.binary_cross_entropy_with_logits(predictions["y2"].squeeze(), outputs["binary_labels"])
        

        ex_id = []
        for seg_batch in batch["segments"]:
            for segment in seg_batch:
                ex_id.extend([segment["id"] for _ in segment["cycles"]["label"]])
        
        self.accuracy_metric.append(ex_id, outputs=outputs, predictions=predictions)
        # ~ for  y_annot, tgt_annot in zip(predictions, targets):
            # ~ #tgt_annot = torch.tensor([i[0] for i in tgt_annot], dtype=torch.long)
            # ~ #tgt_creak = torch.tensor(tgt_creak)
            # ~ #loss += F.cross_entropy(y_annot, tgt_annot)
            # ~ loss += F.binary_cross_entropy_with_logits(y_annot, tgt_annot)
            # ~ #loss += F.cross_entropy(y_creak, tgt_creak)
        #if stage != sb.Stage.TRAIN:
        # ~ for  i, (y_annot, tgt_annot) in enumerate(zip(predictions, raw_targets)):
            # ~ maxes, argmaxes = torch.max(y_annot, dim=1)
            # ~ ex_id = batch["segments"][0][i]["id"]
            # ~ self.accuracy_metric.append([ex_id for _ in range(len(argmaxes))], argmaxes, tgt_annot, logits = y_annot)

        if self.hparams.loss == "both":
            return (loss + loss_reg + loss_binary) / 3
        if self.hparams.loss == "regression":
            return (loss_reg + loss_binary) / 2
        if self.hparams.loss == "classification":
            return loss
        assert False, "Unknown loss type"


    def collect_input(self, x, examples):
        xs = []
        for id_ex, ex in enumerate(examples):
            for i, j, oq, F0 in zip(ex["starts"], ex["ends"], ex["oq"], ex["f0"]):
                l = [x[id_ex,i,:], x[id_ex,j,:]]
                if self.hparams.use_OQ:
                    l.append(torch.tensor(oq, device=self.device))
                if self.hparams.use_F0:
                    l.append(scale_f0(torch.tensor(F0, device=self.device)))
                xs.append(torch.cat(l, dim=0))

        return torch.stack(xs)
    
    def collect_output(self, batch):
        labels = []
        raw_labels = []
        oqs = []
        binary_labels = []
        peakdets = []
        for seg_batch in batch["segments"]:
            for segment in seg_batch:
                #labels.append((segment["cycles"]["label"], segment["cycles"]["creak"]))
                #labels.append(segment["cycles"]["label"])
                #print("jjj", segment["cycles"]["label"])
                l = segment["cycles"]["label"]
                
                oq = []
                for item_oq, item_labels in zip(segment["cycles"]["OQ"], segment["cycles"]["label"]):
                    if item_labels[0] == 0:
                        oq.append(0)
                    else:
                        oq.append(item_oq[item_labels[0]-1])
                
                oqs.extend(oq)
                
                # TODO: continue here: this is wrong
                raw_labels.extend(l)
                binary_labels.extend([float(0 not in item) for item in l])
                l = [torch.tensor(item, dtype=torch.long, device=self.device) for item in l]
                l = [torch.nn.functional.one_hot(item, num_classes=5).float().sum(dim=0) for item in l]
                #l = torch.stack(l)
                labels.extend(l)
                peakdets.extend(segment["cycles"]["OQ"])
                

        # ~ for seg_batch in batch["segments"]:
            # ~ for segment in seg_batch:
                # ~ #labels.append((segment["cycles"]["label"], segment["cycles"]["creak"]))
                # ~ #labels.append(segment["cycles"]["label"])
                # ~ #print("jjj", segment["cycles"]["label"])
                # ~ l = segment["cycles"]["label"]
                
                # ~ # TODO: continue here: this is wrong
                # ~ raw_labels.append(l)
                # ~ l = [torch.tensor(item, dtype=torch.long, device=self.device) for item in l]
                # ~ l = [torch.nn.functional.one_hot(item, num_classes=5).float().sum(dim=0) for item in l]
                # ~ l = torch.stack(l)
                # ~ labels.append(l)

        labels = torch.stack(labels)
        oqs = torch.tensor(oqs, dtype=torch.float, device=self.device) / 100
        binary_labels = torch.tensor(binary_labels, dtype=torch.float, device=self.device)
        
        return {"nary_labels": labels,  
                "raw_labels": raw_labels, 
                "oqs": oqs, 
                "binary_labels": binary_labels,
                "peakdet": peakdets}

    def extract_frames(self, batch_i, segment, x):
        """
        segment: dict id, start, end, duration, cycles
        x: MFCC features
        
        returns:
            dict:
                signal: matrix
                starts: cycle starts
                ends: cycle ends
        """
        ## segment = dictionary
        # ~ print()
        # ~ print()
        #print(segment["id"], segment["duration"])

        context = int(self.hparams.lstm_context * self.hparams.fps)
        
        start_syllable_frame = int(segment["start"] * self.hparams.fps)
        end_syllable_frame = int(segment["end"] * self.hparams.fps)
        
        assert(start_syllable_frame - context > 0)
        segment_s = start_syllable_frame - context
        segment_e = end_syllable_frame + context

        start_cycle_frames = [int(t * self.hparams.fps) - segment_s for t in segment["cycles"]["start"]]
        end_cycle_frames = [int(t * self.hparams.fps) - segment_s for t in segment["cycles"]["end"]]
        open_quotient = [list(map(lambda x: x/100, oq)) for oq in segment["cycles"]["OQ"]]
        F0 =          [[f0] for f0 in segment["cycles"]["f0"]]
        
        signal_segment = x[batch_i, segment_s:segment_e, :]
        
        #print("start", start_cycle_frames, end_cycle_frames, signal_segment.shape)
        # ~ print()
        # ~ print("seg, shape", x.shape, segment["cycles"]["start"], )
        # ~ print("syllable", segment["start"], segment["end"], start_syllable_frame, end_syllable_frame)
        # ~ print("sig, shape", signal_segment.shape, end_cycle_frames)
        # ~ print()
        # ~ print()
        # ~ print(list(segment["cycles"].keys()))
        # print(signal_segment.shape)
        # ~ print(start_cycle_frames)
        # ~ print(end_cycle_frames)
        return {"signal": signal_segment, 
                "starts": start_cycle_frames,
                "ends": end_cycle_frames,
                "oq": open_quotient,
                "f0": F0,
                "length": segment_e - segment_s}
                # ~ "labels": segment["cycles"]["label"],
                # ~ "creak": segment["cycles"]["creak"]}


    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        #self.accuracy_metric = sb.utils.metric_stats.MetricStats(acc_metric)
        self.accuracy_metric = CreakStats()

        if not self.hparams.use_egg:
            self.fake_features = torch.zeros(hparams["batch_size_eval"], 100000, self.hparams.mfcc_dim)

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the beginning of each epoch"""

        stage_stats = {"loss": round(stage_loss, 5)}
        stage_stats.update(self.accuracy_metric.summarize())
        log_stats = {k:stage_stats[k] for k in ["loss", "nacc3", "nacc5", "nacc2", "np2", "nr2", "nf2", "mae", "bacc", "bf2", "reg", "pred_%"]}
        #print(f"Epoch {epoch}", stage, log_stats, flush=True)
        #log_stats["pred"] = stage_stats["pred"]

        if stage == sb.Stage.VALID:
            if epoch == 1:
                self.best_epoch = 1
                self.best_acc = log_stats["nacc3"]
            else:
                if log_stats["nacc3"] >= self.best_acc:
                    self.best_epoch = epoch
                    self.best_acc = log_stats["nacc3"]
            log_stats["best"] = f"i{self.best_epoch}/{self.best_acc}"

        logging.info(f"Epoch {epoch} {stage} {log_stats}")
        
        confusion_table = tabulate(stage_stats["confusion"])
        confusion_table_pc = tabulate(stage_stats["confusion_%"])
        logging.info("\n" + confusion_table)
        logging.info("\n" + confusion_table_pc)

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        if stage == sb.Stage.VALID:
            # ~ epoch_stats = {
                # ~ "epoch": epoch,
            # ~ }

            # ~ self.hparams.train_logger.log_stats(
                # ~ stats_meta=epoch_stats,
                # ~ train_stats=self.train_stats,
                # ~ valid_stats=stage_stats,
            # ~ )
            
            self.checkpointer.save_and_keep_only(
                meta={"nacc3": stage_stats["nacc3"], "epoch": epoch},
                max_keys=["nacc3"],
                num_to_keep=2,
                keep_recent=True
            )
            self.accuracy_metric.export(f'{self.hparams.output_folder}/predictions_{epoch}.tsv')





def load_dataset(section, hparams):
    if hparams["input"] == "egg":
        dynamic_item = [audio_pipeline]
    else:
        assert(hparams["input"] == "wav")
        dynamic_item = [audio_pipeline_wav]
    return DynamicItemDataset.from_json(
                    hparams["dataset"][section],
                    dynamic_items=dynamic_item,
                    output_keys=["signal", "file_path_egg", "duration", "segments"])

def compute_stats(dataset):
    stats = defaultdict(int)
    stats_c = defaultdict(int)
    for item in dataset:
        for seg in item["segments"]:
            for label in seg["cycles"]["label"]:
                stats["all"] += 1
                for l in label:
                    stats[l] += 1
                label_str = "/".join(map(str, label))
                stats_c[f"{label_str}"] += 1
    for i in range(5):
        stats[f"%{i}"] = round(stats[i] / stats["all"] * 100, 2)
    for k in list(stats_c):
        stats_c[f"%{k}"] = round(stats_c[k] / stats["all"] * 100, 2)
    return {"uni":dict(stats), "complex":dict(stats_c)}

if __name__ == "__main__":
    import pprint
    import random
    import numpy
    
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    random.seed(hparams["seed"])
    torch.manual_seed(hparams["seed"])
    numpy.random.seed(hparams["seed"])

    pprint.pprint(hparams)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )



    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s_%(name)s_%(levelname)s %(message)s',
                        datefmt='%m-%d_%H:%M',
                        filename=f'{hparams["output_folder"]}/log',
                        filemode='w')

    train = load_dataset("train", hparams)
    dev = load_dataset("dev", hparams)
    test = load_dataset("test", hparams)
    
    print("train stats")
    pprint.pprint(compute_stats(train))
    print("dev   stats")
    pprint.pprint(compute_stats(dev))
    print("test  stats")
    pprint.pprint(compute_stats(test))
    
    #speechbrain.lobes.features.MFCC(deltas=True, context=True, requires_grad=False, sample_rate=16000, 
    #f_min=0, f_max=None, n_fft=400, n_mels=23, n_mfcc=20, filter_shape='triangular', 
    #param_change_factor=1.0, param_rand_factor=0.0, left_frames=5, right_frames=5, win_length=25, hop_length=10)


    hf = hparams["features"]
    hparams["fps"] = 1000 / hf["hop_length"] # frame per seconds

    output_size = hf["n_mfcc"] * 3 * (hf["left_frames"] + hf["right_frames"] + 1)
    hparams["mfcc_dim"] = output_size

    clf_input_size = 0
    if hparams["use_egg"]:
        clf_input_size += hparams["dimensions"]["h1"] * 4
    if hparams["use_OQ"]:
        clf_input_size += 4
    if hparams["use_F0"]:
        clf_input_size += 1

    assert(hparams["use_OQ"] or hparams["use_egg"])

    modules = {"features": MFCC(**hf),
                                # ~ sample_rate=hparams["sample_rate"],
                                # ~ win_length=hparams["win_length"],
                                # ~ hop_length=hparams["hop_length"],
                                # ~ left_frames=hparams["left_frames"],
                                # ~ right_frames=hparams["right_frames"]
                                # ~ ),
               "encoder": torch.nn.Sequential(torch.nn.LayerNorm(output_size),
                                              torch.nn.Linear(output_size, hparams["dimensions"]["h1"]),
                                              torch.nn.Tanh()),
               "seq": torch.nn.LSTM(input_size=hparams["dimensions"]["h1"],
                                    hidden_size=hparams["dimensions"]["h1"],
                                    num_layers=1,
                                    batch_first=True,
                                    bidirectional=True),
               "FFN" : torch.nn.Sequential(torch.nn.LayerNorm(clf_input_size),
                                                 torch.nn.Linear(clf_input_size, hparams["dimensions"]["h1"]),
                                                 torch.nn.ReLU()),
               "clf_annot" : torch.nn.Linear(hparams["dimensions"]["h1"], 5),
               "regression_oq": torch.nn.Sequential(torch.nn.Linear(hparams["dimensions"]["h1"], 1), torch.nn.Sigmoid()),
               "clf_binary": torch.nn.Linear(hparams["dimensions"]["h1"], 1),

               #"clf_creak": torch.nn.Linear(hparams["dimensions"]["h1"]*4, 4)
            }
              # ~ "encoder": torch.nn.Sequential(torch.nn.Linear(40, 256),
                                           # ~ torch.nn.ReLU()),
              # ~ "pooling": sb.nnet.pooling.StatisticsPooling(),
              # ~ "to_output": torch.nn.Linear(512, 10),
              # ~ "softmax": sb.nnet.activations.Softmax(apply_log=True)}

    #normalize = sb.processing.features.InputNormalization(norm_type="global", update_until_epoch=4)
    #noam_annealing = sb.nnet.schedulers.NoamScheduler(lr_initial=hparams["lr"], n_warmup_steps=200)

    checkpointer = sb.utils.checkpoints.Checkpointer(checkpoints_dir=hparams["save_folder"],
                                                     recoverables={"counter": hparams["epoch_counter"],
                                                                    "optimizer": hparams["Adam"]})


        # ~ checkpoints_dir: !ref <save_folder>
        # ~ recoverables:
            # ~ model: !new CreakBrain
            # ~ noam_scheduler: !ref <noam_annealing>
            # ~ normalizer: !ref <normalize>
            # ~ counter: !ref <epoch_counter>

    brain = CreakBrain(modules, 
                       opt_class=hparams["Adam"],
                       hparams=hparams,
                       run_opts=run_opts,
                       checkpointer=checkpointer)


    brain.fit(brain.hparams.epoch_counter,
              train_set=train, 
              valid_set=dev,
              train_loader_kwargs={"batch_size": hparams["batch_size_train"], "drop_last":False},
              valid_loader_kwargs={"batch_size": hparams["batch_size_eval"], "drop_last":False},)

    brain.evaluate(test_set=test,
              test_loader_kwargs={"batch_size": hparams["batch_size_eval"], "drop_last":False},)
    
