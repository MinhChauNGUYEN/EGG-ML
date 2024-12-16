
import os
import json
import logging
import sys
import subprocess
from collections import defaultdict
import torchaudio
import torch
import random

RANDOM_SEED = 2878273

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from pandas import read_excel

from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.dataio.dataio import read_audio

from speechbrain.processing.speech_augmentation import Resample

##https://github.com/speechbrain/speechbrain/blob/develop/recipes/Fisher-Callhome-Spanish/fisher_callhome_prepare.py

logger = logging.getLogger(__name__)

def segment_audio(audio_path: str,
                    channel: int,
                    start: int,
                    end: int,
                    save_path: str,
                    from_sample_rate: int,
                    to_sample_rate: int):

    start = int(start * from_sample_rate)
    end = int(end  * from_sample_rate)
    num_frames = end - start

    data, _ = torchaudio.load(
        audio_path, frame_offset=start, num_frames=num_frames
    )

    resampler = Resample(orig_freq=from_sample_rate, new_freq=to_sample_rate)

    data = resampler(data)
    data = torch.unsqueeze(data[channel], 0)

    torchaudio.save(save_path, src=data, sample_rate=to_sample_rate)

def time2sec(string):
    h, m, s = string.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def parse_txt(filename):
    segments = []
    with open(filename) as f:
        for line in f:
            if not line or not line[0].isnumeric():
                continue
            sline = line.split()
            segments.append(
                {"id": int(sline[0]),
                "start": time2sec(sline[1]),
                "end": time2sec(sline[2]),
                "duration": time2sec(sline[3])})
    return sorted(segments, key=lambda x: x["start"])

def parse_annotations(annotations_file):
    annotations = read_excel(annotations_file, header=None, engine='openpyxl')
    lines = [l for l in annotations.values]
    #lines = [[int(i[0])] + list(i[1:]) for i in lines]
    data = defaultdict(list)
    for ID, *rest in lines:
        ID = int(ID)
        data[ID].append(rest)
    return data

BEG_SYL, END_SYL, BEG_GLOT, END_GLOT, F0, OQ_1, OQ_2, OQ_3, OQ_4, OQ_MC, CREAK = range(11)

def get_label(line):
    labels = []
    if line[OQ_MC] == 0:
        labels.append(0)
    for method in [OQ_1, OQ_2, OQ_3, OQ_4]:
        if line[OQ_MC] == line[method]:
            labels.append(method - OQ_1 + 1)
    return labels

if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    print(hparams["prepared_data_folder"])
    os.makedirs(hparams["prepared_data_folder"], exist_ok=True)

    files = get_all_files(hparams["data_folder"])
    txt_files = [f for f in files if f.endswith("REGIONS.txt")]


    json_dict = defaultdict(dict)
    for txt_file in txt_files:
        if txt_file.endswith("README.txt"):
            continue
        

        path_parts = txt_file.split(os.path.sep)
        file_id = path_parts[-1].rsplit("_", 1)[0]
        
        wav_main = f"{file_id}_annotated_AUD.wav"
        #wav_mono = f"{file_id}_annotated_mono.wav"
        #subprocess.run(["sox", wav_main, wav_mono, "remix 1"])
        
        wav_egg = f"{file_id}_annotated_EGG.wav"
        
        full_path_main = os.path.join(hparams["data_folder"], wav_main)
        full_path_egg = os.path.join(hparams["data_folder"], wav_egg)

        template_new = os.path.join(hparams["prepared_data_folder"], f"{file_id}_annotated_" + "{:03d}" + ".wav")
        template_new_egg = os.path.join(hparams["prepared_data_folder"], f"{file_id}_annotated_" + "{:03d}_EGG" + ".wav")

        print("wav_main", wav_main)
        id_speaker = wav_main.split("_")[2]
        annotations_file = os.path.join(hparams["data_folder"],f"KTM_DATA{id_speaker}_T4.xlsx")

        
        annotations = parse_annotations(annotations_file)

        segments = parse_txt(txt_file)

        start = 0
        last_segment = 0
        segment_id = 0

        for i in range(len(segments)-1):
            
            # get 
            if segments[i+1]["start"] - segments[i]["end"] > 1.5:
                end = segments[i]["end"] + (segments[i+1]["start"] - segments[i]["end"]) / 2

                # segment start to end, for segments: last segment to i
                new_file = template_new.format(segment_id)
                new_file_egg = template_new_egg.format(segment_id)

                segment_audio(full_path_main, 
                              channel=0,
                              start=start,
                              end=end,
                              save_path = new_file,
                              from_sample_rate=44100,
                              to_sample_rate=hparams["sample_rate"])

                segment_audio(full_path_egg, 
                              channel=0,
                              start=start,
                              end=end,
                              save_path = new_file_egg,
                              from_sample_rate=44100,
                              to_sample_rate=hparams["sample_rate"])


                new_segments = segments[last_segment:i+1]
                for s in new_segments:
                    # syllable start + end
                    s["start"] = s["start"] - start
                    s["end"] = s["end"] - start

                    closings = []
                    openings = []
                    syllable = []
                    
                    cycle_dict = defaultdict(list)
                    
                    if s["id"] in annotations:
                        #print("ID", s["id"])
                        data_lines = annotations[s["id"]]
                        #syllable = data_lines[0][BEG_SYL] - start,  data_lines[0][END_SYL] - start
                        #cycles = []
                        for line in data_lines:
                            cycle_dict["start"].append(line[BEG_GLOT] + s["start"])
                            cycle_dict["end"].append(line[END_GLOT]  + s["start"])
                            #assert(cycle_dict["end"][-1] <= s["end"])
                            cycle_dict["f0"].append(line[F0])
                            cycle_dict["OQ"].append(line[OQ_1:OQ_4+1])
                            cycle_dict["label"].append(get_label(line))
                            cycle_dict["creak"].append(int(line[CREAK]))
                            
                            if line[OQ_MC] != 0:
                                cycle_dict["opening_time"].append(line[OQ_MC] / 100 * (line[END_GLOT] - line[BEG_GLOT]) + line[BEG_GLOT] + start)
                            else:
                                cycle_dict["opening_time"].append(None)

                            #cycles.append(cycle_dict)

                            # ~ closings.append(line[BEG_GLOT])
                            # ~ if line[OQ_MC] != 0:
                                # ~ time_opening = line[OQ_MC] / 100 * (line[END_GLOT] - line[BEG_GLOT]) + line[BEG_GLOT]
                            # ~ else:
                                # ~ time_opening = None
                            # ~ openings.append(time_opening)
                        
                        # ~ closings = [shift + syllable[0] for shift in closings]
                        # ~ openings = [shift + syllable[0] if shift is not None else None for shift in openings]
                        
                        # ~ if not (syllable[0] == s["start"] and syllable[1] == s["end"]):
                            # ~ print(syllable, s["start"], s["end"])
                            
                    
                        # ~ if s["start"] != syllable[0]:
                            # ~ print(s["start"], syllable[0], s["start"] - syllable[0])
                            
                        
                        #s["syllable"] = syllable
                        s["cycles"] = cycle_dict
                        s["id"] = f"{id_speaker}_{s['id']}"
                    
                    # ~ s["closings"] = closings
                    # ~ s["openings"] = openings
            
                
                # DROP segments with no annotations
                new_segments = [d for d in new_segments if "cycles" in d]
                
                if len(new_segments) > 0:
                    json_dict[f"{file_id}_{segment_id}"] = {"file_path_wav": new_file,
                                                           "file_path_egg": new_file_egg,
                                                           "duration": end-start, 
                                                           "segments": new_segments}

                segment_id += 1
                start = end
                last_segment = i+1

    all_keys = sorted(json_dict)
    random.shuffle(all_keys)
    
    N = len(all_keys)
    split = int(N*15/100)
    
    
    test_keys, dev_keys, train_keys  = all_keys[:split], all_keys[split:2*split], all_keys[2*split:]
    
    dicts = {"train": {k: json_dict[k] for k in train_keys},
             "dev": {k: json_dict[k] for k in dev_keys},
             "test": {k: json_dict[k] for k in test_keys},
             "all": json_dict}
    
    json_files = hparams["dataset"]
    for k, v in dicts.items():
        with open(json_files[k], mode="w") as json_f:
            json.dump(v, json_f, indent=2)
    
