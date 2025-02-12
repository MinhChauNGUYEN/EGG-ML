# CLD2025_EGG

The CLD2025_EGG project marks the first exploration of the initiative titled *"Exploring Machine Learning perspectives for electroglottographic signals"*, conducted in 2022. The primary objective of this study was to automate the extraction of the glottal open quotient from electroglottographic recordings.  

You can access the full report of this investigation on [HAL](https://hal.science/hal-04081199).

 ## Corpus
The corpus used comes from a dialect of Muong (Vietnam). This language has a phonetically complex tonal system, combining f0 contours and phonation features. In particular, in the system of 5 + 2 tones, one of the tones includes a transition into creaky voice [(Nguyen,2021)](https://theses.hal.science/tel-03652510). The document [NGUYEN2019_Muong_PhDCorpus.pdf](https://github.com/MinhChauNGUYEN/EGG-ML/blob/main/CLD2025_EGG/NGUYEN2019_Muong_PhDCorpus.pdf) describe all the relevant content of the total corpus.
 
The electroglottographic (EGG) signal was recorded simultaneously with the acoustic signal from 20 speakers (10 men, 10 women). The audio and EGG corpus is openly accessible in the [collection Pangloss](https://pangloss.cnrs.fr/corpus/M%C6%B0%E1%BB%9Dng?lang=en) under a Creative Commons license (CC BY-NC-SA 3.0 fr). 

The measurement of two phonetic parameters—fundamental frequency (f0) and glottal open quotient (Oq)—was performed using peak detection on the derivative of the electroglottographic signal (dEGG). This analysis was conducted with the semi-automatic [PeakDet](https://github.com/alexis-michaud/egg) script, running in MATLAB. 

For those familiar with Praat for acoustic analysis, [PraatDet](https://github.com/kirbyj/praatdet)—a Praat-based tool for EGG analysis inspired by PeakDet—was developed by James Kirby.
 
 ## Experiments
We implemented a bidirectional Long Short-Term Memory (LSTM) neural network to predict the glottal open quotient (Oq) for each glottal cycle from the electroglottographic (EGG) signal. The model was trained using data manually pre-processed with Peakdet. The scripts were created and shared with the authorization of [Maximin Coavoux](https://mcoavoux.github.io/). They are accessible in the folder [cld_tone_analysis-main](https://github.com/MinhChauNGUYEN/EGG-ML/tree/main/CLD2025_EGG/cld_tone_analysis-main)

 ## Acknowledgments
The work presented here is funded by the French-German project “Computational Language Documentation by 2025 / La documentation automatique des langues à l’horizon 2025” (CLD 2025, ANR-19CE38-0015-04), conducted by an interdisciplinary team associating linguists and computer scientists (from the field of Natural Language Processing). The work was carried out at LIG (Laboratoire d’Informatique de Grenoble, UMR 5217), under the joint supervision of a linguist, Solange Rossato, and a computer scientist, Maximin Coavoux (in collaboration with Alexis Michaud, from LACITO, UMR 7107).
