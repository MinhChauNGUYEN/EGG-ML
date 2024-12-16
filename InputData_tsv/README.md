# .tsv Results - PeakDet Analysis

This folder contains 20 `.tsv` files, each representing the results of measurements obtained using the **PeakDet** tool for each speaker. These files serve as input data for the new project aimed at exploring the application of **Machine Learning** in estimating the **Open Quotient (Oq)** from **Electroglottographic (EGG) signals**.

## Data Description

Each `.tsv` file includes:
- **uid:** Unique dentifier of each syllable of the corpus.
- **syl_begin:** The beginning time of the syllable (in second).
- **syl_end:** The end time of the syllable (in second).
- **cyc_begin:** The beginning time of each glottal cycle.
- **cyc_end:** The end time of each glottal cycle.  
- **f0:** Fundamental frequency values (in Hz).
- **Oq_1:** Open quotient values, or Oq values (in %) are automatically calculated by PeaKDet, method 1: maxima on unsmoothed signal.
- **Oq_2:** Oq values (in %) are automatically calculated by PeaKDet, method 2: maxima on smoothed signal.
- **Oq_3:** Oq values (in %) are automatically calculated by PeaKDet, method 3: barycentre of peaks on unsmoothed signal.
- **Oq_4:** Oq (in %) are automatically calculated by PeaKDet, method 4: barycentre of peaks on smoothed signal.
- **Oq_gold:** The Oq values were retained after checking the opening peaks in the DEGG signal (by the user). The zeros mean that the Oq values at these cycles have been suppressed due to the imprecise opening peaks.
- **creak:** The resut Creak Detection: (0) means no creak, (1) means press voice or Single-pulsed creak, (2) means aperiodic creak, (3) double-pulsed creak.

## .tsv Files with Linked Data

This table provides links between each `.tsv` file and its corresponding **Electroglottographic (EGG)** file and **audio** file. The files are stored with unique DOIs for downloading on Pangloss Collection.

| Speaker ID | TSV File        | DOI on Pangloss                                   |                              
|------------|-----------------|-----------------------------------------------|
| Speaker F3  | [crdo-MTQ_KTM_F3.tsv](./crdo-MTQ_KTM_F3.tsv) | [DOI](https://doi.org/10.24397/pangloss-0006760) | 
| Speaker02  | [crdo-MTQ_KTM_F7.tsv](./crdo-MTQ_KTM_F7.tsv) | [DOI](https://doi.org/example_egg2) | 
| Speaker03  | [crdo-MTQ_KTM_F9.tsv](./crdo-MTQ_KTM_F9.tsv) | [DOI](https://doi.org/example_egg3) | 



## Purpose

This dataset is designed to:
1. Train and evaluate Machine Learning models for estimating **Open Quotient (Oq)** from EGG signals.
2. Provide baseline data for comparison with future methodologies in signal processing and analysis.
