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

|Order| Speaker ID  | TSV File             | DOI on Pangloss                                       |                              
|-----|-------------|----------------------|-------------------------------------------------------|
|1| Speaker F3  | [crdo-MTQ_KTM_F3.tsv](./crdo-MTQ_KTM_F3.tsv) | [https://doi.org/10.24397/pangloss-0006760](https://doi.org/10.24397/pangloss-0006760) | 
|2| Speaker F7  | [crdo-MTQ_KTM_F7.tsv](./crdo-MTQ_KTM_F7.tsv) | [https://doi.org/10.24397/pangloss-0006769](https://doi.org/10.24397/pangloss-0006769) | 
|3| Speaker F9  | [crdo-MTQ_KTM_F9.tsv](./crdo-MTQ_KTM_F9.tsv) | [https://doi.org/10.24397/pangloss-0006777](https://doi.org/10.24397/pangloss-0006777) | 
|4| Speaker F10 | [crdo-MTQ_KTM_F10.tsv](./crdo-MTQ_KTM_F10.tsv) | [https://doi.org/10.24397/pangloss-0006783](https://doi.org/10.24397/pangloss-0006783) | 
|5| Speaker F12 | [crdo-MTQ_KTM_F12.tsv](./crdo-MTQ_KTM_F12.tsv) | [https://doi.org/10.24397/pangloss-0006789](https://doi.org/10.24397/pangloss-0006789) | 
|6| Speaker F13 | [crdo-MTQ_KTM_F13.tsv](./crdo-MTQ_KTM_F13.tsv) | [https://doi.org/10.24397/pangloss-0006791](https://doi.org/10.24397/pangloss-0006791) |
|7| Speaker F17 | [crdo-MTQ_KTM_F17.tsv](./crdo-MTQ_KTM_F17.tsv) | [https://doi.org/10.24397/pangloss-0006803](https://doi.org/10.24397/pangloss-0006803) | 
|8| Speaker F19 | [crdo-MTQ_KTM_F19.tsv](./crdo-MTQ_KTM_F19.tsv) | [https://doi.org/10.24397/pangloss-0006807](https://doi.org/10.24397/pangloss-0006807) | 
|9| Speaker F20 | [crdo-MTQ_KTM_F20.tsv](./crdo-MTQ_KTM_F20.tsv) | [https://doi.org/10.24397/pangloss-0006809](https://doi.org/10.24397/pangloss-0006809) | 
|10| Speaker F21| [crdo-MTQ_KTM_F21.tsv](./crdo-MTQ_KTM_F21.tsv) | [https://doi.org/10.24397/pangloss-0006811](https://doi.org/10.24397/pangloss-0006811) | 
|11| Speaker M1 | [crdo-MTQ_KTM_M1.tsv](./crdo-MTQ_KTM_M1.tsv) | [https://doi.org/10.24397/pangloss-0006764](https://doi.org/10.24397/pangloss-0006764) | 
|12| Speaker M5 | [crdo-MTQ_KTM_M5.tsv](./crdo-MTQ_KTM_M5.tsv) | [https://doi.org/10.24397/pangloss-0006771](https://doi.org/10.24397/pangloss-0006771) |
|13| Speaker M7 | [crdo-MTQ_KTM_M7.tsv](./crdo-MTQ_KTM_M7.tsv) | [https://doi.org/10.24397/pangloss-0006765](https://doi.org/10.24397/pangloss-0006765) | 
|14| Speaker M8 | [crdo-MTQ_KTM_M8.tsv](./crdo-MTQ_KTM_M8.tsv) | [https://doi.org/10.24397/pangloss-0006767](https://doi.org/10.24397/pangloss-0006767) | 
|15| Speaker M9 | [crdo-MTQ_KTM_M9.tsv](./crdo-MTQ_KTM_M9.tsv) | [https://doi.org/10.24397/pangloss-0006775](https://doi.org/10.24397/pangloss-0006775) | 
|16| Speaker M10| [crdo-MTQ_KTM_M10.tsv](./crdo-MTQ_KTM_M10.tsv) | [https://doi.org/10.24397/pangloss-0006779](https://doi.org/10.24397/pangloss-0006779) | 
|17| Speaker M11| [crdo-MTQ_KTM_M11.tsv](./crdo-MTQ_KTM_M11.tsv) | [https://doi.org/10.24397/pangloss-0006781](https://doi.org/10.24397/pangloss-0006781) | 
|18| Speaker M12| [crdo-MTQ_KTM_M12.tsv](./crdo-MTQ_KTM_M12.tsv) | [https://doi.org/10.24397/pangloss-0006787](https://doi.org/10.24397/pangloss-0006787) |
|19| Speaker M13| [crdo-MTQ_KTM_M13.tsv](./crdo-MTQ_KTM_M13.tsv) | [https://doi.org/10.24397/pangloss-0006793](https://doi.org/10.24397/pangloss-0006793) | 
|20| Speaker M14| [crdo-MTQ_KTM_M14.tsv](./crdo-MTQ_KTM_M14.tsv) | [https://doi.org/10.24397/pangloss-0006797](https://doi.org/10.24397/pangloss-0006797) |


## Purpose

This dataset is designed to:
1. Train and evaluate Machine Learning models for estimating **Open Quotient (Oq)** from EGG signals.
2. Provide baseline data for comparison with future methodologies in signal processing and analysis.
