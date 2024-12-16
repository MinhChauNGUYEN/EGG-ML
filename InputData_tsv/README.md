# .tsv Results - PeakDet Analysis

This folder contains 20 `.tsv` files, each representing the results of measurements obtained using the **PeakDet** tool for each speaker. These files serve as input data for the new project aimed at exploring the application of **Machine Learning** in estimating the **Open Quotient (Oq)** from **Electroglottographic (EGG) signals**.
# TSV Results - Peakdet Analysis

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

## EGG and acoustic data linked to this data: 

## Purpose

This dataset is designed to:
1. Train and evaluate Machine Learning models for estimating **Open Quotient (Oq)** from EGG signals.
2. Provide baseline data for comparison with future methodologies in signal processing and analysis.
