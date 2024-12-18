# Description of Excel File Contents
The Excel files contain data processed in the study by semi-automatic [PeakDet](https://github.com/alexis-michaud/egg) tool. Each file consists of 12 columns, which are detailed below:

## Column Details
- **Column A:** (UID) Unique dentifier of each syllable of the corpus.
- **Column B:** (Start Time) Beginning time of the syllable (in seconds).
- **Column C:** (End Time): End time of the syllable (in seconds).
- **Column D:** (Glottal Cycle Start Time): Beginning time of each glottal cycle.
- **Column E:** (Cycle End Time): End time of each glottal cycle.
- **Column F:** f0 Values: Fundamental frequency values (in HZ).
- **Column G:** Oq values calculated by Peakdet (method: maxima on unsmoothed signal).
- **Column H:** Oq values calculated by Peakdet (method: maxima on smoothed signal).
- **Column I:** Oq values calculated by Peakdet (method: barycentre of peaks on unsmoothed signal).
- **Column J:** Oq values calculated by Peakdet (method: barycentre of peaks on smoothed signal).
- **Column K:** Checked Oq Values.
  - Oq values retained after verifying opening peaks in the DEGG signal (manually checked by the user).
  - A value of 0 indicates that the corresponding Oq value has been suppressed due to imprecise opening peaks.
- **Column L:** Creak Detection Results:
  - 0: No creak detected.
  - 1: Pressed voice or single-pulsed creak.
  - 2: Aperiodic creak.
  - 3: Double-pulsed creak.
## Note: 
Detailed explanations and illustrative figures for these columns are provided in the accompanying file: EGG_GlottalCycle_Display-Explanation.docx.
