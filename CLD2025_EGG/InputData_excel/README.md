# Description of Excel File Contents
The Excel files contain data processed in the study by semi-automatic [PeakDet](https://github.com/alexis-michaud/egg) tool. Each file consists of 12 columns, which are detailed below:

## Column Details
- **Column A:** (UID) Unique dentifier of each syllable of the corpus.
- **Column B:** (Start Time) Beginning time of the syllable (in seconds).
- **Column C:** (End Time): End time of the syllable (in seconds).
- **Column D:** (Glottal Cycle Start Time): Beginning time of each glottal cycle.
- **Column E:** (Cycle End Time): End time of each glottal cycle.
- **Column F:** f0 Values - Fundamental frequency values (in HZ).
- **Column G:** Oq values (in%) calculated by Peakdet (method: maxima on unsmoothed signal).
- **Column H:** Oq values (in%) calculated by Peakdet (method: maxima on smoothed signal).
- **Column I:** Oq values (in%) calculated by Peakdet (method: barycentre of peaks on unsmoothed signal).
- **Column J:** Oq values (in%) calculated by Peakdet (method: barycentre of peaks on smoothed signal).
- **Column K:** Checked Oq Values (in%).
  - Oq values retained after verifying opening peaks in the derivative of EGG signal (manually checked by the user).
  - A value of 0 indicates that the corresponding Oq value has been suppressed due to imprecise opening peaks.
- **Column L:** Creak Detection Results:
  - 0: No creak detected.
  - 1: Pressed voice or single-pulsed creak.
  - 2: Aperiodic creak.
  - 3: Double-pulsed creak.
## Note: 
Some explanations and illustrative figures detailing the glottal cycle —specifically the information in columns B (syllable start time), C (syllable end time), D (glottal cycle start time), and E (glottal cycle end time), as well as the measurements of f0 and Oq —are available in the accompanying document: [EGG_GlottalCycle_Display-Explanation.pdf](https://github.com/MinhChauNGUYEN/EGG-ML/blob/main/CLD2025_EGG/InputData_excel/GlottalCycle_Display-Explaination.pdf).
