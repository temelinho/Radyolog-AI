# Merlin Abdominal CT Dataset

Merlin Abdominal CT Dataset is an abdominal CT dataset consisting of 25,494 scans from 18,317 patients. Each scan is paired with its corresponding radiology report. The dataset includes abdominal and pelvis CT exams conducted between 2012 and 2018 at the Stanford Hospital Emergency Department, selected using CPT codes (72192, 72193, 72194, 74150, 74160, 74170, 74176, 74177, and 74178) through the STARR tool. For each exam, the DICOM series with the largest slice count was converted into NIfTI format, compressing the scans and removing patient-identifiable metadata. 

**GitHub:** [Merlin Code Repository](https://github.com/StanfordMIMI/Merlin)  
**Hugging Face:** [Merlin Hugging Face](https://huggingface.co/stanfordmimi/Merlin)

---

##  📂 Repository Structure

- **`merlin_data/`**  
  Contains 25,494 abdominal CT scans in NIfTI format.

- **`reports_final.xlsx`**  
  Excel file containing:
  - The **findings** section of the radiology report for each CT scan in `merlin_data/`
  - The **dataset split** (e.g., train, validation, test) for that scan
  - A flag indicating whether it was included in our **few-shot experiments**

- **`zero_shot_findings_disease_cls.csv`**  
  CSV file containing disease identifications from zero-shot evaluation.  
  Labels are based on positive/negative prompt matching from the report findings:  
    - missing: -1
    - negative: 0
    - positive: 1

- **`five_years_disease_task.csv`**  
CSV file containing **multi-disease 5-year prediction labels**. The column `merlin_split` indicates the subset of the test set used for the five-year disease prediction task. Only some test set samples are included, and this column reflects how those samples were further divided to create the splits for this specific task, as described in the Merlin paper.

- **`metadata.csv`**  
CSV file containing metadata for the Merlin dataset, including:
  - **Demographics**: Age, Gender, Race
  - **Acquisition parameters**: CT phase (phase), scanner manufacturer and model (manufacturer, manufacturermodelname), tube voltage (kvp), slice thickness (slicethickness), and tube current (ma; xraytubecurrent)

---

## 🔗 Citation

If you use the Merlin Dataset in your research, please cite:

```bibtex
@article{blankemeier2024merlin,
  title={Merlin: A vision language foundation model for 3d computed tomography},
  author={Blankemeier, Louis and Cohen, Joseph Paul and Kumar, Ashwin and Van Veen, Dave and Gardezi, Syed Jamal Safdar and Paschali, Magdalini and Chen, Zhihong and Delbrouck, Jean-Benoit and Reis, Eduardo and Truyts, Cesar and others},
  journal={Research Square},
  pages={rs--3},
  year={2024}
}
