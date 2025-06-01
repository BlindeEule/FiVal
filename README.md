# Purpose of this repository
This respository serves to compute sentiment scores for Finnish spontaneous speech transcriptions using Emily Öhman's SELF & FEIL  Emotion Lexicons for Finnish. The code was developed for a BSc. project at Tampere University.

# Installation and Use
1. Install the dependencies
2. Clone the [SELF-FEIL repository by Helsinki-NLP](https://github.com/Helsinki-NLP/SELF-FEIL) inside of this repository.
3. Configure the pipeline in config.json and add your own data
4. Run the pipeline with the command 
    python score_cal.py --config config.json

# Dependencies
  - python=3.9
  - numpy<2
  - pandas
  - pip
  - pip:
    - h5py
    - trankit
    - joblib

# Configuring the pipeline
- unique_name: Name of the directory (not the path) where results should be stored. Has to be a non-existing name in out_path. (Unique String)
- out_path: Path to the directory where the results should be stored.  (String)
- data_path: Path to the dataset for score calculation. Dataset either in csv or pkl format. (String)
- self_path: Path to the SELF lexicon. (String)
- feil_path: Path to the FEIL lexicon. (String)

- number_of_samples: Creates a random subset of number_of_samples length of the dataset. (Integer/null)
- text_column: Name of the column in the dataframe where the text is stored. (String)
- lemma_column: If lemmatization is already precomputed, set this parameter to the name of the respective column. (String/null)

- run_parallel: Parallel lemmatization. (true/false)
- batch_size: Batch size for parallel lemmatization. (Integer/null)
- fast_scores: Fast valence score calculation without majority vote, therefore less accurate. (true/false)

- write_to_file: Save computed scores to .pkl file. (true/false)
- verbose: Shows calculation time and progress of the pipeline. (true/false)

# Limitations
Note that the emotion intensity scores from the FEIL lexicon are not considered during score calculation. Furthermore, the pipeline has high computational cost due to the Trankit lemmatization pipeline. If possible, it is recommended to use parallel implementation on multiple CPU cores for large datasets.

# Citations
This repository is based on the work of other researchers. Please cite their works when using the code in your research.

Öhman, Emily S. "SELF & FEIL: Emotion Lexicons for Finnish." Digital Humanities in the Nordic and Baltic Countries Publications, vol. 4, no. 1, 2022, pp. 424–32, https://doi.org/10.5617/dhnbpub.11320.

Minh Van Nguyen, Lai, V. D., Amir Pouran Ben Veyseh, & Nguyen, T. H. (2021). Trankit: A Light-Weight Transformer-based Toolkit for Multilingual Natural Language Processing. https://doi.org/10.48550/arXiv.2101.03289.

