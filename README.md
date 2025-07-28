# PoolParty-4-TSC
Source code and experimental results for the paper "A Deep Dive into Alternatives to the Global Average Pooling for Time Series Classification" AALTD 2025.

A study on candidates to replace Global Average Pooling (GAP) in neural network architectures for time series classification.


## Usage

* Download and unpack the [2018 UCR archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/)
  [📎](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/UCRArchive_2018.zip)
* Download and unpack the [2018 UAE multivariate TSC](https://www.timeseriesclassification.com/)
  [📎](https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip)
  * The CharacterTrajectories dataset in the archive causes problems (metadata related),
    to solve them, download the
    [CharacterTrajectories](https://www.timeseriesclassification.com/description.php?Dataset=CharacterTrajectories)
    [📎](https://www.timeseriesclassification.com/aeon-toolkit/CharacterTrajectories.zip)
    dataset and replace with the original one.
* Change `data_folder` in [`data.py`](data.py) according to the previous steps.


### Requirements

* Python `3.11`
* TensorFlow `2.16.1`
  * Deep learning
* NumPy `1.26.4`
  * Data manipulation
* aeon `1.0`
  * Read `.ts` dataset files
* scikit-learn `1.5.2`
  * Preprocess labels
