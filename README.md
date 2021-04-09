# Local-Time-Series-Interpretability-over-MHA

## Description

The given code can provide a local abstraction of time series data based on MHA. To archive this, we train a transformer model on a time series classification problem which was symbolized by the SAX algorithm. Over two given thresholds the MHA attention matrix is used to abstract the data. The code further includes a visualization for local interpretability over the abstracted data. With a human in the loop process, a human can use the local visualization to improve the thresholds for each dataset/classification problem. Additionally, a re-evaluation model is trained to show how well the reduced data performs and by how much the data is reduced. We argue that the visualized abstractions are better interpretable than the normal input data, which is helpful to understand the underlying classification problem.

The project contains a Jupiter notebook which provides the model from the publication and also includes the weights for the published results (in the zip "saves_paper"). The code uses by default two datasets (linked below) and trains 5 folds for cross-validation. Each fold trains 6 models:

- Normal input data
- Symbolic data (SAX)
- Average based threshold with interpolated missing data
- Average based threshold with mising data masked
- Max based theshold with interpolated missing data
- Max based threshold with mising data masked

At the end of the notebook the abstracted data can be analysed with the given visualisations.

## Dependencies
A list of all needed dependencies (other versions can work but are not guaranteed to do so):

tensorflow==2.2.0<br>
seaborn==0.10.1<br>
scipy==1.4.1<br>
scikit-learn==0.23.2<br>
pyts==0.11.0<br>
pandas==1.0.0<br>
numpy==1.18.5<br>
matplotlib==3.3.1<br>
tensorflow_addons==0.11.2<br>
tensorflow_probability==0.7.0<br>


## Cite and publications
This code represents the used model for the following publication: TODO

Preprint: <br>
https://martin.atzmueller.net/paper/VisualizingAbstractedTransformerAttentionLocalInterpretability-SchwenkeAtzmueller-2021-preprint.pdf

If you use, build upon this work or if it helped in any other way, please cite the linked publication.


## Datasets

Included datasets are:

http://www.timeseriesclassification.com/description.php?Dataset=SyntheticControl <br>
http://www.timeseriesclassification.com/description.php?Dataset=ECG5000
