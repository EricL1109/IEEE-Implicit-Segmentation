# Automatic Phonetic Segmentation of the Yuhmu Language Using Mel Scale Spectral Parameters

**Authors:**  
- Eric Ramos-Aguilar  
- J. Arturo Olvera-López  
- Iván Olmos-Pineda  
- Ricardo Ramos-Aguilar  

**Affiliation:**  
- BUAP, UPIIT-IPN  
- BUAP  
- BUAP  
- UPIIT-IPN  

---
## Methodology for the Implicit Segmentation of the Yuhmu Language
![Image of method](graphicalabstract_.png)

Para la ejecución de los scripts en Python (*.py*), es necesario contar con las siguientes librerías instaladas en el entorno:

```bash
pip install pandas
pip install scipy
pip install statsmodels
pip install numpy
pip install pillow
pip install scikit-learn
pip install librosa
pip install matplotlib
```

A folder named **prueba** is included, which contains a set of audio files used for the analysis. Since this sample is not public, it is referenced within the main code `Overlapped Spectrum Image Generator.py`.

---

## Statistical Analysis: Friedman and ANOVA

To perform statistical analyses using the Friedman and ANOVA tests, the script `Friedman-ANOVA analysis.py` is used. This analysis requires the file `SER Results.xlsx`, which contains the complete results of all the analyzed audio combinations, including the values of **SER (Segment Error Rate)** and the expected number of phonemes for each evaluated word.

---

## Clustering Analysis: K-means

Finally, the K-means analysis is performed using the script `K-means analysis.py`. This procedure uses the spectrum images resulting from the audio samples to identify patterns and group the data. The results allow for observing the distribution and similarity of the phonetic segments in each case, an example of this can be seen in the following image, which is included and explained in the paper.

![Image of kmeans](k-means_results.png)

