# GenoDrawing: An autoencoder framework for image prediction from SNP markers

Advancements in genome sequencing have facilitated whole genome characterization of numerous
plant species, providing an abundance of genotypic data for genomic analysis. Genomic selection
and neural networks, particularly deep learning, have been developed to predict complex traits from
dense genotypic data. Autoencoders, a neural network model to extract features from images in
an unsupervised manner, has proven to be useful for plant phenotyping. This study introduces an
autoencoder framework, GenoDrawing, for predicting and retrieving apple images from a low-depth
single nucleotide polymorphism (SNP) array, potentially useful in predicting traits that are difficult
to define. GenoDrawing demonstrated proficiency in its task while using a small dataset of
shape-related SNPs, and multiple experiments were conducted to evaluate the impact of SNP selection
and shape relation. Results indicated that the correct relationship of SNPs with visual traits
had a significant impact on the generated images, consistent with biological interpretation. While
using significant SNPs is crucial, incorporating additional, unrelated SNPs results in performance
degradation for simple NN architectures that cannot easily identify the most important inputs. The
proposed GenoDrawing method is a practical framework for exploring genomic prediction in fruit
tree phenotyping, particularly beneficial for small to medium breeding companies to predict economically
significant heritable traits. Although GenoDrawing has limitations, it sets the groundwork
for future research in image prediction from genomic markers. 

![GenoDrawing_example](Figures/GenoDrawing_examples.png)


