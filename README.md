# flair-tiger

This repository contains the code to finetune pre-trained contextual word embeddings to predict POS tags (part of speech) of the German TIGER corpus using [Flair Framework](https://github.com/flairNLP/flair). The experiments are performed on [TIGER corpus version 2.2](https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/tiger-corpus/download/start.html). To train the embeddings, we use Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aflueckiger/flair-tiger/blob/main/flair-tiger-pos-tagging.ipynb)

## Introduction

POS-tagging is sometimes considered as a solved problem, however, it is tested in highly idealised conditions (Giesbrecht 2009). In general, the performance drops significantly when the data is noisy as in the context of the web. Hence, neural character-based models are promising to use them in such conditions and may boost the performance since there are no Out-Of-Vocabulary words (OOV) (Horsmann et Zesch 2017). The results achieved recently with contextual string embedding and a standard BiLSTM-CRF architecture for POS tagging problem in English are SOTA by now (Akbik et al. 2018).

Yet, neural character-based models are much more costly to train compared to Statistical taggers (e.g. TreeTagger) which work very well for proper text. Our ablation experiments show for the fine-tuning only little data is needed to get high accuracy. This means the model generalizes well with only small data provided which is important to reduce the computational resources to train a model. Moreover, the model works without any feature engineering. Due to the good generalization and the end-to-end framework, this approach is also attractive for low resource languages for which no big corpora exist.

LSTMs is considered to require large amounts of training data with a minimum of 50k tokens to  get good results (Horsmann et Zesch 2017). Therefore, our objective is to train the model with contextual word embedding on different amount of data of German to describe the effect of the dataset size on the tagging accuracy applying the new type of embedding. Another goal is to estimate the model performance on the data rich with OOV-words to learn how well a tagger generalizes to unseen words.

## Materials

As the training dataset, the [TIGER Corpus]([http://www.ims.uni-stuttgart.de/forschung/ressourcen/ korpora/TIGERCorpus/download/start.html](http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/download/start.html)) from the University of Stuttgart was chosen. The Corpus (versions 2.1 and 2.2) consists of 719.826 tokens (app. 41.000 sentences) of German newspaper text, taken from the Frankfurter Rundschau, and is the largest available annotated corpus of German. It is semi-automatically annotated with POS information, using the standard STTS tagset (54 tags).

All the models were firstly evaluated on the test subset from the TIGER corpus. Then, to learn how well the models can deal with different types of texts, we estimated some of the models on the Computer-Mediated Communication (CMC) and Web datasets. The CMC dataset consists of the data samples from different CMC genres: e.g. social and professional chat, tweets, Wikipedia talk pages, blog comments, Whatsapp conversations. The web corpora subset was sampled from different text genres on the Web. Both datasets were used for the EmpiriST 2015 shared task[1](#sdfootnote1sym), and  have been annotated using the “STTS IBK” tagset, which is based on the standard tagset for German (Beißwenger et al. 2015; Thater 2018). In our case, datasets can be used as the data that are rich with OOV-words (e.g. also including such words as *Heyyyyyyy, tadaaaa, remarcmarc, tschööö*, etc.) and non-standard syntactic structure. The tags that were not present in the training data were filtered out: e.g. EMOIMG (“Emoticon, als Grafik-Ikon dargestellt”), EML (“E-Mail-Adresse”), AKW (“Aktionswort: *lach* freu, grübel *lol*”), etc.

## Method

We trained the models on the samples of different sizes: starting from 35.6k and 71.5k tokens (5% and 10% of the corpus), then increasing each next training set by 10% of the corpus until 50%, and finally training with 70% and 100% of all the data (see table 1).

To train the models, the sequence tagger model of the Flair framework was used. Forward and backward embeddings for German for each of the words were built with FlairEmbeddings module that operates on the character level (Akbik et al. 2018).

The architecture of the model is a standard LSTM-CRF consisting of a bidirectional LSTM on top of each embedding layer and a CRF in a subsequent decoding layer. When there is only little data, say less 250k of tokens, one can keep the embeddings for the sequences in memory, which considerably speeds up the process (appr. 3 vs. 15 min per epoch). Due to restricted resources, the model was trained for 20 epochs at maximum using the hyperparameters proposed by Akbik et al. 2018.

## Results

The results are quite promising and outperform the current SOTA. When evaluated on the test set from the TIGER data, accuracy (micro average) is higher than 98% for all of the models, except the one trained on the smallest dataset: yet, its score is still very close: 97.8%. For the models with more data (>300k), the results are getting closer to 99%, which is the level of human annotators. 

**Table 1: Final scores for the models with different amount of training data**

| Model | data split | **n of tokens** | **n of docs** | Accuracy (=micro F1 avg.) | Accuracy / Macro F1 avg.                 |
| --- | --- | --- | --- | --- | --- |
| Ours (LSTM + Flair)      | 0.05                                                         | 35'621          | 2045          | 0.9781                    | 0.8563     |
| Ours (LSTM + Flair)                                       | 0.1                                                          | 71520           | 4089          | 0.9833     | 0.9031    |
| Ours (LSTM + Flair)                                       | 0.2                                                          | 143709          | 8177          | 0.9856                    | 0.8998     |
| Ours (LSTM + Flair)                                       | 0.3                                                          | 217622          | 12265         | 0.987       | 0.9068     |
| Ours (LSTM + Flair)                                       | 0.4                                                          | 287325          | 16354         | 0.9873                    | 0.9017                   |
| Ours (LSTM + Flair)                                       | 0.5                                                          | 360571          | 20442         | 0.9878     | 0.9107     |
| Ours (LSTM + Flair)                                       | 0.7                                                          | 503520          | 28619         | 0.9891     | 0.9078      |
| Ours (LSTM + Flair)                                       | 1.0                                                          | 719826          | 40883         | 0.9893** | **0.9112** |
| LSTM + Word-Char embedding(Horsmann et Zesch 2017) | 1.0 | -               | -             | acc 0.976                                |                                          |

The evaluation on the CMC (4722 tokens) and WEB (7425 tokens) data yields results of lower accuracy (see Table 2). The decrease is especially noticeable for the CMC dataset since it contains chat, blog comments and twitter texts that are highly inconsistent and provide many OOV-words [2](#sdfootnote2sym). Comparing to the previous results on the same test set, the accuracy is 1% lower. The model presented by Thater 2017, however, was trained on the data including the “in-domain” subset from EmpiriST corpus that helped the model to adapt to the test set. From this perspective, our evaluation is more strict that does not prevent our model to demonstrate high accuracy performance. On the Web data, the accuracy stays high and in this case, our approach achieves better scores than by the previous evaluation on the same dataset: 96% vs. 93%. (To evaluate the models on the additional test sets, classification_report from sklearn.metrics was used). 

**Table 2: Evaluation on CMC and Web test data (accuracy)**

| Model                                        | model (split)                   | Acc. on CMC | Acc. on WEB |
| -------------------------------------------- | ------------------------------- | ----------- | ----------- |
| Ours (LSTM + Flair)                          | 0.05                            | 0.86        | 0.95        |
| Ours (LSTM + Flair)                          | 0.1                             | 0.87        | 0.96        |
| Ours (LSTM + Flair)                          | 0.5                             | 0.87        | 0.96        |
| Ours (LSTM + Flair)                          | 1.0                             | 0.87        | 0.97        |
| HMM + distributional smoothing (Thater 2017) | trained on the TIGER + EmpiriST | 0.8838      | 0.9335      |

Training and evaluating a model on two different datasets inevitably leads to the drop of the accuracy, also partly due to the inter-annotator agreement that will be always lower between two different sets than within the single one. The analysis of the errors on the CMC dataset shows that the most frequent classes with low scores (precision, recall, F1) are:

**ITJ**       0.76      0.89      0.82 — “Interjektion (mhm, ach, tja)”[3](#sdfootnote3sym); in 45% of all misclassifications substituted by FM (“Fremdsprachliches Material”).

**NE**       0.64      0.52      0.57 — “Eigennamen”; in most of the cases it is misclassified as NN (32% of all misclassifications) or FM (58% of all misclassifications).

**KOUS**       0.79      0.76      0.77 — “unterordnende Konjunktion mit Satz (*weil, dass, damit, wenn, ob*)”; is misclassified as PWAV (“adverbiales Interrogativ- oder Relativpronomen”).

Among other mistakes there were substitutions of imperative verbs by finite verbs, misclassification of answer particles (*danke, ja*) and wrong classification of ambiguous POS (see Table 3). Ambiguous POS are, for example, ***beste*** **<ADJA/NN** from the 2d sentence, ***endlich*** **<ADV/ADJD**> from the 4th sentence, ***etwas*** **<PIS/PIAT>** from the 5th sentence. As these results show, the contextual embeddings are still not able to disambiguate some of the more difficult cases. 

**Table 3: Some of the errors (from the CMC)**

| Sentence <pred/true> | Comment |
| --- | --- |
| 1) 		**Danke <VVFIN/PTKANT**> für <APPR> den <ART> 		Tip <NN> . <$.> **schreibe <VVFIN/VVIMP**> 		meine <PPOSAT> **addy <NE/NN**> auf <PTKVZ> : 		<$.> [ <$(> Email-Adresse <NN> ] <$(> **ja 		<PTKVZ/PTKANT**> , <$,> **tschüss <VVFIN/ITJ**> 		ihr <PPER> zwei <CARD> . <$.> | PTKANT — Antwortpartikel — classified as a finite verb. Imperative verb classified as a finite verb.<br />ITJ — Interjektion — classified as a finite verb. |
| 2) Und 		<KON> viele <PIAT> Geschenke <NN> und <KON> 		einen <ART> guten <ADJA> Rutsch <NN> und <KON> 		überhaupt <ADV> das <ART> **beste <ADJA/NN**> 		! <$.> du <PPER> mich <PPER> auch <ADV> 		... <$(> hübsche <ADJA> **nikolaus <NE/NN**> 		! <$.> du <PPER> mich <PPER> **mehr <ADV/PIS**> 		! <$.> **na <ADV/ITJ> szia <NE/FM>** !!!! 		<$.> | ITJ — Interjektion — classified as an adverb.        |
| 3) 		HErr <NE> Özdemir <NE> , <$,> antworten 		<VVFIN/**VVIMP** > Sie <PPER> ! <$.> | Imperative verb classified as a finite verb.         |
| 4) 		verherend <ADJD> ist <VAFIN> die <ART> 		ideologische <ADJA> drogenpolitik <NN> **a <FM/KOKOM> 		la <FM/KOKOM> dsu <FM/NE>** Wollen <VMFIN> Sie 		<PPER> den <ART> Irrsinn <NN> nicht <PTKNEG> 		**endlich <ADV/ADJD**> beenden <VVINF> , <$,> 		die <ART> Alltagsdrogen <NN> Nikotin <NN> , <$,> 		Medikamentenmissbrauch <NN> und <KON> Alkohol <NN> 		zu <PTKZU> verharmlosen <VVINF> und <KON> den 		<ART> Gebrauch <NN> von <APPR> 		Cannabis-Produkten <NN> zu <PTKZU> kriminalisieren 		<VVINF> . <$.> | KOKOM — Vergleichspartikel ohne Satz; <br />FM — 		Fremdsprachliches Material (not necessarily a fault of the tagger, since this can be considered as short sequence of FM). |
| 5) 		Wissenschaft <NN> sagt <VVFIN> **etwas <PIS/PIAT>** 		anderes <PIS> ! <$.> | PIAT — attributierendes Indefinitpronomen ohne Determiner; <br />PIS — substituierendes Indefinitpronomen. Two categories are very close to each other. |
| 6) die 		<ART> niederländer <NN> bedauern <VVFIN> heute 		<ADV> ihre <PPOSAT> entscheidung <NN> die <ART> 		freigabe <NN> von <APPR> **haschisch <ADJD/NN>** 		. <$.> | The form is close to an adjective one with suffix -isch. |



## Discussion

As we have shown, using contextual embeddings with a BiLSTM-CRF-Tagger yields new SOTA results for POS tagging in German. While having a fairly simple architecture, the tagger achieves high accuracy results even with the small data (<50k token) that is especially good for some languages, when only small datasets are available. Testing the models on the datasets with many OOV-words proved its strong ability to generalisation. The improvement comparing to the previous models is observed testing on web and CMC texts. The model still fails to disambiguate some of tags which means that contextual embeddings may be not enough to solve some highly ambiguous problems. Additional dependency-based word embeddings may improve the results even further, however, at the cost of increasing the complexity of the model.

Further experiments can also include a separate evaluation of in-vocabulary and OOV-words.

## References

Beißwenger, Michael, Sabine Bartsch, Stefan Evert, and Kay-Michael Würzner. "EmpiriST 2015: A shared task on the automatic linguistic annotation of computer-mediated communication and web corpora." In *Proceedings of the 10th Web as Corpus Workshop*, pp. 44-56. 2016.

Giesbrecht, Eugenie, and Stefan Evert. "Is part-of-speech tagging a solved task? An evaluation of POS taggers for the German web as corpus." In *Proceedings of the fifth Web as Corpus workshop*, pp. 27-35. 2009.

Horsmann, Tobias, and Torsten Zesch. "Do LSTMs really work so well for PoS tagging?–A replication study." In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*, pp. 727-736. 2017.

Thater, Stefan. "Fine-Grained POS Tagging of German Social Media and Web Texts." In *International Conference of the German Society for Computational Linguistics and Language Technology*, pp. 72-80. Springer, Cham, 2017.

## Footnotes


​	[1](#sdfootnote1anc)  https://sites.google.com/site/empirist2015/home/shared-task-data

​	[2](#sdfootnote2sym) Since our model is character-based, there are no actual OOV-words but rather unseen words during the training phase. We still refer to them as OOV in the following text.

​	[3](#sdfootnote3anc)   Annotation guidelines: https://sites.google.com/site/empirist2015/home/annotation-guidelines.



## Authors

- Alex Flueckiger
- Iuliia Nigmatulina