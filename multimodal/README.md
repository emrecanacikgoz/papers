# Multimodality Survey

## VL-Bert
[r221008:1908:Microsoft:940cite:VL-BERT:Pre-training of Generic Visual-Linguistic Representations.pdf](https://arxiv.org/pdf/1908.08530.pdf). 

VL-BERT takes both visual and linguistic embedded features as input. Each input element is either of a word from the input sentence, or a region-of-interest (RoI) from the input image, together with certain special elements to disambiguate different input formats. For each input element, its embedding feature is the summation of four types of embedding: token embedding, visual feature embedding, segment embedding, and sequence position embedding. The visual feature embedding is newly introduced for capturing visual clues, while the other three embeddings follow the design in the original BERT paper. VL-BERT is pre-trained on two different tasks: Masked Language Modeling with Visual Clues and Masked RoI Classification with Linguistic Clues. In MLM with Visual Clues task, each word in the input sentence are masked randomly with the probability of 15% and the model is trained to predict masked words by looking on the unmasked words in the sentence together with the visual features. Masked RoI Classification with Linguistic Clues: To avoid any visual clue leakage from the visual feature embedding of other elements, the pixels laid in the masked RoI are set as zeros before applying Fast R-CNN (What?). During pre-training, the final output feature corresponding to the masked RoI is fed into a classifier with Softmax cross-entropy loss for object category classification. The category label predicted by pre-trained Faster R-CNN is set as the ground-truth. In VL-BERT, the parameters of Fast R-CNN are also updated. To avoid visual clue leakage in the pre-training task of Masked RoI Classification with Linguistic Clues, the masking operation is applied to the input raw pixels, other than the feature maps produced by layers of convolution. VL-BERT is pre-trained on Cenceptual Captions Dataset (visual-linguistic), Books Corpus (text), English Wikipedia (text). The captions of CC is clauses that are too simple and short which leads overfitting. So, VL-BERT is also pretrained on text-only Books Corpus and the English Wikipedia datasets. 

### Appendix
```
- Token Embedding: a special token which is assigned for each of the inputs (i.e. [CLS], [WORD], [MASK], [SEP], [IMG], [END]).
- Visual Feature Embedding: Visual feature embeddings are combined form of visual appearance features and visual geometry embeddings. The visual appearance feature is extracted by applying a Fast R-CNN detector as the visual feature embedding (of 2048-d in paper). The visual geometry embedding is designed to inform VL-BERT the geometry location of each input visual element in image. Each RoI is characterized by a 4-d vector, as ( xLT , yLT , xRB , hRB ). Following the practice in Relation Networks (Hu et al., 2018), the 4-d vector is embedded into a high-dimensional representation (of 2048-d in paper) by computing sine and cosine functions of different wavelengths. The visual feature embedding is added to all the inputs, which is an output of a fully connected layer taking the concatenation of visual appearance feature and visual geometry embedding as input.
- Segment Embedding:  A learned segment embedding is added to every input element for indicating which segment it belongs to. There are three types of segments in VL-BERT: A, B, C. They are defined to separate input elements from different sources. A and B are defined for the words from the first and second input sentence, and C for the RoIs from the input image. For QA the format is <Question, Answer, Image> where A denotes Question, B denotes Answer, and C denotes Image. For Image-Caption task, the input format is <Caption, Image> where A denotes Caption, and C denotes Image.
- Sequence Position Embedding: Same in BERT. Each input element indicated its order in the input sequence and the learnable sequence position embedding is added to every input element indicating its order in the input sequence. The sequence position embedding for all visual elements are the same since there is no natural order among input visual elements.
```
![alt text](https://github.com/emrecanacikgoz/papers/blob/main/multimodal/figs/vlbert.png)



## Lxmert
[r221114:1908:UNC:1130cite:LXMERT:Learning Cross-Modality Encoder Representations from Transformers.pdf](https://arxiv.org/pdf/1908.07490.pdf)

Lxmert built on two single-modal network architectures that is used for source sentences and images respectively and a follow-up cross-modal Encoder combines these two modality. Lxmert consist of three encoders: object relationship encoder, language encoder, and cross-modality encoder. It is pre-trained on 5 different tasks: masked language modeling, masked object prediction via RoI-feature regressin, masked objectd predicition via detected-label classification, cross-modallity matching, and image question answering. On the other hand, is tis pre-trained on MS COCO(captioning), Visual Genome (captioning), VQA v2.0 (question answering), GQA (question answering), and VG-QA (question answering) datasets. They pre-train all the encoders and embedding layers from scratch, they didn't use any LLM embeddings for initialization. They used WordPiece tokenizer as in Bert for language side before giving them to the language encoder and they used 101-layer Faster R-CNN (pre-trained on Visual Genome) as feature extractor for image side before feeding them to object-relationship encoder. They accepted these detected labels output as ground truths. The outputs of the object-relationship encoder and language encoder are given to cross-modality encoder to align the features between these two modality to learn the joint representations. At the end, model produces three outputs as vision output (RoI Feature Regression + Detected-Label Classification), cross-modality output (Cross-Modality Matching + Q/A), and language output (Masked Cross-Modality LM). Pros: pre-trained on a massive visual question answering data. Cons: Single-stream architecture that contaions two single modal Transformer for vision and language, followed by one cross-modal Transformer for joint representation.

![alt text](https://github.com/emrecanacikgoz/papers/blob/main/multimodal/figs/lxmert.png)


## Visual-Bert
To-Do

## ViLBert
To-Do

