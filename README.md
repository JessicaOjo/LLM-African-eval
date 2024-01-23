# How Good Are Large Language Models on African Languages

[paper](https://arxiv.org/abs/2311.07978)

>Recent advancements in natural language processing have led to the proliferation of large language models (LLMs). These models have been shown to yield good performance, using in-context learning, even on unseen tasks and languages. Additionally, they have been widely adopted as language-model-as-a-service commercial APIs like GPT-4 API. However, their performance on African languages is largely unknown. We present an analysis of three popular large language models (mT0, LLaMa 2, and GPT-4) on five tasks (news topic classification, sentiment classification, machine translation, question answering, and named entity recognition) across 30 African languages, spanning different language families and geographical regions. Our results suggest that all LLMs produce below-par performance on African languages, and there is a large gap in performance compared to high-resource languages like English most tasks. We find that GPT-4 has an average or impressive performance on classification tasks but very poor results on generative tasks like machine translation. Surprisingly, we find that mT0 had the best overall on cross-lingual QA, better than the state-of-the-art supervised model (i.e. fine-tuned mT5) and GPT-4 on African languages. Overall, LLaMa 2 records the worst performance due to its limited multilingual capabilities and English-centric pre-training corpus. In general, our findings present a call-to-action to ensure African languages are well represented in large language models, given their growing popularity.

## Languages Evaluated

|Language              |Family/branch                |Region                |Script |speakers|NewsClass |Sentiment |NER |QA  |MT|No. of tasks|
|:---------------------|:---------------------------:|:---------------------|:------|:------:|:--------:|:--------:|:--:|----|--|:----------:|
|Hausa (hau)           |Afro-Asiatic / Chadic        |West Africa           |Latin  |77M     |✓         |✓         |✓   |✓   |✓ |5           | 
|Amharic (amh)         |Afro-Asiatic / Ethio-Semitic |East Africa           |Ge’ez  |57M     |✓         |✓         |✓   |✗   |✓ |4           | 
|Oromo (orm)           |Afro-Asiatic / Cushitic      |East Africa           |Latin  |37M     |✓         |✓         |✗   |✗   |✗ |2           |
|Algerian Arabic (arq) |Afro-Asiatic / Semitic       |North Africa          |Arabic |41M     |✗         |✓         |✗   |✗   |✗ |1           |
|Moroccan Arabic (ary) |Afro-Asiatic / Semitic       |North Africa          |Arabic |33M     |✗         |✓         |✗   |✗   |✗ |1           |
|Somali (som)          |Afro-Asiatic / Cushitic      |East Africa           |Latin  |22M     |✓         |✗         |✗   |✗   |✗ |1           |
|Tigrinya (tig)        |Afro-Asiatic / Ethio-Semitic |East Africa           |Ge’ez  |9M      |✓         |✓         |✗   |✗   |✗ |1           |
|Kiswahili (swa)       |Niger-Congo / Bantu          |East & Central Africa |Latin  |71M-106M|✓         |✓         |✓   |✓   |✓ |5           |
|Yorùbá (yor)          |Niger-Congo / Volta-Niger    |West Africa           |Latin  |46M     |✓         |✓         |✓   |✓   |✓ |5           |
|Igbo (ibo)            |Niger-Congo / Volta-Niger    |West Africa           |Latin  |31M     |✓         |✓         |✓   |✓   |✓ |5           |
|Kinyarwanda (kin)     |Niger-Congo / Bantu          |East Africa           |Latin  |10M     |✗         |✓         |✓   |✓   |✓ |4           |
|Twi (twi)             |Niger-Congo / Kwa            |West Africa           |Latin  |9M      |✗         |✓         |✓   |✓   |✓ |4           |
|Luganda (lug)         |Niger-Congo / Bantu          |Central Africa        |Latin  |11M     |✓         |✗         |✓   |✗   |✓ |3           |
|isiXhosa (xho)        |Niger-Congo / Bantu          |Southern Africa       |Latin  |19M     |✓         |✗         |✓   |✗   |✓ |3           |
|isiZulu (zul)         |Niger-Congo / Bantu          |Southern Africa       |Latin  |27M     |✗         |✗         |✓   |✓   |✓ |3           |
|chiShona (sna)        |Niger-Congo / Bantu          |Southern Africa       |Latin  |11M     |✓         |✗         |✓   |✗   |✓ |3           |
|Wolof (wol)           |Niger-Congo / Senegambia     |West Africa           |Latin  |5M      |✗         |✗         |✓   |✓   |✓ |3           |
|Bambara (bam)         |Niger-Congo / Mande          |West Africa           |Latin  |14M     |✗         |✗         |✓   |✗   |✓ |2           |
|Fon (fon)             |Niger-Congo / Volta-Niger    |West Africa           |Latin  |14M     |✗         |✗         |✗   |✓   |✓ |2           |
|Éwé (ewe)             |Niger-Congo / Kwa            |West Africa           |Latin  |7M      |✗         |✗         |✓   |✗   |✓ |2           |
|Ghomálá’ (bbj)        |Niger-Congo / Grassfields    |Central               |Latin  |1M      |✗         |✗         |✓   |✗   |✓ |2           |
|Chichewa (nya)        |Niger-Congo / Bantu          |South-East Africa     |Latin  |14M     |✗         |✗         |✓   |✗   |✓ |2           |
|Mossi (mos)           |Niger-Congo / Gur            |West Africa           |Latin  |8M      |✗         |✗         |✓   |✗   |✓ |2           |
|Setswana (tsn)        |Niger-Congo / Bantu          |Southern Africa       |Latin  |14M     |✗         |✗         |✓   |✗   |✓ |2           |
|Bemba (bem)           |Niger-Congo / Bantu          |South, East & Central |Latin  |4M      |✗         |✗         |✗   |✓   |✗ |1           |
|Lingala (lin)         |Niger-Congo / Bantu          |Central Africa        |Latin  |40M     |✓         |✗         |✗   |✗   |✗ |1           |
|Rundi (run)           |Niger-Congo / Bantu          |East Africa           |Latin  |11M     |✓         |✗         |✗   |✗   |✗ |1           |
|Xitsonga (tso)        |Niger-Congo / Bantu          |Southern Africa       |Latin  |7M      |✗         |✓         |✗   |✗   |✗ |1           |
|Luo (luo)             |Nilo-Saharan                 |East Africa           |Latin  |4M      |✗         |✗         |✓   |✗   |✗ |1           |
|Naija (pcm)           |English Creole               |West Africa           |Latin  |121M    |✓         |✓         |✓   |✗   |✓ |4           |
| Languages/task       |                             |                      |       |        |14        |13        |20  |10  |20|            |


### BibTeX entry and citation info
```
@misc{ojo2023good,
      title={How good are Large Language Models on African Languages?}, 
      author={Jessica Ojo and Kelechi Ogueji and Pontus Stenetorp and David I. Adelani},
      year={2023},
      eprint={2311.07978},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
