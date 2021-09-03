# Bert\_fine\_tuning

**개인 study를 위한 자료입니다.**  
 **그러다보니 내용에 잘못된 점이 있습니다.**

## 1 Advantages of Fine-Tuning <a id="1-advantages-of-fine-tuning"></a>

pre-trained된 model을 사용한, Fine-Tuning은 아래와 같은 장점이 있습니다.

1. Time
2. Less Data
3. Better Results

실제 구현 예제로, huggingface\[[1](https://huggingface.co/transformers/custom_datasets.html)\]에서 제공하는 pre-trained model을 바탕으로, fine-tuning을 진행해보려고 합니다. 구현하고자 하는 model은 아래와 그림과 같습니다.\[[2](https://www.groundai.com/project/sentence-bert-sentence-embeddings-using-siamese-bert-networks/1)\] ![SBERT architecture with consine-smiliarity](https://dj-woo.github.io/img/bert_fine_tuning/sbert.png) 코드는 주로 \[[3](https://mccormickml.com/2019/07/22/BERT-fine-tuning)\]을 참조했습니다. fine-tuning을 위한 code는 크게 아래와 같이 나누어져 있습니다.

1. bertFineTuningWithConnectionData.py : fine-tuning을 진행합니다.
2. ConnectionBert.py : pre-trained 된 model을 load 합니다.
3. ConnectionDataset.py : fine-tuning을 위한 Dataset, DataLoader가 정의 되어 있습니다.

## 2 구현 <a id="2-&#xAD6C;&#xD604;"></a>

### 1 bertFineTuningWithConnectionData <a id="1-bertfinetuningwithconnectiondata"></a>

`transformers`를 사용하여, pre-trained model를 불러오는 명령어는 매우 간단합니다. 하지만, 특별한 사정으로 미리 다운을 받아서 사용해야 할 경우, 아래 방법을 사용하면 됩니다.

> huggingsface 접속 -&gt; MODELS\(우측상단\) -&gt; 원하는 model 검색 -&gt; `List all files in model` 클릭 -&gt; `config.json`, `pythorch_model.bin`, `vocab.txt`를 원하는 directory에 저장 -&gt; directory load

여기서는 “transformers\bert\bert-base-uncased"에 위 3 파일을 저장해 놓고 사용했습니다.

### 2 ConnectionBert <a id="2-connectionbert"></a>

pre-trained 된 data를 load하여 사용하는 것은 매우 간단합니다.

model에 대한 자세한 설명은 [Docs»Transformers](https://huggingface.co/transformers/)에서 확인 할 수 있습니다. 가장 기본이 되는 BertModel의 경우, embedding layer + bertEncoder + pooled layer로 되어있습니다. 자세한 내부 weight parameter는 `print(model)`로 확인 할 수 있습니다.

```text
BertModel(
  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(30522, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): BertEncoder(
    (layer): ModuleList(
      (0): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (1~11): BertLayer()
    )
  )
  (pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
)      
```

### 3. ConnectionDataset <a id="3-connectiondataset"></a>

Dataset은 `__len__()`과 `__getitem__()`만 구현해주면, 쉽게 구현할 수 있습니다.

### Reference <a id="reference"></a>

* **\[1\]** [huggingface](https://huggingface.co/transformers/custom_datasets.html)
* **\[2\]** [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://www.groundai.com/project/sentence-bert-sentence-embeddings-using-siamese-bert-networks/1)
* **\[3\]** [BERT Fine-Tuning Tutorial with PyTorch](https://www.groundai.com/project/sentence-bert-sentence-embeddings-using-siamese-bert-networks/1)

