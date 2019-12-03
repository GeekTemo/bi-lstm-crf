# INTRODUCTION
- implementation for the BI-LSTM-CRF model
    - [Zhiheng Huang, Wei Xu, and Kai Yu. 2015 - Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)
- special for NLP sequence tagging tasks
- EASY TO USE
    - train your own NLP model such as WORD SEGMENTATION, POS TAGGING, NAMED ENTITY RECOGNITION

# INSTALL
- dependencies
    - [Pytorch](https://pytorch.org/)
- install
    ```sh
    $ pip install bi-lstm-crf
    ```

# MODEL
```text
Embedding -> Bidirectional LSTM/GRU -> Linear -> CRF
```

# TRAIN
### dataset
- data structure
    ```
    corpus_dir
        char.json
        tag.json
        data.csv or data.json
    ```
- `vocab.json`
    - a list of unique CHARs or WORDs, dependent of your choice
    - examples
        - CHAR-based: `["市", "领", "导", "到", "成", "都", ...]`
        - WORD-based: `["市", "领导", "到", "成都", ...]`
- `tag.json`
    - a list of tags
    - examples
        - WORD SEGMENTATION: `["B", "I"]`
        - NER: `["O", "B-ORG", "I-ORG", ...]`
- `data.json`
    ```python
    [   # a list of sentences
        [
            ["市", "领", "导", "到", "成", "都", ...],  # sentence
            ["B", "B", "I", "B", "B", "I", ...],        # tags
        ],
        ...
    ]
    ```
- `data.csv`
    sentence | tags
    ---|---
    `"[""市"", ""领"", ""导"", ""到"", ""成"", ""都"", ...]"` | `"[""B"", ""B"", ""I"", ""B"", ""B"", ""I"", ...]"`
    ... | ...

### train
- sh
    ```sh
    $ python -m bi-lstm-crf corpus_dir --model_dir "model_xxx"
    ```
- python
    ```python
    from bi_lstm_crf import train
    
    train(corpus_dir="corpus_xxx", model_dir="model_xxx")
    ```
- other options
    - model parameters
        name | default | description
        --- | --- | ---
        embedding_dim | 100 | the dimension of the embedding layer
        hidden_dim | 128 | the dimension of the RNN hidden state
        num_rnn_layer | 1 | the number of RNN layers
        rnn | "lstm" | RNN type, choice: "lstm", "gru"
    - training options
        name | default | description
        --- | --- | ---
        num_epoch | 20 | number of epoch to train
        lr | 1e-3 | learning rate
        batch_size | 1000 | batch size
        recovery | True | continue to train from the saved model in model_dir


# PREDICT
- usage
    ```python
    from bi_lstm_crf import BiRnnCrf
    
    model = BiRnnCrf(model_dir)
    print(mode("some text"))
    ```
