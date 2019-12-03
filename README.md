A PyTorch implementation of the BI-LSTM-CRF model.

# Features:
- General implementation of CRF module
    - Full vectorized implementation: 1) no "for loop" in score sentence algorithm; 2) single level "for loop" in viterbi and forward algorithm
    - START/STOP tags are automatically added in CRF
    - A inner Linear Layer is included which transform from features space to tag space
- General implementation of BI-LSTM-CRF model
- CUDA supported
- Full support for batch computation: training, predicting
- Specialized for NLP sequence tagging tasks
- Easy to train your own sequence tagging models
- MIT License

# INSTALL
- dependencies
    - Python 3
    - [PyTorch](https://pytorch.org/)
- install
    ```sh
    $ pip install bi-lstm-crf
    ```

# TRAIN
### corpus
- prepare your corpus in the specified [structure and format](https://github.com/jidasheng/bi-lstm-crf/wiki/corpus-structure-and-format). 
- there is also a sample corpus in `bi_lstm_crf/app/sample_corpus`.


### train
- training
    ```sh
    $ python -m bi_lstm_crf corpus_dir --model_dir "model_xxx"
    ```
    - more [options](https://github.com/jidasheng/bi-lstm-crf/wiki/training-options)
- training results
    - the training results are saved in the model_dir, you can load and draw loss curve
        ```python
        import pandas as pd
        import matplotlib.pyplot as plt
        
        df = pd.read_csv(".../model_dir/loss.csv")
        df[["train_loss", "val_loss"]].ffill().plot(grid=True)
        plt.show()
        ```

# PREDICT
```python
from bi_lstm_crf.app import WordsTagger

model = WordsTagger(model_dir)
print(model(["市领导到成都..."]))  # CHAR-based model
print(model([["市", "领导", "到", "成都", ...]]))  # WORD-based model
```

# MODULES
There modules are generally built and can be used in other projects
- CRF
- BiRnnCrf

# References
- [Zhiheng Huang, Wei Xu, and Kai Yu. 2015. Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991). arXiv:1508.01991.
- Pytorch tutorial [ADVANCED: MAKING DYNAMIC DECISIONS AND THE BI-LSTM CRF](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)
