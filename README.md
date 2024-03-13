# Eavaluating Multilingual Language Models for Cross-Lingual ESG Issue Identification

<!-- Wing Yan Li, Emmanuele Chersoni, Cindy Sing Bik Ngai. 2024 *Evaluating Multilingual Language Models for Cross-Lingual ESG Issue Identification.* Under Review. -->

This repository provides codes and dataset for the paper titled: *Evaluating Multilingual Language Models for Cross-Lingual ESG Issue Identification*. Under Review.


### Dataset
The modified dataset with a unifies label space is provided in `data/`.

Datasets are `.json` files with an additional `bin_label` column storing lists of binarised labels per instance.

`data/labels.pickles` is the list of 35 pre-defined ESG key issues (labels).


### Usage for off-the-shelf experiments
Sentence representations have to be extracted and saved to a pickle file prior to the classification.
Saving directory should looks like `{YOUR_PICKLE_PATH}/{LANG}_[train or test].pkl`.

Train model.
```sh
python offTheShelf/train.py --pickle_path {YOUR_PICKLE_PATH}
                            --out_path {YOUR_OUTPUT_PATH}
```

Test model.
```sh
python offTheShelf/test.py --pickle_path {YOUR_PICKLE_PATH}
                           --classifier_path {YOUR_OUTPUT_PATH}
```

### Usage for fine-tuning experiments
GPU can be used for fine-tuning models during training.

```sh
CUDA_VISIBLE_DEVICES=0 python fineTuning/train.py
                                --base_model {YOUR_BASE_MODEL_NAME}
                                --best_model_metric f1_macro
                                --model_name {YOUR_MODEL_NAME}
                                --epoch 50
                                --best_model_dir {DIR_TO_SAVE_MODELS}
```

Test model.
```sh
CUDA_VISIBLE_DEVICES=0 python fineTuning/inference.py
                                --trained_model {YOUR_BEST_MODEL}
                                --out_path {DIR_TO_SAVE_RESULTS}
                                --test_file data/{LANG}_test.pkl
```
