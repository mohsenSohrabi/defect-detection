# Defect Detection in Source Code (CodeXGLUE Dataset Code-Code)

This project is focused on the exploration and application of various models, including CNN, LSTM, and Transformers, on the Code-Code dataset from the CodeXGLUE benchmark. This dataset is specifically curated for defect detection in source code, providing examples of both defective and non-defective code. We preprocess this data and construct a model capable of detecting defects in the source code.

## Dataset

The original dataset can be accessed [here](https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view). We have downloaded and preprocessed it using a `preprocess.py` script, which can be found in the dataset folder. **Note that** I got this script from [here](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection/dataset). The preprocessed dataset includes `train.jsonl`, `valid.jsonl`, and `test.jsonl` files. For ease of use, we have trimmed these files and extracted two essential columns: `func` and `target`. The `func` column contains the code, and `target` is our label, which is either 0 or 1. After trimming, we saved them as 'train_processed.jsonl', 'valid_processed.jsonl', and 'test_processed.jsonl' for further use.

## Models

This project incorporates three models: CNN, LSTM, and Transformers. These models can be located in the `models` folder.

## Running the Models

Each command is associated with running a specific model:

- `py run.py --model cnn` : Executes the CNN model.
- `py run.py --model lstm` : Executes the LSTM model.
- `py run.py --model transformers` : Executes the Transformers model.

Please note that on Linux or Mac, `python` should be used in place of `py`.

## Contributing

We welcome pull requests. For significant changes, please open an issue first to discuss what you would like to change.

Please ensure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
