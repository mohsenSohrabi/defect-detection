# Defect Detection in Source Code (CodeXGLUE Dataset Code-Code)

This project involves the investigation and implementation of models, including CNN, LSTM, and Transformers, on the Code-Code dataset of the CodeXGLUE benchmark. The dataset is specifically designed for defect detection in source code and comprises examples of code with and without defects. We preprocess the data and develop a model to detect defects in the source code.

## Dataset

The original dataset can be downloaded from [here](https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view). We have downloaded and preprocessed it using a `preprocess.py` script located in the dataset folder. The preprocessed dataset includes `train.jsonl`, `valid.jsonl`, and `test.jsonl` files. For convenience, we have trimmed the files and extracted two necessary columns: `func` and `target`. The `func` column includes the code, and `target` is our label, which is either 0 or 1.

## Models

This project includes three models: CNN, LSTM, and Transformers. These models can be found in the `models` folder. 

## Running the Models

Each command corresponds to running a specific model:

- `py run.py --model cnn` : This command is used to run the CNN model.
- `py run.py --model lstm` : This command is used to run the LSTM model.
- `py run.py --model transformers` : This command is used to run the Transformers model.

Please note that on Linux or Mac, you should use `python` instead of `py`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
