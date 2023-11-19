# Defect Detection in Source Code (CodeXGLUE Dataset Code-Code)

In this project, we investigate and run models including CNN, LSTM, and Transformers on the Code-Code dataset of the CodeXGLUE benchmark. The dataset is designed for defect detection in source code and contains examples of code with and without defects. We preprocess the data and create a model to detect defects in the source code.

## Dataset

The original dataset can be downloaded from [here]. We have downloaded and preprocessed it by running a `preprocess.py` existing in the dataset folder. The preprocessed dataset includes `train.jsonl`, `valid.jsonl`, and `test.jsonl` files. For convenience, we trim the files and extract two necessary columns from it: `func` and `target`. Here, `func` includes the code and `target` is our label which is either 0 or 1.

## Models

We include three models: CNN, LSTM, and Transformers which can be found in the `models` folder. 

## Running the Models

To run each of the models, you can use one of the below commands:

```bash
py run.py --model cnn
py run.py --model lstm 
py run.py --model transformers
```

Note: In Linux or Mac, one should use `python` instead of `py`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
