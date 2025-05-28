from typing import Union, Literal
import pandas as pd

class MbppDataset:
    def __init__(self, split: Union[Literal['train'], Literal['val'], Literal['test'], Literal['prompt']],):
        self._splits = {'train': 'full/train-00000-of-00001.parquet', 'test': 'full/test-00000-of-00001.parquet', 'val': 'full/validation-00000-of-00001.parquet', 'prompt': 'full/prompt-00000-of-00001.parquet'}
        self.df = pd.read_parquet("hf://datasets/google-research-datasets/mbpp/" + self._splits[split])
        # self.df = self.df.sample(frac=0.2).reset_index(drop=True)
        self.df = process_data(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.df.iloc[index]

class MbppDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            self._shuffle_indices()
        self.index = 0

    def _shuffle_indices(self):
        import random
        random.shuffle(self.indices)

    def __iter__(self):
        batch = []
        for i in range(len(self.indices)):
            batch.append(self.dataset[self.indices[i]])
            if len(batch) == self.batch_size or i == len(self.indices) - 1:
                yield batch
                batch = []

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration

        batch_indices = self.indices[self.index:self.index + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        self.index += self.batch_size
        return batch

def process_data(df: pd.DataFrame):
    tasks = []
    for i, data_entry in df.iterrows():
        prompt = data_entry["text"]
        test_case = data_entry["test_list"]
        tests = ""
        for test in test_case:
            tests+="\n"+test
        text = f"""
**Task**:
```python
{prompt}
```
Your code should pass these tests:
```python
{tests}
```
"""
        tasks.append(text)
    df["task"] = tasks
    return df

MbppDataset(split='test')