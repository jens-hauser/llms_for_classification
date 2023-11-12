from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score


def get_tokenized_dataset(name, tokenizer, max_length=128):
    def tokenize_function(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=max_length
        )

    data = load_dataset(name)
    if "tweet" in data["train"].column_names:
        data = data.rename_column("tweet", "text")
        data = data.rename_column("class", "label")
        for col in ['hate_speech_count', 'offensive_language_count', 'neither_count', 'count']:
            data = data.remove_columns(col)

    tokenized_datasets = data.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    if "test" not in tokenized_datasets:
        tokenized_datasets = tokenized_datasets["train"].train_test_split(0.2, stratify_by_column="labels", shuffle=True)

    return tokenized_datasets["train"].shuffle(seed=42), tokenized_datasets[
        "test"
    ].shuffle(seed=42)



def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {"f1": f1, "accuracy": acc}
