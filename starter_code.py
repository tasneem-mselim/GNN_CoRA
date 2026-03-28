import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


DATA_FILES = {
    'train_val': 'train_val_labeled.csv',
    'adjacency': 'adjacency_matrix.csv',
    'test_features': 'test_features_only.csv',
}


def get_path(data_dir, filename):
    return os.path.join(data_dir, filename)


def ensure_files_exist(data_dir):
    missing = []
    for fname in DATA_FILES.values():
        path = get_path(data_dir, fname)
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        msg = ["Missing dataset files. Place the required CSV files in the data directory or pass --data-dir."]
        msg.append("Required files:")
        msg.extend(f"  - {os.path.basename(path)}" for path in [get_path(data_dir, f) for f in DATA_FILES.values()])
        msg.append("")
        msg.append("Example:")
        msg.append("  python starter_code.py --data-dir .")
        raise FileNotFoundError("\n".join(msg))


def infer_split_indices(df):
    if 'split' in df.columns:
        split = df['split'].astype(str).str.lower()
        train_ids = df.loc[split == 'train', 'id'].astype(int).to_numpy()
        val_ids = df.loc[split == 'val', 'id'].astype(int).to_numpy()
        if train_ids.size > 0 and val_ids.size > 0:
            return train_ids, val_ids

    if 'subset' in df.columns:
        subset = df['subset'].astype(str).str.lower()
        train_ids = df.loc[subset == 'train', 'id'].astype(int).to_numpy()
        val_ids = df.loc[subset == 'val', 'id'].astype(int).to_numpy()
        if train_ids.size > 0 and val_ids.size > 0:
            return train_ids, val_ids

    if len(df) >= 640:
        ids = df['id'].astype(int).to_numpy()
        return ids[:140], ids[140:640]

    raise ValueError(
        'Unable to infer train/validation split from train_val_labeled.csv. '
        'Please include a `split` column or provide 640 labeled nodes in standard order.'
    )


def normalize_adj(adj):
    adj = adj.astype(np.float64)
    adj = adj + np.eye(adj.shape[0], dtype=adj.dtype)
    row_sum = np.sum(adj, axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return adj / row_sum


def parse_args():
    parser = argparse.ArgumentParser(description='Run a non-TensorFlow Cora baseline.')
    parser.add_argument('--data-dir', default='.', help='Directory containing dataset CSV files')
    parser.add_argument('--output', default='submission.csv', help='Submission output filename')
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    ensure_files_exist(data_dir)

    print('Loading data...')
    df_train_val = pd.read_csv(get_path(data_dir, DATA_FILES['train_val']))
    adj = pd.read_csv(get_path(data_dir, DATA_FILES['adjacency']), header=None).values
    df_test = pd.read_csv(get_path(data_dir, DATA_FILES['test_features']))

    num_nodes = adj.shape[0]
    feature_columns = [c for c in df_train_val.columns if c.startswith('word_')]
    if len(feature_columns) == 0:
        raise ValueError('No feature columns found in train_val_labeled.csv.')

    print(f'  num_nodes = {num_nodes}')
    print(f'  num_features = {len(feature_columns)}')

    features = np.zeros((num_nodes, len(feature_columns)), dtype=np.float32)
    labels = np.full((num_nodes,), -1, dtype=np.int32)

    train_val_ids = df_train_val['id'].astype(int).to_numpy()
    features[train_val_ids] = df_train_val[feature_columns].astype(np.float32).to_numpy()
    labels[train_val_ids] = df_train_val['target'].astype(int).to_numpy()

    test_ids = df_test['id'].astype(int).to_numpy()
    test_feature_columns = [c for c in df_test.columns if c.startswith('word_')]
    if feature_columns != test_feature_columns:
        raise ValueError('Feature columns in test_features_only.csv do not match train_val_labeled.csv')
    features[test_ids] = df_test[feature_columns].astype(np.float32).to_numpy()

    train_ids, val_ids = infer_split_indices(df_train_val)
    train_ids = train_ids.astype(int)
    val_ids = val_ids.astype(int)

    num_classes = int(df_train_val['target'].astype(int).max() + 1)
    print(f'  num_classes = {num_classes}')

    print('Training multinomial Naive Bayes on labeled data...')
    model = MultinomialNB()
    model.fit(features[train_val_ids], labels[train_val_ids])

    if val_ids.size > 0:
        print('Computing validation accuracy...')
        val_pred = model.predict(features[val_ids])
        val_true = labels[val_ids]
        val_acc = accuracy_score(val_true, val_pred)
        print(f'Validation accuracy: {val_acc:.4f}')

    submission = pd.DataFrame({'id': test_ids, 'target': model.predict(features[test_ids]).astype(int)})
    submission.to_csv(args.output, index=False)
    print(f"Saved submission to '{args.output}'")


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)

