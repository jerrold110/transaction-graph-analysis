import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, auc,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / '1_traditional_model_results'
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(filepath_or_buffer=(Path(__file__).resolve().parent.parents[1] / "fraudTrain.csv"),
                index_col=0)
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df = df.loc[
    (df['trans_date_trans_time'] >= '2019-01-01')
    & (df['trans_date_trans_time'] < '2020-03-01')
].copy()


def add_transaction_history_features(transactions):
    """Build card histories from [current time - window, current time)."""
    transactions = transactions.loc[:, [
        'trans_date_trans_time', 'cc_num', 'merchant',
        'trans_num', 'is_fraud', 'amt',
    ]].copy()
    transactions = transactions.sort_values(
        'trans_date_trans_time', kind='stable'
    ).reset_index(drop=True)

    windows = {
        '1h': np.timedelta64(1, 'h'),
        '24h': np.timedelta64(24, 'h'),
        '7d': np.timedelta64(7, 'D'),
    }
    scopes = {
        'global': ['cc_num'],
        'merchant': ['cc_num', 'merchant'],
    }
    features = {}

    for scope, keys in scopes.items():
        for window in windows:
            prefix = f'{scope}_{window}'
            features[f'{prefix}_is_first'] = np.zeros(len(transactions), dtype=np.int8)
            features[f'{prefix}_recency_seconds'] = np.zeros(len(transactions))
            features[f'{prefix}_transaction_count'] = np.zeros(len(transactions), dtype=np.int64)
            features[f'{prefix}_total_amt'] = np.zeros(len(transactions))
            features[f'{prefix}_average_amt'] = np.zeros(len(transactions))

        for positions in transactions.groupby(keys, sort=False, dropna=False).indices.values():
            history = transactions.iloc[positions]
            times = history['trans_date_trans_time'].to_numpy(dtype='datetime64[ns]')
            amounts = history['amt'].to_numpy(dtype=float)
            cumulative_amounts = np.concatenate(([0.0], np.cumsum(amounts)))

            # Exclude all transactions at the current timestamp, including ties.
            ends = np.searchsorted(times, times, side='left')
            for window, duration in windows.items():
                prefix = f'{scope}_{window}'
                starts = np.searchsorted(times, times - duration, side='left')
                counts = ends - starts
                totals = cumulative_amounts[ends] - cumulative_amounts[starts]
                has_history = counts > 0
                recency = np.zeros(len(history))
                recency[has_history] = (
                    (times[has_history] - times[ends[has_history] - 1])
                    / np.timedelta64(1, 's')
                )

                features[f'{prefix}_is_first'][positions] = ~has_history
                features[f'{prefix}_recency_seconds'][positions] = recency
                features[f'{prefix}_transaction_count'][positions] = counts
                features[f'{prefix}_total_amt'][positions] = totals
                features[f'{prefix}_average_amt'][positions] = np.divide(
                    totals, counts, out=np.zeros(len(history)), where=has_history
                )

    return pd.concat([
        transactions[['trans_date_trans_time', 'trans_num', 'is_fraud', 'amt']],
        pd.DataFrame(features, index=transactions.index),
    ], axis=1)


def split_by_time(transactions):
    dates = transactions['trans_date_trans_time']
    train = transactions.loc[(dates >= '2019-02-01') & (dates < '2019-11-01')].copy()
    test = transactions.loc[(dates >= '2019-11-01') & (dates < '2020-02-01')].copy()
    return train, test


def save_metric_results(metrics, output_dir, model_order):
    """Save ordered results and grouped bars for each evaluation split."""
    metrics = metrics.set_index(['model', 'dataset']).reindex(
        pd.MultiIndex.from_product(
            [model_order, ['Train', 'Test']], names=['model', 'dataset']
        )
    ).reset_index()
    metrics.to_csv(output_dir / 'traditional_model_metrics.csv', index=False)
    metric_columns = ['accuracy', 'precision', 'recall', 'f1', 'pr_auc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'PR AUC']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, split in zip(axes, ['Train', 'Test']):
        values = metrics.loc[metrics['dataset'] == split].set_index('model')
        values.loc[model_order, metric_columns].rename(
            columns=dict(zip(metric_columns, labels))
        ).plot.bar(ax=ax, rot=0, width=0.8)
        ax.set_title(f'{split} metrics')
        ax.set_xlabel('Model')
        ax.set_ylim(0, 1.08)
        ax.grid(axis='y', alpha=0.25)
        ax.set_axisbelow(True)
        ax.get_legend().remove()
    axes[0].set_ylabel('Score')
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=5)
    fig.suptitle('Fraud detection model comparison (decision threshold = 0.5)')
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(output_dir / 'traditional_model_metrics.png', dpi=150)
    plt.close(fig)
    return metrics


# Display original data counts before replacing the raw columns with features.
raw_train, raw_test = split_by_time(df)
for name, dataset in [('Train', raw_train), ('Test', raw_test)]:
    counts = dataset['is_fraud'].value_counts().sort_index()
    print(f'\n{name}: original data', flush=True)
    print(counts.to_string())
    print(dataset.shape, flush=True)
del raw_train, raw_test, dataset

print('\nGenerating transaction history features...', flush=True)
df = add_transaction_history_features(df)
train_df, test_df = split_by_time(df)
excluded = ['trans_date_trans_time', 'trans_num', 'is_fraud']
X_train = train_df.drop(columns=excluded)
X_test = test_df.drop(columns=excluded)
y_train = train_df['is_fraud']
y_test = test_df['is_fraud']
print(f'Model input shapes: train={X_train.shape}, test={X_test.shape}', flush=True)

# Fit scaling and class weights on training data only. No resampling is used.
class_counts = y_train.value_counts()
models = {
    "Logistic regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=1,
        ),
    ),
    "Random forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=4,
        random_state=1,
    ),
    "XGBoost": xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=float(class_counts[0] / class_counts[1]),
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=4,
        random_state=1,
    ),
}

results = []
fig, ax = plt.subplots(figsize=(8, 6))
for name, model in models.items():
    print(f'\nTraining {name}...', flush=True)
    model.fit(X_train, y_train)
    for split, X, y in [('Train', X_train, y_train), ('Test', X_test, y_test)]:
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        curve_precision, curve_recall, _ = precision_recall_curve(y, probabilities)
        # Trapezoidal area under the precision-recall curve (not average precision).
        pr_auc = auc(curve_recall, curve_precision)
        results.append({
            'model': name,
            'dataset': split,
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions, zero_division=0),
            'f1': f1_score(y, predictions, zero_division=0),
            'pr_auc': pr_auc,
        })
        if split == 'Test':
            probability_path = (
                OUTPUT_DIR
                / f"traditional_model_{name.lower().replace(' ', '_')}_test_probabilities.csv"
            )
            pd.DataFrame({
                'trans_num': test_df['trans_num'].to_numpy(),
                'amt': test_df['amt'].to_numpy(),
                'is_fraud': y_test.to_numpy(),
                'fraud_probability': probabilities,
            }).to_csv(probability_path, index=False)
            print(f'Saved test probabilities: {probability_path}', flush=True)
            ax.plot(curve_recall, curve_precision,
                    label=f'{name} (PR AUC = {pr_auc:.4f})')

metrics = pd.DataFrame(results)
print('\nMetrics (fraud = positive class; decision threshold = 0.5):', flush=True)
print(metrics.to_string(index=False, float_format=lambda value: f'{value:.6f}'), flush=True)
script_dir = Path(__file__).resolve().parent
metrics_path = OUTPUT_DIR / 'traditional_model_metrics.csv'
metrics = save_metric_results(metrics, OUTPUT_DIR, list(models))

prevalence = float(y_test.mean())
ax.axhline(prevalence, linestyle='--', color='gray',
           label=f'Chance (fraud prevalence = {prevalence:.4f})')
ax.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02))
ax.set_title(
    'Test precision-recall curves: November 2019 - January 2020\n'
    f'Test samples: {len(y_test):,} '
    f'({int((y_test == 0).sum()):,} legitimate; {int((y_test == 1).sum()):,} fraud)'
)
ax.legend(loc='upper right')
fig.tight_layout()
plot_path = OUTPUT_DIR / 'traditional_model_pr_auc.png'
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f'\nSaved metrics: {metrics_path}\nSaved precision-recall plot: {plot_path}', flush=True)
