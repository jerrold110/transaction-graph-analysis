"""Decision framework for the saved test-set fraud probabilities.

Inputs in 1_traditional_model_results beside this script (columns: trans_num, amt, is_fraud, fraud_probability):
    traditional_model_logistic_regression_test_probabilities.csv
    traditional_model_random_forest_test_probabilities.csv
    traditional_model_xgboost_test_probabilities.csv

fraud_probability is the predicted probability of is_fraud = 1.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_FILES = {
    'Logistic regression': 'traditional_model_logistic_regression_test_probabilities.csv',
    'Random forest': 'traditional_model_random_forest_test_probabilities.csv',
    'XGBoost': 'traditional_model_xgboost_test_probabilities.csv',
}
THRESHOLDS = np.arange(2, 10) / 10


def load_predictions(folder):
    """Require identical transactions, amounts and labels across algorithms."""
    datasets = {}
    reference = None
    for model, filename in MODEL_FILES.items():
        data = pd.read_csv(folder / filename)
        required = ['trans_num', 'amt', 'is_fraud', 'fraud_probability']
        if not set(required).issubset(data.columns):
            raise ValueError(f'{filename}: missing required columns')
        if data.empty or data[required].isna().any().any():
            raise ValueError(f'{filename}: empty data or missing values')
        if not data['trans_num'].is_unique:
            raise ValueError(f'{filename}: duplicate transaction identifiers')
        if not data['is_fraud'].isin([0, 1]).all():
            raise ValueError(f'{filename}: invalid labels')
        if not data['fraud_probability'].between(0, 1).all():
            raise ValueError(f'{filename}: invalid probabilities')
        if not np.isfinite(data['amt']).all() or (data['amt'] < 0).any():
            raise ValueError(f'{filename}: invalid transaction values')
        data = data.set_index('trans_num').sort_index()
        identity = data[['amt', 'is_fraud']]
        if reference is not None:
            pd.testing.assert_frame_equal(reference, identity)
        reference = identity
        datasets[model] = data
    return datasets


def evaluate_decisions(data, flag, fraud_cost, legitimate_cost):
    """Realized cost uses true labels, never probabilities as labels."""
    fraud = data['is_fraud'].to_numpy() == 1
    amounts = data['amt'].to_numpy()
    flag = np.asarray(flag, dtype=bool)
    tp = int(np.count_nonzero(flag & fraud))
    fp = int(np.count_nonzero(flag & ~fraud))
    fn = int(np.count_nonzero(~flag & fraud))
    tn = int(np.count_nonzero(~flag & ~fraud))
    investigation_cost = float(tp * fraud_cost + fp * legitimate_cost)
    missed_fraud_cost = float(amounts[~flag & fraud].sum())
    return {
        'total_cost': investigation_cost + missed_fraud_cost,
        'investigation_cost': investigation_cost,
        'missed_fraud_cost': missed_fraud_cost,
        'flagged_count': tp + fp,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'true_positive_cost': float(tp * fraud_cost),
        'false_positive_cost': float(fp * legitimate_cost),
        'true_negative_cost': 0.0,
        'false_negative_cost': missed_fraud_cost,
    }


def run_scenario(datasets, scenario, fraud_cost, legitimate_cost):
    threshold_rows, bayes_rows, best_rows = [], [], []
    for model, data in datasets.items():
        p = data['fraud_probability'].to_numpy()
        context = {
            'scenario': scenario, 'model': model,
            'flag_fraud_cost': fraud_cost,
            'flag_legitimate_cost': legitimate_cost,
            'test_samples': len(data),
        }
        rows = []
        for threshold in THRESHOLDS:
            rows.append({
                **context, 'framework': 'Threshold optimisation',
                'threshold': float(threshold),
                **evaluate_decisions(data, p >= threshold, fraud_cost, legitimate_cost),
            })
        threshold_rows.extend(rows)
        # Break equal-cost ties by choosing the lower threshold.
        best_rows.append(min(rows, key=lambda r: (r['total_cost'], r['threshold'])))
        risk_allow = data['amt'].to_numpy() * p
        risk_flag = fraud_cost * p + legitimate_cost * (1 - p)
        bayes_row = {
            **context, 'framework': 'Bayes minimum risk', 'threshold': np.nan,
            **evaluate_decisions(data, risk_flag <= risk_allow, fraud_cost, legitimate_cost),
        }
        bayes_rows.append(bayes_row)
        best_rows.append(bayes_row)
    return pd.DataFrame(threshold_rows), pd.DataFrame(bayes_rows), pd.DataFrame(best_rows)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_thresholds(results, folder):
    fig, ax = plt.subplots(figsize=(9, 5))
    for model in MODEL_FILES:
        rows = results.loc[results['model'] == model]
        ax.plot(rows['threshold'], rows['total_cost'], marker='o', label=model)
        best = rows.sort_values(['total_cost', 'threshold']).iloc[0]
        ax.scatter([best['threshold']], [best['total_cost']], marker='*', s=160,
                   label=f"{model}: selected threshold {best['threshold']:.1f}")
    ax.set(title='Baseline threshold optimisation (flag costs: fraud 2.5, legitimate 3.5)',
           xlabel='Fraud probability threshold', ylabel='Total realised cost (Euros)', xticks=THRESHOLDS)
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, folder / 'baseline_TO_total_cost.png')


def plot_comparison(results, folder):
    """Compare only the lowest-cost model from each decision framework."""
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f"{r['model']}\n{r['framework']}" +
              (f"\nSelected threshold: {r['threshold']:.1f}" if pd.notna(r['threshold']) else '')
              for _, r in results.iterrows()]
    bars = ax.bar(labels, results['total_cost'], color=['tab:blue', 'tab:orange'])
    ax.bar_label(bars, labels=[f'{cost:,.2f}' for cost in results['total_cost']], padding=3)
    ax.set_ylabel('Total realised cost (Euros)')
    ax.set_title('Lowest-cost threshold optimisation vs lowest-cost Bayes minimum risk')
    ax.set_ylim(0, results['total_cost'].max() * 1.15)
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    save_figure(fig, folder / 'baseline_TO_BMR_comparison.png')


def save_category_breakdown(results, folder):
    rows = []
    for _, result in results.iterrows():
        for category, prefix in [('TP', 'true_positive'), ('FP', 'false_positive'),
                                 ('TN', 'true_negative'), ('FN', 'false_negative')]:
            count_column = prefix + 's'
            rows.append({
                'model': result['model'], 'framework': result['framework'],
                'selected_threshold': result['threshold'], 'category': category,
                'count': result[count_column],
                'realised_cost': result[f'{prefix}_cost'],
            })
    pd.DataFrame(rows).to_csv(folder / 'baseline_TO_BMR_category_breakdown.csv', index=False)
    labels = [f"{r['model']}\n{r['framework']}" +
              (f"\nSelected threshold: {r['threshold']:.1f}" if pd.notna(r['threshold']) else '')
              for _, r in results.iterrows()]
    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(results))
    width = 0.2
    for offset, (category, prefix) in enumerate([
        ('TP', 'true_positive'), ('FP', 'false_positive'),
        ('TN', 'true_negative'), ('FN', 'false_negative'),
    ]):
        bars = ax.bar(x + (offset - 1.5) * width,
                      results[prefix + 's'], width, label=category)
        ax.bar_label(bars, labels=[f'{cost:,.2f}' for cost in results[prefix + '_cost']],
                     padding=4, fontsize=8, rotation=90)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_yscale('log')
    ax.set_ylabel('Transaction count (log scale)')
    ax.set_title('Transaction counts by prediction category\nNumbers above bars are total realised costs')
    ax.set_ylim(1, results[['true_positives', 'false_positives',
                           'true_negatives', 'false_negatives']].to_numpy().max() * 5)
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(title='Prediction category')
    save_figure(fig, folder / 'baseline_TO_BMR_category_breakdown.png')


def run_cost_sensitivity_analysis(datasets, winners, folder):
    """Analyse cost sensitivity for baseline winners; keep the selected threshold fixed.

    Legitimate flag costs are increments above fraud flag costs, giving
    four fraud costs times three increments = twelve combinations.
    Bayes decisions adapt to each new cost matrix.
    """
    rows = []
    for fraud_cost in [1, 2.5, 3.5, 5]:
        for increment in [1, 2, 4]:
            legitimate_cost = fraud_cost + increment
            for _, winner in winners.iterrows():
                data = datasets[winner['model']]
                p = data['fraud_probability'].to_numpy()
                if winner['framework'] == 'Threshold optimisation':
                    flag = p >= winner['threshold']
                else:
                    flag = fraud_cost * p + legitimate_cost * (1 - p) <= data['amt'].to_numpy() * p
                rows.append({
                    'model': winner['model'], 'framework': winner['framework'],
                    'threshold': winner['threshold'], 'flag_fraud_cost': fraud_cost,
                    'flag_legitimate_cost': legitimate_cost,
                    'legitimate_cost_increment': increment, 'test_samples': len(data),
                    **evaluate_decisions(data, flag, fraud_cost, legitimate_cost),
                })
    results = pd.DataFrame(rows)
    results.to_csv(folder / 'sensitivity_TO_BMR_results.csv', index=False)
    summary = results.pivot(index=['flag_fraud_cost', 'flag_legitimate_cost'],
                            columns=['model', 'framework'], values='total_cost')
    summary.to_csv(folder / 'sensitivity_TO_BMR_cost_comparison.csv')
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(summary))
    for j, column in enumerate(summary.columns):
        winner = winners.loc[winners['framework'] == column[1]].iloc[0]
        label = f'{column[0]} — {column[1]}'
        if pd.notna(winner['threshold']):
            label += f" (fixed threshold {winner['threshold']:.1f})"
        bars = ax.bar(x + (j - 0.5) * 0.38, summary[column], 0.38, label=label)
        ax.bar_label(bars, labels=[f'{v:,.2f}' for v in summary[column]],
                     rotation=90, padding=3, fontsize=8)
    ax.set_xticks(x, [f'{f:g} / {l:g}' for f, l in summary.index])
    ax.set_xlabel('Flag and investigate costs: fraud / legitimate')
    ax.set_ylabel('Total realised cost (Euros)')
    ax.set_title('Cost sensitivity analysis of the two baseline winners\n'
                 'Legitimate transaction flag cost = Fraud transaction flag cost + 1, 2, or 4 euros')
    ax.set_ylim(0, summary.to_numpy().max() * 1.3)
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend()
    save_figure(fig, folder / 'sensitivity_TO_BMR_total_cost.png')
    print('\nCost sensitivity analysis (12 combinations; baseline algorithms and threshold retained):')
    print(summary.to_string())


def main():
    folder = Path(__file__).resolve().parent
    datasets = load_predictions(folder / '1_traditional_model_results')
    output = folder / '2_decision_framework_results'
    output.mkdir(exist_ok=True)
    thresholds, bayes, baseline = run_scenario(datasets, 'baseline', 2.5, 3.5)
    thresholds.to_csv(output / 'baseline_TO_threshold_search.csv', index=False)
    bayes.to_csv(output / 'baseline_BMR_results.csv', index=False)
    winners = baseline.loc[baseline.groupby('framework', sort=False)['total_cost'].idxmin()]
    winners.to_csv(output / 'baseline_TO_BMR_comparison.csv', index=False)
    winners.to_csv(output / 'baseline_TO_BMR_winners.csv', index=False)
    plot_thresholds(thresholds, output)
    plot_comparison(winners, output)
    save_category_breakdown(winners, output)
    run_cost_sensitivity_analysis(datasets, winners, output)
    print('\nLowest-cost algorithm for each baseline framework:')
    print(winners[['model', 'framework', 'threshold', 'total_cost']].to_string(index=False))
    print(f'\nSaved tables and plots to {output}')
    print('Thresholds were selected on test labels: exploratory results, not an unbiased holdout evaluation.')


if __name__ == '__main__':
    main()
