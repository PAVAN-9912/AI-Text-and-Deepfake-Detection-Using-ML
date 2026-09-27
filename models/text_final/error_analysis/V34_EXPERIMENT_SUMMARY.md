# V3.4 Experiment Summary

## Final Validation Performance (N=998)

- **Accuracy**: 89.88%
- **Precision**: 88.98%
- **Recall**: 90.27%
- **F1-Score**: 89.62%
- **ROC-AUC**: 95.63%

- **True Negatives**: 461
- **False Positives**: 54 (Rate: 10.49%)
- **False Negatives**: 47 (Rate: 9.73%)
- **True Positives**: 436

## Length-Stratified Thresholds

- `very_short`: 0.49
- `short`: 0.53
- `medium`: 0.64
- `long`: 0.59

## Performance by Length Group

length_group  samples  threshold  accuracy  precision  recall     f1  false_positive  false_negative  FP_rate  FN_rate
  very_short      240       0.49    0.8833     0.8417  0.9182 0.8783              19               9   0.1462   0.0818
       short      435       0.53    0.8851     0.8786  0.8786 0.8786              25              25   0.1092   0.1214
      medium      223       0.64    0.9193     0.9231  0.9231 0.9231               9               9   0.0849   0.0769
        long      100       0.59    0.9500     0.9787  0.9200 0.9485               1               4   0.0200   0.0800

