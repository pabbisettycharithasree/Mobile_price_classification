# Model Evaluation Summary

**Best model:** `svc`  |  **CV accuracy:** 0.9594

| Metric | Test value |
|---|---|
| Accuracy | 0.97 |
| Precision (macro) | 0.9703 |
| Recall (macro) | 0.97 |
| F1 (macro) | 0.97 |

## Classification report
```
               precision    recall  f1-score   support

      Low (0)       0.98      0.99      0.99       100
   Medium (1)       0.98      0.94      0.96       100
     High (2)       0.94      0.97      0.96       100
Very High (3)       0.98      0.98      0.98       100

     accuracy                           0.97       400
    macro avg       0.97      0.97      0.97       400
 weighted avg       0.97      0.97      0.97       400

```

## Top 5 features
- `ram`: 0.6933
- `battery_power`: 0.2197
- `px_height`: 0.1245
- `px_width`: 0.1172
- `px_area`: 0.0145

Figures: `outputs/confusion_matrix.png`, `outputs/feature_importance.png`
