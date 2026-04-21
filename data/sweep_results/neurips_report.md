# NeurIPS Reproducibility Report

Generated: 2026-04-21 15:01:36

## Summary Statistics

| Agent | Mean | Std | 95% CI | n |
|-------|------|-----|--------|---|
| HCAPO | 0.2344 | 0.0176 | [0.2240, 0.2547] | 3 |
| MIRA | 0.2214 | 0.0193 | [0.1996, 0.2365] | 3 |
| KLONG | 0.2116 | 0.0135 | [0.1961, 0.2207] | 3 |
| MEMEX | 0.2258 | 0.0275 | [0.2022, 0.2560] | 3 |

## Pairwise Comparisons

| Comparison | Mean Diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| hcapo vs mira | 0.0130 | 0.5382 | 0.705 | No |
| hcapo vs klong | 0.0228 | 0.0075 | 1.455 | Yes*** |
| hcapo vs memex | 0.0086 | 0.2189 | 0.375 | No |
| mira vs klong | 0.0098 | 0.5813 | 0.587 | No |
| mira vs memex | -0.0044 | 0.8712 | -0.184 | No |
| klong vs memex | -0.0142 | 0.1843 | -0.654 | No |

## Integration Tests

- **easy_bench**: 10/10 checks passed
- **medium_bench**: 10/10 checks passed
- **hard_bench**: 10/10 checks passed
