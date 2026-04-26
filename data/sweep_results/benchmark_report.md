# Benchmark Reproducibility Report

Generated: 2026-04-21 20:21:23

## Summary Statistics

| Agent | Mean | Std | 95% CI | n |
|-------|------|-----|--------|---|
| HCAPO | 0.2215 | 0.0127 | [0.2100, 0.2303] | 5 |
| MIRA | 0.2094 | 0.0095 | [0.2023, 0.2165] | 5 |
| KLONG | 0.2152 | 0.0222 | [0.1977, 0.2286] | 5 |
| MEMEX | 0.2148 | 0.0270 | [0.1943, 0.2352] | 5 |

## Pairwise Comparisons

| Comparison | Mean Diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| hcapo vs mira | 0.0121 | 0.1823 | 1.076 | No |
| hcapo vs klong | 0.0063 | 0.3849 | 0.348 | No |
| hcapo vs memex | 0.0067 | 0.6370 | 0.319 | No |
| mira vs klong | -0.0058 | 0.6674 | -0.340 | No |
| mira vs memex | -0.0053 | 0.6656 | -0.264 | No |
| klong vs memex | 0.0004 | 0.9756 | 0.018 | No |

## Integration Tests

- **easy_bench**: 10/10 checks passed
- **medium_bench**: 10/10 checks passed
- **hard_bench**: 10/10 checks passed
