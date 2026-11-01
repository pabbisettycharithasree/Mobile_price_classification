# File placement map

| Reference area | Mobile Price overlay | Purpose |
|---|---|---|
| `api/` | `api/` | FastAPI, schemas, loader, preprocessing, request logging |
| `deployment/` | `deployment/` | Deployment copy of existing model artifacts |
| `k8s/` | `k8s/` | Kubernetes manifests |
| `lab11_commands/` | `lab11_commands/` | Docker/Kubernetes commands |
| `src/mlops_pipeline/11_*` | same | Lab 11 artifact preparation |
| `src/mlops_pipeline/12_*` | same | Lab 12 drift/performance monitoring scripts |
| `src/mlops_pipeline/13_*` | same | Lab 13 integration runner |
| `tests/` | `tests/` | API, container and performance smoke tests |
| `.github/workflows/` | same | Lab 13 runner and monitoring workflows |

No Lab 1–9 files are included for replacement.
