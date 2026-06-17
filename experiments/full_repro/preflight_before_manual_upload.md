# 中文阅读说明：preflight_before_manual_upload.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Preflight Before Manual Upload

| Check | OK | Detail |
|---|---:|---|
| git_commit | True | `762a483` |
| git_worktree_clean | False | `M experiments/full_repro/pytorch_official/dryrun/status.json
 M experiments/full_repro/pytorch_official/dryrun/stderr.log
 M experiments/full_repro/pytorch_official/dryrun/stdout.log
 M experiments/full_repro/pytorch_official/env_matrix/env_check_relaxed-pypi.json
 M experiments/full_repro/pytorch_official/env_matrix/env_check_relaxed-pypi.md
 M experiments/full_repro/pytorch_official/env_matrix/env_check_relaxed-pypi.stdout.log
 M experiments/full_repro/pytorch_official/env_matrix/env_check_strict-cu118.json
 M experiments/full_repro/pytorch_official/env_matrix/env_check_strict-cu118.md
 M experiments/full_repro/pytorch_official/env_matrix/env_check_strict-cu118.stdout.log
 M experiments/full_repro/pytorch_official/env_matrix/env_check_strict-pypi.json
 M experiments/full_repro/pytorch_official/env_matrix/env_check_strict-pypi.md
 M experiments/full_repro/pytorch_official/env_matrix/env_check_strict-pypi.stdout.log
 M experiments/full_repro/pytorch_official/env_matrix/env_matrix_summary.json
 M experiments/full_repro/pytorch_official/env_matrix/env_matrix_summary.md
 M experiments/full_repro/pytorch_official/env_matrix/import_smoke_relaxed-pypi.log
 M experiments/full_repro/pytorch_official/env_matrix/import_smoke_strict-cu118.log
 M experiments/full_repro/pytorch_official/env_matrix/import_smoke_strict-cu118.status
 M experiments/full_repro/pytorch_official/env_matrix/import_smoke_strict-pypi.log
 M experiments/full_repro/pytorch_official/env_matrix/import_smoke_strict-pypi.status
 M experiments/full_repro/pytorch_official/env_matrix/install_relaxed-pypi.log
 M experiments/full_repro/pytorch_official/env_matrix/install_relaxed-pypi.status
 M experiments/full_repro/pytorch_official/env_matrix/install_strict-cu118.log
 M experiments/full_repro/pytorch_official/env_matrix/install_strict-cu118.status
 M experiments/full_repro/pytorch_official/env_matrix/install_strict-pypi.log
 M experiments/full_repro/pytorch_official/env_matrix/install_strict-pypi.status
 M experiments/full_repro/pytorch_official/env_matrix/network_probe.json
 M experiments/full_repro/pytorch_official/env_matrix/official_repo_imports.json
 M experiments/full_repro/pytorch_official/env_matrix/official_repo_imports.md
 M experiments/full_repro/pytorch_official/env_matrix/selected_env.sh
 M scripts/check_official_pytorch_env.py
 M scripts/install_official_env_matrix.sh
?? experiments/full_repro/pytorch_official/env/env_check_strict-cu118.json
?? experiments/full_repro/pytorch_official/env/env_check_strict-cu118.md
?? experiments/full_repro/pytorch_official/env/env_check_strict-cu118.stdout.log
?? experiments/full_repro/pytorch_official/env_matrix/matrix_runner.out
?? experiments/full_repro/pytorch_official/env_matrix/matrix_runner.pid` |
| official_venv_exists | True | `/mnt/data/sj/venvs/tadsr_official_pytorch` |
| official_env_installed | False | `env_check_strict.json PASS` |
| wheelhouse_exists | False | `/mnt/data/sj/wheelhouse/tadsr_official_pytorch` |
| incoming_dir_exists | True | `/mnt/data/sj/incoming` |
| checkpoints_dir_exists | True | `/mnt/data/sj/checkpoints/TADSR/preset/weights` |
| datasets_dir_exists | True | `/mnt/data/sj/datasets/TADSR` |
| run_after_assets_ready_exists | True | `scripts/run_after_assets_ready.sh` |
| verify_official_assets_exists | True | `scripts/verify_official_assets.py` |
| final_audit_runs | True | `python3 scripts/final_audit.py` |

## NEXT USER ACTION

1. Download zty557/TADSR weights on a networked machine.
2. Put them into `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights/`.
3. If official environment is still missing, prepare wheelhouse in `/mnt/data/sj/wheelhouse/tadsr_official_pytorch/`.
4. Run:

```bash
cd /mnt/data/sj/projects/TADSR-Jittor
bash scripts/run_after_assets_ready.sh
```
