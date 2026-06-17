# Experimental CLI metadata-only plan

`TADSR_EXPERIMENTAL_CLI_METADATA_ONLY_PLAN: PASS`

本文件记录未来可选的 experimental metadata-only CLI 计划。本轮没有实现该 CLI，没有打开 production full inference，也没有生成图片或视频。

未来建议命令：

```bash
python -m jittor_tadsr_full.tadsr_full_experimental --metadata-only --num-inference-steps 1 --disable-image-save
```

当前 production guard 必须保持：

```bash
python -m jittor_tadsr_full.tadsr_full --full-inference
```

该命令仍应抛出 `NotImplementedError`。
