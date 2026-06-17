#!/usr/bin/env python3
from __future__ import annotations

from unet_downblock2_resnet1_common import add_metrics, blocked_result, load_oracle, resnet1_tester, status_from_metrics, write_report


def main() -> int:
    tensors, blocked = load_oracle()
    name = "jittor_unet_downblock2_resnet1_synthetic_alignment"
    title = "TADSR UNet down_blocks.2.resnets.1 Synthetic Alignment"
    if blocked:
        result = blocked_result(name, title, blocked)
        print("TADSR_UNET_DOWNBLOCK2_RESNET1_SYNTHETIC_ALIGNMENT:", result["status"])
        return 0 if result["status"] == "PASS" else 1
    tester = resnet1_tester()
    got = tester.run(tensors["synthetic_downblock2_resnet1_input"], tensors["synthetic_downblock2_resnet1_temb"], return_intermediates=False)
    diag = {"output": add_metrics(got, tensors["synthetic_downblock2_resnet1_output"], tolerance=2e-4)}
    status = status_from_metrics(diag["output"])
    write_report(name, title, {"status": status, "diagnostics": diag, "note": "Synthetic isolated hidden_states + temb -> down_blocks.2.resnets.1 only."})
    print("TADSR_UNET_DOWNBLOCK2_RESNET1_SYNTHETIC_ALIGNMENT:", status)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
