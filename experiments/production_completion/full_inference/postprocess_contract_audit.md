# Official TADSR postprocess/output contract audit

`TADSR_POSTPROCESS_CONTRACT_AUDIT: PASS`
`TADSR_OUTPUT_IMAGE_SAVE_POLICY_AUDIT: PASS`
`TADSR_POSTPROCESS_NOT_EXECUTED_GUARD: PASS`

## Contract

- decoded tensor range: TADSR_test.forward clamps decoded tensor to [-1, 1].
- tensor-to-image scaling: `output_image[0].cpu() * 0.5 + 0.5 followed by transforms.ToPILImage()`
- channel order: PIL RGB input is used; torchvision ToPILImage handles tensor channel order C,H,W to PIL image.
- color fix: optional adain_color_fix or wavelet_color_fix after ToPILImage.
- save policy: output_pil.save(os.path.join(args.output_dir, bname))

## Current project status

- Postprocess contract is documented.
- Image save was not executed.
- No PNG/JPG/MP4 model output was generated.
- This does not upgrade `JITTOR_FULL_INFERENCE`.
