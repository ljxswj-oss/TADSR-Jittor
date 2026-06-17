# Final Audit Report

```text
FINAL AUDIT SUMMARY
JITTOR_TINY_TRAINING: PASS - tiny train log/loss curve/checkpoint present
JITTOR_TINY_TESTING: PASS - tiny test metrics and compare grid present
PYTORCH_TINY_ALIGNMENT: PASS - alignment report and compare curves present
PYTORCH_OFFICIAL_VENV: BLOCKED - \mnt\data\sj\venvs\tadsr_official_pytorch
PYTORCH_OFFICIAL_ENV_MATRIX: PASS - at least one env fully passed
PYTORCH_OFFICIAL_SELECTED_ENV: BLOCKED - selected_env.sh missing or points to no usable venv
PYTORCH_OFFICIAL_ENV_STRICT_CU118: PASS - strict-cu118 passed
PYTORCH_OFFICIAL_ENV_STRICT_PYPI: PASS - strict-pypi passed
PYTORCH_OFFICIAL_ENV_RELAXED_PYPI: PARTIAL - version mismatches: torch, torchvision
PYTORCH_OFFICIAL_REPO_IMPORTS: PASS - all official repo imports passed
PYTORCH_OFFICIAL_DRYRUN: PASS - test_tadsr.py CLI checked
PYTORCH_OFFICIAL_ASSETS: BLOCKED - missing: time_vae, unet, vae, text_encoder, tokenizer, scheduler, feature_extractor, bert-base-uncased, DAPE.pth, ram_swin_large_14m.pth, tadsr.pkl
OFFICIAL_PYTORCH_ORACLE: PASS - outputs=4 inputs=4
PYTORCH_OFFICIAL_BENCHMARK: BLOCKED_DATASET_MISSING - missing benchmark datasets: RealSR, DRealSR, RealLR200
WHEELHOUSE: BLOCKED - 0 files in \mnt\data\sj\wheelhouse\tadsr_official_pytorch
OFFLINE_PACK_GUIDE: PASS - offline_pack guide exists
JITTOR_GPU: BLOCKED - cuDNN8 required; cuDNN9 not acceptable
JITTOR_BASE_WEIGHT_CONVERSION: PASS - tadsr/DAPE/RAM conversion report
JITTOR_DIFFUSERS_WEIGHT_CONVERSION: PASS - diffusers NPZ conversion verification
PYTORCH_ORACLE_TENSORS: PASS - preprocess/scheduler oracle tensors exported
TADSR_SCHEDULER_USAGE_AUDIT: PASS - official TADSR scheduler usage path audited without running full inference
TADSR_SCHEDULER_CONFIG_AUDIT: PASS - official scheduler class/config/tensor contract audited
TADSR_SCHEDULER_TIMESTEP_CONTRACT_AUDIT: PASS - official set_timesteps(1) contract audited
TADSR_SCHEDULER_STEP_CONTRACT_AUDIT: PASS - official scheduler.step one-step contract audited for boundary testing only
TADSR_SCHEDULER_ORACLE_TENSORS: PASS - official scheduler-only one-step oracle tensors exported; no denoising loop or VAE decode
TADSR_UNET_SCHEDULER_ONE_STEP_ORACLE: PASS - optional one-step scheduler oracle using existing UNet full-forward tensors
TADSR_SCHEDULER_TIMESTEPS_ALIGNMENT: PASS - Jittor scheduler boundary tester matches official timesteps
TADSR_SCHEDULER_SCALE_MODEL_INPUT_ALIGNMENT: NOT_APPLICABLE_NOOP - scale_model_input is aligned or explicitly no-op for the audited DDPMScheduler path
TADSR_SCHEDULER_STEP_ALIGNMENT: PASS - Jittor one-step scheduler boundary output matches official oracle
TADSR_UNET_SCHEDULER_ONE_STEP_ALIGNMENT: PASS - optional existing-UNet-output plus scheduler one-step boundary matches official oracle
TADSR_SCHEDULER_BOUNDARY_ALIGNMENT: PASS - scheduler boundary / minimal denoising-step contract is audited and aligned; full loop still unopened
TADSR_MINIMAL_INTEGRATION_AUDIT: PASS - official minimal TADSR latent one-step path audited from source code; no production full inference
TADSR_GET_X0_FROM_RES_AUDIT: PASS - official get_x0_from_res formula audited: latent / sqrt(alpha_prod_t) - model_pred
TADSR_MINIMAL_LATENT_ONE_STEP_CONTRACT_AUDIT: PASS - minimal latent-only encode -> UNet -> x0 contract defined without full loop
TADSR_MINIMAL_LATENT_ONE_STEP_ORACLE: PASS - official minimal latent-only oracle exported; no VAE decode or image output
TADSR_GET_X0_FROM_RES_ORACLE: PASS - official get_x0_from_res oracle tensor exported
TADSR_MINIMAL_DECODE_BOUNDARY_ORACLE: PASS - official minimal one-step decode/clamp tensor oracle exported; no image output or full inference
TADSR_MINIMAL_VAE_ENCODE_ALIGNMENT: PASS - Jittor actual TimeVAE encode/sample/scale boundary matches minimal oracle
TADSR_MINIMAL_UNET_MODEL_PRED_ALIGNMENT: PASS - Jittor UNet full-forward model_pred matches minimal oracle
TADSR_GET_X0_FROM_RES_ALIGNMENT: PASS - Jittor alpha/get_x0_from_res output matches official oracle
TADSR_MINIMAL_LATENT_ONE_STEP_ALIGNMENT: PASS - minimal latent one-step outputs match official oracle
TADSR_MINIMAL_DECODE_INPUT_ALIGNMENT: PASS - Jittor decode_input tensor matches official x0/scaling_factor oracle
TADSR_MINIMAL_DECODED_OUTPUT_ALIGNMENT: PASS - Jittor TimeVAE actual decoder original_forward output matches official tensor oracle
TADSR_MINIMAL_FINAL_CLAMPED_OUTPUT_ALIGNMENT: PASS - Jittor decoded output clamped to [-1, 1] matches official tensor oracle
TADSR_MINIMAL_DECODE_BOUNDARY_ALIGNMENT: PASS - minimal one-step decode boundary is PASS only when decode_input, decoded_output and final_clamped_output all align
TADSR_MINIMAL_LATENT_INTEGRATION_DRY_RUN: PASS - minimal latent-only dry-run combines VAE encode, UNet model_pred and get_x0_from_res without opening full inference
TADSR_MINIMAL_ONE_STEP_DECODE_DRY_RUN: PASS - minimal one-step dry-run includes tensor-only decode/clamp boundary while still avoiding full inference
JITTOR_PREPROCESS_ALIGNMENT: PASS - preprocess tensor alignment
JITTOR_SCHEDULER_ALIGNMENT: PASS - scheduler tensor alignment
JITTOR_WEIGHT_LOADING_ALIGNMENT: PASS - converted NPZ loading check
JITTOR_LORA_MAPPING: PARTIAL - LoRA key table; forward not claimed
JITTOR_TIME_VAE_LOADING: PARTIAL - time-VAE loading checklist; forward not claimed
OFFICIAL_TIME_VAE_AUDIT: PASS - official TimeAwareAutoencoderKL module tree and state-key audit
TIME_VAE_ORACLE_TENSORS: PASS - PyTorch time-VAE block oracle tensors exported
TIME_VAE_CONV_IN_ALIGNMENT: PASS - Jittor-side conv_in output aligned with PyTorch oracle
TIME_VAE_FIRST_BLOCK_ALIGNMENT: PASS - first TimeAware ResnetBlock2D numerical alignment
LORA_PAIR_ANALYSIS: PASS - LoRA A/B key-pair discovery
LORA_MERGE_VALIDATION: PASS - at least one LoRA pair merge formula numerically aligned
TIME_VAE_LORA_ALIGNMENT_REPORT: PASS - TimeAware VAE and LoRA progress report
TIME_VAE_DOWNBLOCK0_AUDIT: PASS - official encoder.down_blocks.0 structure and key mapping audit
TIME_VAE_RESNET1_ALIGNMENT: PASS - encoder.down_blocks.0.resnets.1 numerical alignment
TIME_VAE_DOWNSAMPLER_ALIGNMENT: PASS - encoder.down_blocks.0.downsamplers.0 numerical alignment or NOT_APPLICABLE
TIME_VAE_DOWNBLOCK0_ALIGNMENT: PASS - full encoder.down_blocks.0 stage numerical alignment
TIME_VAE_DOWNBLOCK1_AUDIT: PASS - official encoder.down_blocks.1 structure and key mapping audit
TIME_VAE_DOWNBLOCK1_ORACLE_TENSORS: PASS - PyTorch oracle tensors for encoder.down_blocks.1 exported
TIME_VAE_DOWNBLOCK1_RESNET0_ALIGNMENT: PASS - encoder.down_blocks.1.resnets.0 128->256 channel-change alignment
TIME_VAE_DOWNBLOCK1_RESNET1_ALIGNMENT: PASS - encoder.down_blocks.1.resnets.1 numerical alignment
TIME_VAE_DOWNBLOCK1_DOWNSAMPLER_ALIGNMENT: PASS - encoder.down_blocks.1.downsamplers.0 numerical alignment
TIME_VAE_DOWNBLOCK1_ALIGNMENT: PASS - full encoder.down_blocks.1 stage numerical alignment
TIME_VAE_ENCODER_STAGE01_ALIGNMENT: PASS - encoder.conv_in + down_blocks.0 + down_blocks.1 composition alignment
TIME_VAE_DOWNBLOCK2_AUDIT: PASS - official encoder.down_blocks.2 structure and key mapping audit
TIME_VAE_DOWNBLOCK2_ORACLE_TENSORS: PASS - PyTorch oracle tensors for encoder.down_blocks.2 exported
TIME_VAE_DOWNBLOCK2_RESNET0_ALIGNMENT: PASS - encoder.down_blocks.2.resnets.0 256->512 channel-change alignment
TIME_VAE_DOWNBLOCK2_RESNET1_ALIGNMENT: PASS - encoder.down_blocks.2.resnets.1 numerical alignment
TIME_VAE_DOWNBLOCK2_DOWNSAMPLER_ALIGNMENT: PASS - encoder.down_blocks.2.downsamplers.0 numerical alignment or NOT_APPLICABLE
TIME_VAE_DOWNBLOCK2_ALIGNMENT: PASS - full encoder.down_blocks.2 stage numerical alignment
TIME_VAE_ENCODER_STAGE012_ALIGNMENT: PASS - encoder.conv_in + down_blocks.0 + down_blocks.1 + down_blocks.2 composition alignment
TIME_VAE_DOWNBLOCK3_AUDIT: PASS - official encoder.down_blocks.3 structure and key mapping audit
TIME_VAE_DOWNBLOCK3_ORACLE_TENSORS: PASS - PyTorch oracle tensors for encoder.down_blocks.3 exported
TIME_VAE_DOWNBLOCK3_RESNET0_ALIGNMENT: PASS - encoder.down_blocks.3.resnets.0 numerical alignment
TIME_VAE_DOWNBLOCK3_RESNET1_ALIGNMENT: PASS - encoder.down_blocks.3.resnets.1 numerical alignment
TIME_VAE_DOWNBLOCK3_DOWNSAMPLER_ALIGNMENT: NOT_APPLICABLE - encoder.down_blocks.3 downsampler status; NOT_APPLICABLE if absent
TIME_VAE_DOWNBLOCK3_ALIGNMENT: PASS - full encoder.down_blocks.3 stage numerical alignment
TIME_VAE_ENCODER_STAGE0123_ALIGNMENT: PASS - encoder.conv_in + down_blocks.0 + down_blocks.1 + down_blocks.2 + down_blocks.3 composition alignment
TIME_VAE_MIDBLOCK_AUDIT: PASS - official encoder.mid_block structure and key mapping audit
TIME_VAE_MIDBLOCK_ORACLE_TENSORS: PASS - PyTorch oracle tensors for encoder.mid_block exported
TIME_VAE_MIDBLOCK_RESNET0_ALIGNMENT: PASS - encoder.mid_block.resnets.0 numerical alignment
TIME_VAE_MIDBLOCK_ATTENTION_ALIGNMENT: PASS - encoder.mid_block.attentions.0 numerical alignment or NOT_APPLICABLE
TIME_VAE_MIDBLOCK_RESNET1_ALIGNMENT: PASS - encoder.mid_block.resnets.1 numerical alignment or NOT_APPLICABLE
TIME_VAE_MIDBLOCK_ALIGNMENT: PASS - full encoder.mid_block stage numerical alignment
TIME_VAE_ENCODER_STAGE0123_MID_ALIGNMENT: PASS - encoder.conv_in + down_blocks.0..3 + mid_block composition alignment
TIME_VAE_ENCODER_TAIL_AUDIT: PASS - official encoder tail and quant_conv structure/key mapping audit
TIME_VAE_ENCODER_TAIL_ORACLE_TENSORS: PASS - PyTorch oracle tensors for encoder tail and quant_conv exported
TIME_VAE_ENCODER_TAIL_NORM_ALIGNMENT: PASS - encoder.conv_norm_out numerical alignment
TIME_VAE_ENCODER_TAIL_ACT_ALIGNMENT: PASS - encoder.conv_act numerical alignment
TIME_VAE_ENCODER_TAIL_CONV_OUT_ALIGNMENT: PASS - encoder.conv_out numerical alignment
TIME_VAE_ENCODER_TAIL_ALIGNMENT: PASS - full encoder tail numerical alignment
TIME_VAE_QUANT_CONV_ALIGNMENT: PASS - deterministic quant_conv moments tensor alignment
TIME_VAE_ENCODER_STAGE0123_MID_TAIL_ALIGNMENT: PASS - encoder.conv_in + down_blocks.0..3 + mid_block + tail composition alignment
TIME_VAE_ENCODER_STAGE0123_MID_TAIL_QUANT_ALIGNMENT: PASS - encoder-side deterministic path through quant_conv alignment
TIME_VAE_DECODER_ENTRY_AUDIT: PASS - official decoder entry, DGD split/mode and post_quant_conv audit
TIME_VAE_DECODER_ENTRY_ORACLE_TENSORS: PASS - PyTorch decoder-entry oracle tensors exported
TIME_VAE_MOMENTS_SPLIT_ALIGNMENT: PASS - posterior moments mean/logvar split and clamp alignment
TIME_VAE_POSTERIOR_MODE_ALIGNMENT: PASS - posterior mode(mean) alignment; no sampling
TIME_VAE_POST_QUANT_CONV_ALIGNMENT: PASS - post_quant_conv numerical alignment
TIME_VAE_DECODER_CONV_IN_ALIGNMENT: PASS - decoder.conv_in numerical alignment
TIME_VAE_DECODER_ENTRY_ALIGNMENT: PASS - post_quant_conv + decoder.conv_in synthetic latent path alignment
TIME_VAE_QUANT_TO_DECODER_ENTRY_ALIGNMENT: PASS - quant moments -> mode -> decoder entry alignment
TIME_VAE_ENCODER_TO_DECODER_ENTRY_ALIGNMENT: PASS - deterministic encoder-to-decoder-entry bridge alignment
TIME_VAE_DECODER_ENTRY_BRIDGE_ALIGNMENT: PASS - posterior mode + post_quant_conv + decoder.conv_in deterministic bridge; decoder body is checked in later rows
TIME_VAE_DECODER_MIDBLOCK_AUDIT: PASS - official decoder.mid_block topology and key audit
TIME_VAE_DECODER_MIDBLOCK_ORACLE_TENSORS: PASS - PyTorch decoder.mid_block oracle tensors exported
TIME_VAE_DECODER_MIDBLOCK_RESNET0_ALIGNMENT: PASS - decoder.mid_block.resnets.0 alignment
TIME_VAE_DECODER_MIDBLOCK_ATTENTION_ALIGNMENT: PASS - decoder.mid_block.attentions.0 alignment or NOT_APPLICABLE
TIME_VAE_DECODER_MIDBLOCK_RESNET1_ALIGNMENT: PASS - decoder.mid_block.resnets.1 alignment
TIME_VAE_DECODER_MIDBLOCK_ALIGNMENT: PASS - full decoder.mid_block deterministic alignment
TIME_VAE_DECODER_MIDBLOCK_SYNTHETIC_ALIGNMENT: PASS - isolated synthetic hidden -> decoder.mid_block alignment
TIME_VAE_DECODER_ENTRY_MIDBLOCK_ALIGNMENT: PASS - decoder entry -> decoder.mid_block bridge alignment
TIME_VAE_QUANT_TO_DECODER_MIDBLOCK_ALIGNMENT: PASS - quant moments -> posterior mode -> decoder.mid_block bridge alignment
TIME_VAE_ENCODER_TO_DECODER_MIDBLOCK_ALIGNMENT: PASS - encoder -> quant -> decoder.mid_block deterministic bridge alignment
TIME_VAE_DECODER_MIDBLOCK_BRIDGE_ALIGNMENT: PASS - deterministic decoder.mid_block aligned through isolated, decoder-entry and encoder-to-decoder paths; later decoder stages are checked separately
TIME_VAE_DECODER_UPBLOCKS_AUDIT: PASS - official decoder.up_blocks audit; up_blocks.0 focus plus later-block topology notes
TIME_VAE_DECODER_UPBLOCK0_AUDIT: PASS - official decoder.up_blocks.0 topology and key audit
TIME_VAE_DECODER_UPBLOCK0_ORACLE_TENSORS: PASS - PyTorch decoder.up_blocks.0 oracle tensors exported
TIME_VAE_DECODER_UPBLOCK0_RESNET0_ALIGNMENT: PASS - decoder.up_blocks.0.resnets.0 alignment
TIME_VAE_DECODER_UPBLOCK0_RESNET1_ALIGNMENT: PASS - decoder.up_blocks.0.resnets.1 alignment
TIME_VAE_DECODER_UPBLOCK0_RESNET2_ALIGNMENT: PASS - decoder.up_blocks.0.resnets.2 alignment
TIME_VAE_DECODER_UPBLOCK0_UPSAMPLER0_ALIGNMENT: PASS - decoder.up_blocks.0.upsamplers.0 nearest-2x + conv alignment
TIME_VAE_DECODER_UPBLOCK0_ALIGNMENT: PASS - full decoder.up_blocks.0 deterministic alignment
TIME_VAE_DECODER_UPBLOCK0_SYNTHETIC_ALIGNMENT: PASS - isolated synthetic hidden -> decoder.up_blocks.0 alignment
TIME_VAE_DECODER_MIDBLOCK_UPBLOCK0_ALIGNMENT: PASS - decoder.mid_block -> decoder.up_blocks.0 bridge alignment
TIME_VAE_DECODER_ENTRY_MIDBLOCK_UPBLOCK0_ALIGNMENT: PASS - decoder entry -> decoder.mid_block -> decoder.up_blocks.0 bridge alignment
TIME_VAE_QUANT_TO_DECODER_UPBLOCK0_ALIGNMENT: PASS - quant moments -> posterior mode -> decoder.up_blocks.0 bridge alignment
TIME_VAE_ENCODER_TO_DECODER_UPBLOCK0_ALIGNMENT: PASS - encoder -> quant -> decoder.up_blocks.0 deterministic bridge alignment
TIME_VAE_DECODER_UPBLOCK0_BRIDGE_ALIGNMENT: PASS - deterministic decoder.up_blocks.0 aligned through isolated, decoder-entry and encoder-to-decoder paths; later up blocks and tail are checked separately
TIME_VAE_DECODER_UPBLOCK1_AUDIT: PASS - official decoder.up_blocks.1 topology and key audit
TIME_VAE_DECODER_UPBLOCK1_ORACLE_TENSORS: PASS - PyTorch decoder.up_blocks.1 oracle tensors exported
TIME_VAE_DECODER_UPBLOCK1_RESNET0_ALIGNMENT: PASS - decoder.up_blocks.1.resnets.0 alignment
TIME_VAE_DECODER_UPBLOCK1_RESNET1_ALIGNMENT: PASS - decoder.up_blocks.1.resnets.1 alignment
TIME_VAE_DECODER_UPBLOCK1_RESNET2_ALIGNMENT: PASS - decoder.up_blocks.1.resnets.2 alignment
TIME_VAE_DECODER_UPBLOCK1_UPSAMPLER0_ALIGNMENT: PASS - decoder.up_blocks.1.upsamplers.0 nearest-2x + conv alignment
TIME_VAE_DECODER_UPBLOCK1_ALIGNMENT: PASS - full decoder.up_blocks.1 deterministic alignment
TIME_VAE_DECODER_UPBLOCK1_SYNTHETIC_ALIGNMENT: PASS - isolated synthetic hidden -> decoder.up_blocks.1 pressure alignment
TIME_VAE_DECODER_UPBLOCK0_UPBLOCK1_ALIGNMENT: PASS - decoder.up_blocks.0 -> decoder.up_blocks.1 bridge alignment
TIME_VAE_DECODER_ENTRY_MIDBLOCK_UPBLOCKS01_ALIGNMENT: PASS - decoder entry -> decoder.mid_block -> decoder.up_blocks.0 -> decoder.up_blocks.1 bridge alignment
TIME_VAE_QUANT_TO_DECODER_UPBLOCK1_ALIGNMENT: PASS - quant moments -> posterior mode -> decoder.up_blocks.1 bridge alignment
TIME_VAE_ENCODER_TO_DECODER_UPBLOCK1_ALIGNMENT: PASS - encoder -> quant -> decoder.up_blocks.1 deterministic bridge alignment
TIME_VAE_DECODER_UPBLOCK1_BRIDGE_ALIGNMENT: PASS - deterministic decoder.up_blocks.1 aligned through isolated, decoder-entry, quant-to-decoder and encoder-to-decoder paths; later up blocks and tail are checked separately
TIME_VAE_DECODER_UPBLOCK2_AUDIT: PASS - official decoder.up_blocks.2 topology and key audit
TIME_VAE_DECODER_UPBLOCK2_ORACLE_TENSORS: PASS - PyTorch decoder.up_blocks.2 oracle tensors exported
TIME_VAE_DECODER_UPBLOCK2_RESNET0_ALIGNMENT: PASS - decoder.up_blocks.2.resnets.0 512->256 shortcut alignment
TIME_VAE_DECODER_UPBLOCK2_RESNET1_ALIGNMENT: PASS - decoder.up_blocks.2.resnets.1 alignment
TIME_VAE_DECODER_UPBLOCK2_RESNET2_ALIGNMENT: PASS - decoder.up_blocks.2.resnets.2 alignment
TIME_VAE_DECODER_UPBLOCK2_UPSAMPLER0_ALIGNMENT: PASS - decoder.up_blocks.2.upsamplers.0 nearest-2x + conv alignment
TIME_VAE_DECODER_UPBLOCK2_ALIGNMENT: PASS - full decoder.up_blocks.2 deterministic alignment
TIME_VAE_DECODER_UPBLOCK2_SYNTHETIC_ALIGNMENT: PASS - isolated synthetic hidden -> decoder.up_blocks.2 pressure alignment
TIME_VAE_DECODER_UPBLOCKS012_ALIGNMENT: PASS - decoder.up_blocks.0 -> decoder.up_blocks.1 -> decoder.up_blocks.2 bridge alignment
TIME_VAE_DECODER_ENTRY_MIDBLOCK_UPBLOCKS012_ALIGNMENT: PASS - decoder entry -> decoder.mid_block -> decoder.up_blocks.0 -> decoder.up_blocks.1 -> decoder.up_blocks.2 bridge alignment
TIME_VAE_QUANT_TO_DECODER_UPBLOCK2_ALIGNMENT: PASS - quant moments -> posterior mode -> decoder.up_blocks.2 bridge alignment
TIME_VAE_ENCODER_TO_DECODER_UPBLOCK2_ALIGNMENT: PASS - encoder -> quant -> decoder.up_blocks.2 deterministic bridge alignment
TIME_VAE_DECODER_UPBLOCK2_BRIDGE_ALIGNMENT: PASS - deterministic decoder.up_blocks.2 aligned through isolated, decoder-entry, quant-to-decoder and encoder-to-decoder paths; decoder.up_blocks.3 and tail are checked separately
TIME_VAE_DECODER_UPBLOCK3_AUDIT: PASS - official decoder.up_blocks.3 topology and key audit
TIME_VAE_DECODER_UPBLOCK3_ORACLE_TENSORS: PASS - PyTorch decoder.up_blocks.3 oracle tensors exported
TIME_VAE_DECODER_UPBLOCK3_RESNET0_ALIGNMENT: PASS - decoder.up_blocks.3.resnets.0 256->128 shortcut alignment
TIME_VAE_DECODER_UPBLOCK3_RESNET1_ALIGNMENT: PASS - decoder.up_blocks.3.resnets.1 alignment
TIME_VAE_DECODER_UPBLOCK3_RESNET2_ALIGNMENT: PASS - decoder.up_blocks.3.resnets.2 alignment
TIME_VAE_DECODER_UPBLOCK3_UPSAMPLER0_ALIGNMENT: NOT_APPLICABLE - decoder.up_blocks.3 has no official upsampler; NOT_APPLICABLE is expected
TIME_VAE_DECODER_UPBLOCK3_ALIGNMENT: PASS - full decoder.up_blocks.3 deterministic alignment
TIME_VAE_DECODER_UPBLOCK3_SYNTHETIC_ALIGNMENT: PASS - isolated synthetic hidden -> decoder.up_blocks.3 pressure alignment
TIME_VAE_DECODER_UPBLOCKS0123_ALIGNMENT: PASS - decoder.up_blocks.0 -> decoder.up_blocks.1 -> decoder.up_blocks.2 -> decoder.up_blocks.3 bridge alignment
TIME_VAE_DECODER_ENTRY_MIDBLOCK_UPBLOCKS0123_ALIGNMENT: PASS - decoder entry -> decoder.mid_block -> decoder.up_blocks.0 -> decoder.up_blocks.1 -> decoder.up_blocks.2 -> decoder.up_blocks.3 bridge alignment
TIME_VAE_QUANT_TO_DECODER_UPBLOCK3_ALIGNMENT: PASS - quant moments -> posterior mode -> decoder.up_blocks.3 bridge alignment
TIME_VAE_ENCODER_TO_DECODER_UPBLOCK3_ALIGNMENT: PASS - encoder -> quant -> decoder.up_blocks.3 deterministic bridge alignment
TIME_VAE_DECODER_UPBLOCK3_BRIDGE_ALIGNMENT: PASS - deterministic decoder.up_blocks.3 aligned through isolated, decoder-entry, quant-to-decoder and encoder-to-decoder paths
TIME_VAE_DECODER_UPBLOCKS_ALIGNMENT: PASS - decoder up_blocks.0/1/2/3 are aligned as block-level and bridge-level components
TIME_VAE_DECODER_TAIL_AUDIT: PASS - official decoder tail topology and key audit
TIME_VAE_DECODER_TAIL_ORACLE_TENSORS: PASS - PyTorch decoder tail oracle tensors exported
TIME_VAE_DECODER_TAIL_NORM_ALIGNMENT: PASS - decoder.conv_norm_out alignment
TIME_VAE_DECODER_TAIL_ACT_ALIGNMENT: PASS - decoder.conv_act SiLU alignment
TIME_VAE_DECODER_TAIL_CONV_OUT_ALIGNMENT: PASS - decoder.conv_out alignment
TIME_VAE_DECODER_TAIL_ALIGNMENT: PASS - full isolated decoder tail alignment
TIME_VAE_DECODER_TAIL_SYNTHETIC_ALIGNMENT: PASS - isolated synthetic hidden -> decoder tail alignment
TIME_VAE_DECODER_UPBLOCKS0123_TO_TAIL_ALIGNMENT: PASS - decoder.up_blocks.0~3 -> decoder tail bridge alignment
TIME_VAE_DECODER_ENTRY_MIDBLOCK_UPBLOCKS0123_TAIL_ALIGNMENT: PASS - decoder entry -> mid_block -> up_blocks.0~3 -> tail bridge alignment
TIME_VAE_QUANT_TO_DECODER_TAIL_ALIGNMENT: PASS - quant moments -> posterior mode -> decoder tail bridge alignment
TIME_VAE_ENCODER_TO_DECODER_TAIL_ALIGNMENT: PASS - encoder -> quant -> deterministic decoder tail bridge alignment
TIME_VAE_DECODER_TAIL_BRIDGE_ALIGNMENT: PASS - deterministic decoder tail aligned through isolated, decoder-entry, quant-to-decoder and encoder-to-decoder paths
TIME_VAE_DETERMINISTIC_DECODER_ALIGNMENT: PASS - post_quant_conv -> decoder.conv_in -> decoder.mid_block -> decoder.up_blocks.0~3 -> decoder tail deterministic stack alignment
TIME_VAE_FULL_DECODER_ALIGNMENT: PASS - deterministic decoder stack module-level alignment only; not stochastic VAE sampling, not full VAE API, and not full TADSR inference
TIME_VAE_DETERMINISTIC_RECONSTRUCTION_ALIGNMENT: PASS - deterministic encoder -> quant_conv moments -> posterior mode -> decoder stack -> tail alignment; no random sampling
TIME_VAE_FULL_API_AUDIT: PASS - official TimeAwareAutoencoderKL API/method/config audit for full boundary
TIME_VAE_PIPELINE_USAGE_AUDIT: PASS - official TADSR pipeline VAE usage audited: encode/sample/scale/decode/clamp plus tiled-hook note
TIME_VAE_FULL_BOUNDARY_CONTRACT_AUDIT: PASS - minimal deterministic TimeVAE boundary contract defined from official TADSR usage
TIME_VAE_FULL_BOUNDARY_ORACLE_TENSORS: PASS - official PyTorch oracle tensors for TimeVAE encode/sample/scale/decode/clamp boundary exported
TIME_VAE_FULL_BOUNDARY_ENCODE_ALIGNMENT: PASS - Jittor encoder/quant moments alignment inside the TimeVAE full-boundary tester
TIME_VAE_FULL_BOUNDARY_DECODE_ALIGNMENT: PASS - Jittor decode path alignment inside the TimeVAE full-boundary tester
TIME_VAE_FULL_BOUNDARY_POSTPROCESS_ALIGNMENT: PASS - Jittor clamp[-1,1] postprocess alignment for the TimeVAE boundary
TIME_VAE_FULL_BOUNDARY_ALIGNMENT: PASS - alignment-only non-tiled encode/sample/scale/decode/clamp TimeVAE boundary; no scheduler/full inference
TIME_VAE_TILED_HOOK_AUDIT: PASS - official VAEHook patch targets and dispatch behavior audited; no Jittor tiled implementation claimed
TIME_VAE_TILED_PIPELINE_USAGE_AUDIT: PASS - official TADSR_test VAEHook usage and encode/decode call paths audited
TIME_VAE_TILED_BOUNDARY_CONTRACT_AUDIT: PASS - official tiled VAE boundary contract audited, including encoder tiling and decoder hook blocker
TIME_VAE_TILED_ORACLE_FEASIBILITY: BLOCKED_DECODER_HOOK_CONTRACT - official tiled encode/decode oracle feasibility; BLOCKED is acceptable if contract is explicitly diagnosed
TIME_VAE_TILED_ORACLE_TENSORS: BLOCKED_DECODER_HOOK_CONTRACT - official tiled VAE oracle tensors or blocked metadata exported without running scheduler/full inference
TIME_VAE_TILED_VS_NONTILED_REFERENCE_RECORDED: NOT_APPLICABLE - tiled-vs-non-tiled comparison recorded only when a truthful official tiled oracle exists
TIME_VAE_ACTUAL_VAEHOOK_BEHAVIOR_DECISION: PASS - design decision: mirror official actual VAEHook behavior rather than corrected tiled decoder
TIME_VAE_ACTUAL_VAEHOOK_BEHAVIOR_AUDIT: PASS - official actual VAEHook behavior audited: encoder hook can tile, decoder hook dispatches original_forward
TIME_VAE_OFFICIAL_MIRROR_POLICY: PASS - mainline policy mirrors official actual behavior; corrected tiled decoder is beyond official and deferred
TIME_VAE_DECODER_TILED_PATH_STATUS: BLOCKED_DECODER_HOOK_CONTRACT - official decoder tiled path remains blocked/not reachable because decoder hook has time_vae=False
TIME_VAE_ACTUAL_ENCODER_TILED_ORACLE_FEASIBILITY: PASS - controlled official actual encoder hook/tiled path feasibility
TIME_VAE_ACTUAL_VAEHOOK_ORACLE_TENSORS: PASS - official actual VAEHook behavior oracle tensors exported; not an ideal corrected tiled decoder oracle
TIME_VAE_ACTUAL_ENCODER_TILED_ORACLE_TENSORS: PASS - actual encoder hook oracle tensors with encoder tiled path trigger status
TIME_VAE_ACTUAL_DECODER_HOOK_ORIGINAL_FORWARD_ORACLE: PASS - actual decoder hook oracle confirms installed hook uses original_forward and not tiled decode
TIME_VAE_ACTUAL_VAEHOOK_ORACLE_CONTRACT_TEST: PASS - metadata-only contract test for official actual VAEHook behavior oracle
TIME_VAE_ACTUAL_ENCODER_TILE_QUEUE_ALIGNMENT: PASS - Jittor actual-hook wrapper mirrors official encoder VAEHook task queue, tile split/crop/write semantics, and cross-tile GroupNorm bookkeeping
TIME_VAE_ACTUAL_ENCODER_TILED_RAW_OUTPUT_ALIGNMENT: PASS - Jittor tiled encoder raw output before quant_conv against official actual VAEHook raw encoder oracle
TIME_VAE_ACTUAL_ENCODER_TILED_MOMENTS_ALIGNMENT: PASS - Jittor actual-hook wrapper computes official tiled encoder output plus quant_conv moments parity
TIME_VAE_ACTUAL_ENCODER_TILED_POSTERIOR_ALIGNMENT: PASS - Jittor actual-hook wrapper posterior tensors against official actual encoder oracle
TIME_VAE_ACTUAL_ENCODER_TILED_ALIGNMENT: PASS - official-actual encoder hook behavior alignment through tiled task queue and quant_conv
TIME_VAE_ACTUAL_DECODER_ORIGINAL_FORWARD_ALIGNMENT: PASS - official-actual decoder hook dispatches to original_forward and aligns numerically
TIME_VAE_ACTUAL_DECODER_HOOK_BEHAVIOR_ALIGNMENT: PASS - decoder hook behavior parity: installed hook, time_vae=False, original_forward path
TIME_VAE_ACTUAL_VAEHOOK_BEHAVIOR_ALIGNMENT: PASS - Jittor official-actual VAEHook behavior wrapper/tester status; does not invent a corrected tiled decoder
TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT: PASS - actual hook full boundary from encode/sample/scale/decode/clamp; decoder remains official original_forward and full TADSR inference remains closed
TIME_VAE_TILED_DECODER_ALIGNMENT: NOT_APPLICABLE_BLOCKED_DECODER_HOOK_CONTRACT - not applicable for official actual behavior because decoder VAEHook uses original_forward, not vae_tile_forward
TADSR_UNET_OVERVIEW_AUDIT: PASS - official UNet overview audit: config, input contract, LoRA state, entry modules
TADSR_UNET_ENTRY_AUDIT: PASS - official UNet entry audit: conv_in, time_proj, time_embedding
TADSR_UNET_ENTRY_ORACLE_TENSORS: PASS - PyTorch oracle tensors for UNet entry exported
TADSR_UNET_ENTRY_EFFECTIVE_WEIGHTS: PASS - effective static conv_in + time_embedding weights exported
TADSR_UNET_CENTER_INPUT_ALIGNMENT: PASS - center_input_sample bridge alignment
TADSR_UNET_CONV_IN_ALIGNMENT: PASS - UNet conv_in effective-weight alignment
TADSR_UNET_TIME_PROJ_ALIGNMENT: PASS - UNet Timesteps positional projection alignment
TADSR_UNET_TIME_EMBED_LINEAR1_ALIGNMENT: PASS - UNet time_embedding.linear_1 alignment
TADSR_UNET_TIME_EMBED_ACT_ALIGNMENT: PASS - UNet time_embedding SiLU activation alignment
TADSR_UNET_TIME_EMBED_ALIGNMENT: PASS - UNet full timestep embedding MLP alignment
TADSR_UNET_ENTRY_ALIGNMENT: PASS - UNet entry aggregate alignment before down_blocks.0
TADSR_UNET_ENTRY_VAE_MODE_ALIGNMENT: NOT_APPLICABLE - optional VAE-mode latent bridge; NOT_APPLICABLE until full TimeAware VAE API opens
TADSR_UNET_ENTRY_ALIGNMENT_AGGREGATE: PASS - UNet center input, conv_in, time_proj and time_embedding aligned; full UNet blocks are still unopened
TADSR_UNET_DOWNBLOCK0_AUDIT: PASS - official UNet down_blocks.0 overview audit
TADSR_UNET_DOWNBLOCK0_RESNET0_AUDIT: PASS - official down_blocks.0.resnets.0 audit
TADSR_UNET_DOWNBLOCK0_RESNET0_ORACLE_TENSORS: PASS - PyTorch oracle tensors for down_blocks.0.resnets.0 exported
TADSR_UNET_DOWNBLOCK0_RESNET0_EFFECTIVE_WEIGHTS: PASS - effective static weights for down_blocks.0.resnets.0 exported locally
TADSR_UNET_DOWNBLOCK0_RESNET0_NORM1_ALIGNMENT: PASS - down_blocks.0.resnets.0 norm1 alignment
TADSR_UNET_DOWNBLOCK0_RESNET0_CONV1_ALIGNMENT: PASS - down_blocks.0.resnets.0 conv1 alignment
TADSR_UNET_DOWNBLOCK0_RESNET0_TEMB_PROJ_ALIGNMENT: PASS - down_blocks.0.resnets.0 time_emb_proj alignment
TADSR_UNET_DOWNBLOCK0_RESNET0_CONV2_ALIGNMENT: PASS - down_blocks.0.resnets.0 conv2 alignment
TADSR_UNET_DOWNBLOCK0_RESNET0_ALIGNMENT: PASS - entry hidden/temb -> down_blocks.0.resnets.0 alignment
TADSR_UNET_DOWNBLOCK0_RESNET0_SYNTHETIC_ALIGNMENT: PASS - isolated synthetic hidden/temb -> down_blocks.0.resnets.0 alignment
TADSR_UNET_ENTRY_TO_DOWNBLOCK0_RESNET0_ALIGNMENT: PASS - UNet entry -> down_blocks.0.resnets.0 bridge alignment
TADSR_UNET_DOWNBLOCK0_RESNET0_SHORTCUT_ALIGNMENT: NOT_APPLICABLE - down_blocks.0.resnets.0 shortcut alignment; NOT_APPLICABLE when identity shortcut is official
TADSR_UNET_DOWNBLOCK0_RESNET0_BRIDGE_ALIGNMENT: PASS - first UNet down_blocks.0 leaf ResNet aligned in isolated and entry-bridge paths only; full down_blocks.0 remains unopened
TADSR_UNET_DOWNBLOCK0_ATTENTION0_AUDIT: PASS - official down_blocks.0.attentions.0 audit
TADSR_UNET_DOWNBLOCK0_ATTENTION0_ORACLE_TENSORS: PASS - PyTorch oracle tensors for down_blocks.0.attentions.0 exported
TADSR_UNET_DOWNBLOCK0_ATTENTION0_EFFECTIVE_WEIGHTS: PASS - effective static weights for down_blocks.0.attentions.0 exported locally
TADSR_UNET_DOWNBLOCK0_ATTENTION0_NORM_ALIGNMENT: PASS - attention0 top-level GroupNorm alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION0_PROJ_IN_ALIGNMENT: PASS - attention0 proj_in alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION0_SEQUENCE_ALIGNMENT: PASS - attention0 NCHW-to-sequence alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION0_ATTN1_QKV_ALIGNMENT: PASS - attention0 transformer0 attn1 QKV alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION0_ATTN1_ALIGNMENT: PASS - attention0 transformer0 self-attention output alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION0_AFTER_ATTN1_ALIGNMENT: PASS - attention0 transformer0 after-attn1 residual alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION0_ATTN2_QKV_ALIGNMENT: PASS - attention0 transformer0 attn2 QKV alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION0_ATTN2_ALIGNMENT: PASS - attention0 transformer0 cross-attention output alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION0_AFTER_ATTN2_ALIGNMENT: PASS - attention0 transformer0 after-attn2 residual alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION0_FF_ALIGNMENT: PASS - attention0 transformer0 feed-forward output alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION0_TRANSFORMER0_ALIGNMENT: PASS - attention0 full transformer block alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION0_ALIGNMENT: PASS - isolated down_blocks.0.attentions.0 alignment
TADSR_UNET_ENTRY_RESNET0_ATTENTION0_ALIGNMENT: PASS - UNet entry -> resnet0 -> attention0 bridge alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION0_BRIDGE_ALIGNMENT: PASS - down_blocks.0.attentions.0 aligned in isolated and entry-resnet0 bridge paths only; full down_blocks.0 remains unopened
TADSR_UNET_DOWNBLOCK0_RESNET1_AUDIT: PASS - official down_blocks.0.resnets.1 audit
TADSR_UNET_DOWNBLOCK0_RESNET1_ORACLE_TENSORS: PASS - PyTorch oracle tensors for down_blocks.0.resnets.1 exported
TADSR_UNET_DOWNBLOCK0_RESNET1_EFFECTIVE_WEIGHTS: PASS - effective static weights for down_blocks.0.resnets.1 exported locally
TADSR_UNET_DOWNBLOCK0_RESNET1_NORM1_ALIGNMENT: PASS - down_blocks.0.resnets.1 norm1 alignment
TADSR_UNET_DOWNBLOCK0_RESNET1_CONV1_ALIGNMENT: PASS - down_blocks.0.resnets.1 conv1 alignment
TADSR_UNET_DOWNBLOCK0_RESNET1_TEMB_PROJ_ALIGNMENT: PASS - down_blocks.0.resnets.1 time_emb_proj alignment
TADSR_UNET_DOWNBLOCK0_RESNET1_CONV2_ALIGNMENT: PASS - down_blocks.0.resnets.1 conv2 alignment
TADSR_UNET_DOWNBLOCK0_RESNET1_ALIGNMENT: PASS - entry-attention hidden/temb -> down_blocks.0.resnets.1 alignment
TADSR_UNET_DOWNBLOCK0_RESNET1_SYNTHETIC_ALIGNMENT: PASS - isolated synthetic hidden/temb -> down_blocks.0.resnets.1 alignment
TADSR_UNET_ENTRY_RESNET0_ATTENTION0_RESNET1_ALIGNMENT: PASS - UNet entry -> resnet0 -> attention0 -> resnet1 bridge alignment
TADSR_UNET_DOWNBLOCK0_RESNET1_SHORTCUT_ALIGNMENT: NOT_APPLICABLE - down_blocks.0.resnets.1 shortcut alignment; NOT_APPLICABLE when identity shortcut is official
TADSR_UNET_DOWNBLOCK0_RESNET1_BRIDGE_ALIGNMENT: PASS - down_blocks.0.resnets.1 aligned in isolated and entry-resnet0-attention0 bridge paths only; full down_blocks.0 remains unopened
TADSR_UNET_DOWNBLOCK0_ATTENTION1_AUDIT: PASS - official down_blocks.0.attentions.1 audit
TADSR_UNET_DOWNBLOCK0_ATTENTION1_ORACLE_TENSORS: PASS - PyTorch oracle tensors for down_blocks.0.attentions.1 exported
TADSR_UNET_DOWNBLOCK0_ATTENTION1_EFFECTIVE_WEIGHTS: PASS - effective static weights for down_blocks.0.attentions.1 exported locally
TADSR_UNET_DOWNBLOCK0_ATTENTION1_NORM_ALIGNMENT: PASS - attention1 top-level GroupNorm alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION1_PROJ_IN_ALIGNMENT: PASS - attention1 proj_in alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION1_SEQUENCE_ALIGNMENT: PASS - attention1 NCHW-to-sequence alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION1_ATTN1_QKV_ALIGNMENT: PASS - attention1 transformer0 attn1 QKV alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION1_ATTN1_ALIGNMENT: PASS - attention1 transformer0 self-attention output alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION1_AFTER_ATTN1_ALIGNMENT: PASS - attention1 transformer0 after-attn1 residual alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION1_ATTN2_QKV_ALIGNMENT: PASS - attention1 transformer0 attn2 QKV alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION1_ATTN2_ALIGNMENT: PASS - attention1 transformer0 cross-attention output alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION1_AFTER_ATTN2_ALIGNMENT: PASS - attention1 transformer0 after-attn2 residual alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION1_FF_ALIGNMENT: PASS - attention1 transformer0 feed-forward output alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION1_TRANSFORMER0_ALIGNMENT: PASS - attention1 full transformer block alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION1_ALIGNMENT: PASS - isolated down_blocks.0.attentions.1 alignment
TADSR_UNET_ENTRY_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT: PASS - UNet entry -> resnet0 -> attention0 -> resnet1 -> attention1 bridge alignment
TADSR_UNET_DOWNBLOCK0_ATTENTION1_BRIDGE_ALIGNMENT: PASS - down_blocks.0.attentions.1 aligned in isolated and entry-resnet0-attention0-resnet1 bridge paths only; full down_blocks.0 remains unopened
TADSR_UNET_DOWNBLOCK0_PRE_DOWNSAMPLER_ALIGNMENT: PASS - down_blocks.0 path through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 is aligned
TADSR_UNET_DOWNBLOCK0_DOWNSAMPLER_AUDIT: PASS - official down_blocks.0.downsamplers.0 audit
TADSR_UNET_DOWNBLOCK0_LOCAL_FORWARD_AUDIT: PASS - official local down_blocks.0 manual-chain audit
TADSR_UNET_DOWNBLOCK0_DOWNSAMPLER_ORACLE_TENSORS: PASS - PyTorch oracle tensors for down_blocks.0.downsamplers.0 and local down_blocks.0 exported
TADSR_UNET_DOWNBLOCK0_DOWNSAMPLER_EFFECTIVE_WEIGHTS: PASS - effective static weights for down_blocks.0.downsamplers.0 exported locally
TADSR_UNET_DOWNBLOCK0_DOWNSAMPLER_ALIGNMENT: PASS - isolated down_blocks.0.downsamplers.0 alignment
TADSR_UNET_DOWNBLOCK0_ALIGNMENT: PASS - full local UNet down_blocks.0 alignment: entry -> resnet0 -> attention0 -> resnet1 -> attention1 -> downsampler
TADSR_UNET_DOWNBLOCK0_BRIDGE_ALIGNMENT: PASS - complete local down_blocks.0 bridge aligned through downsampler; later UNet blocks remain unopened
TADSR_UNET_DOWNBLOCK1_AUDIT: PASS - official UNet down_blocks.1 overview audit, resnets.0 baseline context
TADSR_UNET_DOWNBLOCK1_RESNET0_AUDIT: PASS - official down_blocks.1.resnets.0 audit
TADSR_UNET_DOWNBLOCK1_RESNET0_ORACLE_TENSORS: PASS - PyTorch oracle tensors for down_blocks.1.resnets.0 exported
TADSR_UNET_DOWNBLOCK1_RESNET0_EFFECTIVE_WEIGHTS: PASS - effective static weights for down_blocks.1.resnets.0 exported locally
TADSR_UNET_DOWNBLOCK1_RESNET0_NORM1_ALIGNMENT: PASS - down_blocks.1.resnets.0 norm1 alignment
TADSR_UNET_DOWNBLOCK1_RESNET0_CONV1_ALIGNMENT: PASS - down_blocks.1.resnets.0 conv1 alignment
TADSR_UNET_DOWNBLOCK1_RESNET0_TEMB_PROJ_ALIGNMENT: PASS - down_blocks.1.resnets.0 time_emb_proj alignment
TADSR_UNET_DOWNBLOCK1_RESNET0_CONV2_ALIGNMENT: PASS - down_blocks.1.resnets.0 conv2 alignment
TADSR_UNET_DOWNBLOCK1_RESNET0_ALIGNMENT: PASS - down_blocks.1.resnets.0 local alignment
TADSR_UNET_DOWNBLOCK1_RESNET0_SYNTHETIC_ALIGNMENT: PASS - isolated synthetic hidden/temb -> down_blocks.1.resnets.0 alignment
TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_RESNET0_ALIGNMENT: PASS - UNet entry -> full local down_blocks.0 -> down_blocks.1.resnets.0 bridge alignment
TADSR_UNET_DOWNBLOCK1_RESNET0_SHORTCUT_ALIGNMENT: PASS - down_blocks.1.resnets.0 shortcut alignment; channel-changing ResNet should include conv_shortcut
TADSR_UNET_DOWNBLOCK1_RESNET0_BRIDGE_ALIGNMENT: PASS - down_blocks.1.resnets.0 aligned after complete local down_blocks.0 bridge; remaining down_blocks.1 modules are intentionally unopened
TADSR_UNET_DOWNBLOCK1_ATTENTION0_AUDIT: PASS - official down_blocks.1.attentions.0 audit
TADSR_UNET_DOWNBLOCK1_ATTENTION0_ORACLE_TENSORS: PASS - PyTorch oracle tensors for down_blocks.1.attentions.0 exported
TADSR_UNET_DOWNBLOCK1_ATTENTION0_EFFECTIVE_WEIGHTS: PASS - effective static weights for down_blocks.1.attentions.0 exported locally
TADSR_UNET_DOWNBLOCK1_ATTENTION0_NORM_ALIGNMENT: PASS - down_blocks.1.attentions.0 top-level norm alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION0_PROJ_IN_ALIGNMENT: PASS - down_blocks.1.attentions.0 proj_in alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION0_SEQUENCE_ALIGNMENT: PASS - down_blocks.1.attentions.0 NCHW-to-sequence alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION0_ATTN1_QKV_ALIGNMENT: PASS - down_blocks.1.attentions.0 transformer0 attn1 QKV alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION0_ATTN1_ALIGNMENT: PASS - down_blocks.1.attentions.0 transformer0 self-attention output alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION0_AFTER_ATTN1_ALIGNMENT: PASS - down_blocks.1.attentions.0 after-attn1 residual alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION0_ATTN2_QKV_ALIGNMENT: PASS - down_blocks.1.attentions.0 transformer0 attn2 QKV alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION0_ATTN2_ALIGNMENT: PASS - down_blocks.1.attentions.0 transformer0 cross-attention output alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION0_AFTER_ATTN2_ALIGNMENT: PASS - down_blocks.1.attentions.0 after-attn2 residual alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION0_FF_ALIGNMENT: PASS - down_blocks.1.attentions.0 feed-forward output alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION0_TRANSFORMER0_ALIGNMENT: PASS - down_blocks.1.attentions.0 full transformer block alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION0_ALIGNMENT: PASS - isolated down_blocks.1.attentions.0 alignment
TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_RESNET0_ATTENTION0_ALIGNMENT: PASS - UNet entry -> down_blocks.0 -> down_blocks.1.resnets.0 -> attentions.0 bridge alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION0_BRIDGE_ALIGNMENT: PASS - down_blocks.1.attentions.0 aligned after complete local down_blocks.0 and down_blocks.1.resnets.0 bridge; remaining down_blocks.1 modules are intentionally unopened
TADSR_UNET_DOWNBLOCK1_RESNET1_AUDIT: PASS - official down_blocks.1.resnets.1 audit
TADSR_UNET_DOWNBLOCK1_RESNET1_ORACLE_TENSORS: PASS - PyTorch oracle tensors for down_blocks.1.resnets.1 exported
TADSR_UNET_DOWNBLOCK1_RESNET1_EFFECTIVE_WEIGHTS: PASS - effective static weights for down_blocks.1.resnets.1 exported locally
TADSR_UNET_DOWNBLOCK1_RESNET1_NORM1_ALIGNMENT: PASS - down_blocks.1.resnets.1 norm1 alignment
TADSR_UNET_DOWNBLOCK1_RESNET1_CONV1_ALIGNMENT: PASS - down_blocks.1.resnets.1 conv1 alignment
TADSR_UNET_DOWNBLOCK1_RESNET1_TEMB_PROJ_ALIGNMENT: PASS - down_blocks.1.resnets.1 time_emb_proj alignment
TADSR_UNET_DOWNBLOCK1_RESNET1_CONV2_ALIGNMENT: PASS - down_blocks.1.resnets.1 conv2 alignment
TADSR_UNET_DOWNBLOCK1_RESNET1_ALIGNMENT: PASS - down_blocks.1.resnets.1 local alignment
TADSR_UNET_DOWNBLOCK1_RESNET1_SYNTHETIC_ALIGNMENT: PASS - isolated synthetic hidden/temb -> down_blocks.1.resnets.1 alignment
TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_RESNET0_ATTENTION0_RESNET1_ALIGNMENT: PASS - UNet entry -> down_blocks.0 -> down_blocks.1.resnets.0 -> attentions.0 -> resnets.1 bridge alignment
TADSR_UNET_DOWNBLOCK1_RESNET1_SHORTCUT_ALIGNMENT: NOT_APPLICABLE - down_blocks.1.resnets.1 shortcut alignment; should be NOT_APPLICABLE/PASS depending on audited shortcut
TADSR_UNET_DOWNBLOCK1_RESNET1_BRIDGE_ALIGNMENT: PASS - down_blocks.1.resnets.1 aligned after complete local down_blocks.0, down_blocks.1.resnets.0 and down_blocks.1.attentions.0 bridge; remaining down_blocks.1 modules are intentionally unopened
TADSR_UNET_DOWNBLOCK1_PRE_ATTENTION1_ALIGNMENT: PASS - down_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 only
TADSR_UNET_DOWNBLOCK1_ATTENTION1_AUDIT: PASS - official down_blocks.1.attentions.1 audit
TADSR_UNET_DOWNBLOCK1_ATTENTION1_ORACLE_TENSORS: PASS - PyTorch oracle tensors for down_blocks.1.attentions.1 exported
TADSR_UNET_DOWNBLOCK1_ATTENTION1_EFFECTIVE_WEIGHTS: PASS - effective static weights for down_blocks.1.attentions.1 exported locally
TADSR_UNET_DOWNBLOCK1_ATTENTION1_NORM_ALIGNMENT: PASS - down_blocks.1.attentions.1 top-level norm alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION1_PROJ_IN_ALIGNMENT: PASS - down_blocks.1.attentions.1 proj_in alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION1_SEQUENCE_ALIGNMENT: PASS - down_blocks.1.attentions.1 NCHW-to-sequence alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION1_ATTN1_QKV_ALIGNMENT: PASS - down_blocks.1.attentions.1 transformer0 attn1 QKV alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION1_ATTN1_ALIGNMENT: PASS - down_blocks.1.attentions.1 transformer0 self-attention output alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION1_AFTER_ATTN1_ALIGNMENT: PASS - down_blocks.1.attentions.1 after-attn1 residual alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION1_ATTN2_QKV_ALIGNMENT: PASS - down_blocks.1.attentions.1 transformer0 attn2 QKV alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION1_ATTN2_ALIGNMENT: PASS - down_blocks.1.attentions.1 transformer0 cross-attention output alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION1_AFTER_ATTN2_ALIGNMENT: PASS - down_blocks.1.attentions.1 after-attn2 residual alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION1_FF_ALIGNMENT: PASS - down_blocks.1.attentions.1 feed-forward output alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION1_TRANSFORMER0_ALIGNMENT: PASS - down_blocks.1.attentions.1 full transformer block alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION1_ALIGNMENT: PASS - isolated down_blocks.1.attentions.1 alignment
TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT: PASS - UNet entry -> down_blocks.0 -> down_blocks.1 resnet0/attention0/resnet1/attention1 bridge alignment
TADSR_UNET_DOWNBLOCK1_ATTENTION1_BRIDGE_ALIGNMENT: PASS - down_blocks.1.attentions.1 aligned after complete local down_blocks.0, down_blocks.1.resnets.0, attentions.0 and resnets.1 bridge; downsampler is checked separately below
TADSR_UNET_DOWNBLOCK1_PRE_DOWNSAMPLER_ALIGNMENT: PASS - down_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 only
TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_AUDIT: PASS - official down_blocks.1.downsamplers.0 audit
TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_LORA_AUDIT: PASS - LoRA wrapper/static effective conv audit for down_blocks.1.downsamplers.0
TADSR_UNET_DOWNBLOCK1_LOCAL_FORWARD_AUDIT: PASS - official manual-chain local down_blocks.1 audit
TADSR_UNET_DOWNBLOCK1_OUTPUT_STATES_AUDIT: PASS - official down_blocks.1 output_states order audit
TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_ORACLE_TENSORS: PASS - PyTorch oracle tensors for down_blocks.1.downsamplers.0 and local down_blocks.1 exported
TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_EFFECTIVE_WEIGHTS: PASS - effective static weights for down_blocks.1.downsamplers.0 exported locally
TADSR_UNET_DOWNBLOCK1_OUTPUT_STATES_ORACLE_TENSORS: PASS - PyTorch oracle tensors for down_blocks.1 output_states exported
TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_ALIGNMENT: PASS - isolated down_blocks.1.downsamplers.0 alignment
TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_RESNET0_ATTENTION0_RESNET1_ATTENTION1_DOWNSAMPLER_ALIGNMENT: PASS - UNet entry -> down_blocks.0 -> down_blocks.1 resnet0/attention0/resnet1/attention1/downsampler bridge alignment
TADSR_UNET_DOWNBLOCK1_ALIGNMENT: PASS - full local down_blocks.1 alignment through downsampler and output_states
TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_PADDING_ALIGNMENT: NOT_APPLICABLE - official Downsample2D uses Conv2d padding=1 directly; no separate padding op is exported
TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_CONV_ALIGNMENT: PASS - effective static Conv2d alignment for down_blocks.1.downsamplers.0
TADSR_UNET_DOWNBLOCK1_OUTPUT_HIDDEN_ALIGNMENT: PASS - down_blocks.1 hidden_states output equals official block.forward output
TADSR_UNET_DOWNBLOCK1_OUTPUT_STATES_ALIGNMENT: PASS - down_blocks.1 output_states tuple order and tensors match official block.forward
TADSR_UNET_DOWNBLOCK1_BRIDGE_ALIGNMENT: PASS - complete local down_blocks.1 bridge aligned through downsampler and output_states; later UNet blocks remain unopened
TADSR_UNET_DOWNBLOCK2_AUDIT: PASS - official UNet down_blocks.2 overview audit, resnets.0 baseline context
TADSR_UNET_DOWNBLOCK2_RESNET0_AUDIT: PASS - official down_blocks.2.resnets.0 audit
TADSR_UNET_DOWNBLOCK2_RESNET0_LORA_AUDIT: PASS - LoRA/effective weight state for down_blocks.2.resnets.0
TADSR_UNET_DOWNBLOCK2_RESNET0_ORACLE_TENSORS: PASS - PyTorch oracle tensors for down_blocks.2.resnets.0 exported
TADSR_UNET_DOWNBLOCK2_RESNET0_EFFECTIVE_WEIGHTS: PASS - effective static weights for down_blocks.2.resnets.0 exported locally
TADSR_UNET_DOWNBLOCK2_RESNET0_NORM1_ALIGNMENT: PASS - down_blocks.2.resnets.0 norm1 alignment
TADSR_UNET_DOWNBLOCK2_RESNET0_CONV1_ALIGNMENT: PASS - down_blocks.2.resnets.0 conv1 alignment
TADSR_UNET_DOWNBLOCK2_RESNET0_TEMB_PROJ_ALIGNMENT: PASS - down_blocks.2.resnets.0 time_emb_proj alignment
TADSR_UNET_DOWNBLOCK2_RESNET0_CONV2_ALIGNMENT: PASS - down_blocks.2.resnets.0 conv2 alignment
TADSR_UNET_DOWNBLOCK2_RESNET0_ALIGNMENT: PASS - down_blocks.2.resnets.0 local alignment
TADSR_UNET_DOWNBLOCK2_RESNET0_SYNTHETIC_ALIGNMENT: PASS - isolated synthetic hidden/temb -> down_blocks.2.resnets.0 alignment
TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ALIGNMENT: PASS - UNet entry -> full local down_blocks.0 -> full local down_blocks.1 -> down_blocks.2.resnets.0 bridge alignment
TADSR_UNET_DOWNBLOCK2_RESNET0_SHORTCUT_ALIGNMENT: PASS - down_blocks.2.resnets.0 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent
TADSR_UNET_DOWNBLOCK2_RESNET0_BRIDGE_ALIGNMENT: PASS - down_blocks.2.resnets.0 aligned after complete local down_blocks.0 and down_blocks.1 bridge; remaining down_blocks.2 modules are intentionally unopened
TADSR_UNET_DOWNBLOCK2_ATTENTION0_AUDIT: PASS - official down_blocks.2.attentions.0 audit
TADSR_UNET_DOWNBLOCK2_ATTENTION0_ORACLE_TENSORS: PASS - PyTorch oracle tensors for down_blocks.2.attentions.0 exported
TADSR_UNET_DOWNBLOCK2_ATTENTION0_EFFECTIVE_WEIGHTS: PASS - effective static weights for down_blocks.2.attentions.0 exported locally
TADSR_UNET_DOWNBLOCK2_ATTENTION0_NORM_ALIGNMENT: PASS - down_blocks.2.attentions.0 top-level norm alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION0_PROJ_IN_ALIGNMENT: PASS - down_blocks.2.attentions.0 proj_in alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION0_SEQUENCE_ALIGNMENT: PASS - down_blocks.2.attentions.0 NCHW-to-sequence alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION0_ATTN1_QKV_ALIGNMENT: PASS - down_blocks.2.attentions.0 transformer0 attn1 QKV alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION0_ATTN1_ALIGNMENT: PASS - down_blocks.2.attentions.0 transformer0 self-attention output alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION0_AFTER_ATTN1_ALIGNMENT: PASS - down_blocks.2.attentions.0 after-attn1 residual alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION0_ATTN2_QKV_ALIGNMENT: PASS - down_blocks.2.attentions.0 transformer0 attn2 QKV alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION0_ATTN2_ALIGNMENT: PASS - down_blocks.2.attentions.0 transformer0 cross-attention output alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION0_AFTER_ATTN2_ALIGNMENT: PASS - down_blocks.2.attentions.0 after-attn2 residual alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION0_FF_ALIGNMENT: PASS - down_blocks.2.attentions.0 feed-forward output alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION0_TRANSFORMER0_ALIGNMENT: PASS - down_blocks.2.attentions.0 full transformer block alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION0_ALIGNMENT: PASS - isolated down_blocks.2.attentions.0 alignment
TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ATTENTION0_ALIGNMENT: PASS - UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2.resnets.0 -> attentions.0 bridge alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION0_BRIDGE_ALIGNMENT: PASS - down_blocks.2.attentions.0 aligned after complete local down_blocks.0/down_blocks.1/down_blocks.2.resnets.0 bridge; remaining down_blocks.2 modules are intentionally unopened
TADSR_UNET_DOWNBLOCK2_PRE_RESNET1_ALIGNMENT: PASS - down_blocks.2 path is aligned through resnets.0 -> attentions.0 only
TADSR_UNET_DOWNBLOCK2_RESNET1_AUDIT: PASS - official down_blocks.2.resnets.1 audit
TADSR_UNET_DOWNBLOCK2_RESNET1_LORA_AUDIT: PASS - LoRA/effective weight state for down_blocks.2.resnets.1
TADSR_UNET_DOWNBLOCK2_RESNET1_ORACLE_TENSORS: PASS - PyTorch oracle tensors for down_blocks.2.resnets.1 exported
TADSR_UNET_DOWNBLOCK2_RESNET1_EFFECTIVE_WEIGHTS: PASS - effective static weights for down_blocks.2.resnets.1 exported locally
TADSR_UNET_DOWNBLOCK2_RESNET1_NORM1_ALIGNMENT: PASS - down_blocks.2.resnets.1 norm1 alignment
TADSR_UNET_DOWNBLOCK2_RESNET1_CONV1_ALIGNMENT: PASS - down_blocks.2.resnets.1 conv1 alignment
TADSR_UNET_DOWNBLOCK2_RESNET1_TEMB_PROJ_ALIGNMENT: PASS - down_blocks.2.resnets.1 time_emb_proj alignment
TADSR_UNET_DOWNBLOCK2_RESNET1_CONV2_ALIGNMENT: PASS - down_blocks.2.resnets.1 conv2 alignment
TADSR_UNET_DOWNBLOCK2_RESNET1_ALIGNMENT: PASS - down_blocks.2.resnets.1 local alignment
TADSR_UNET_DOWNBLOCK2_RESNET1_SYNTHETIC_ALIGNMENT: PASS - isolated synthetic hidden/temb -> down_blocks.2.resnets.1 alignment
TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ATTENTION0_RESNET1_ALIGNMENT: PASS - UNet entry -> full local down_blocks.0/down_blocks.1 -> down_blocks.2.resnets.0 -> attentions.0 -> resnets.1 bridge alignment
TADSR_UNET_DOWNBLOCK2_RESNET1_SHORTCUT_ALIGNMENT: NOT_APPLICABLE - down_blocks.2.resnets.1 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent
TADSR_UNET_DOWNBLOCK2_RESNET1_BRIDGE_ALIGNMENT: PASS - down_blocks.2.resnets.1 aligned after complete local down_blocks.0/down_blocks.1/down_blocks.2.resnets.0/down_blocks.2.attentions.0 bridge; remaining down_blocks.2 modules are intentionally unopened
TADSR_UNET_DOWNBLOCK2_PRE_ATTENTION1_ALIGNMENT: PASS - down_blocks.2 path is aligned through resnets.0 -> attentions.0 -> resnets.1 only
TADSR_UNET_DOWNBLOCK2_ATTENTION1_AUDIT: PASS - official down_blocks.2.attentions.1 audit
TADSR_UNET_DOWNBLOCK2_ATTENTION1_ORACLE_TENSORS: PASS - PyTorch oracle tensors for down_blocks.2.attentions.1 exported
TADSR_UNET_DOWNBLOCK2_ATTENTION1_EFFECTIVE_WEIGHTS: PASS - effective static weights for down_blocks.2.attentions.1 exported locally
TADSR_UNET_DOWNBLOCK2_ATTENTION1_NORM_ALIGNMENT: PASS - down_blocks.2.attentions.1 top-level norm alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION1_PROJ_IN_ALIGNMENT: PASS - down_blocks.2.attentions.1 proj_in alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION1_SEQUENCE_ALIGNMENT: PASS - down_blocks.2.attentions.1 NCHW-to-sequence alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION1_ATTN1_QKV_ALIGNMENT: PASS - down_blocks.2.attentions.1 transformer0 attn1 QKV alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION1_ATTN1_ALIGNMENT: PASS - down_blocks.2.attentions.1 transformer0 self-attention output alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION1_AFTER_ATTN1_ALIGNMENT: PASS - down_blocks.2.attentions.1 after-attn1 residual alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION1_ATTN2_QKV_ALIGNMENT: PASS - down_blocks.2.attentions.1 transformer0 attn2 QKV alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION1_ATTN2_ALIGNMENT: PASS - down_blocks.2.attentions.1 transformer0 cross-attention output alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION1_AFTER_ATTN2_ALIGNMENT: PASS - down_blocks.2.attentions.1 after-attn2 residual alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION1_FF_ALIGNMENT: PASS - down_blocks.2.attentions.1 feed-forward output alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION1_TRANSFORMER0_ALIGNMENT: PASS - down_blocks.2.attentions.1 full transformer block alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION1_ALIGNMENT: PASS - isolated down_blocks.2.attentions.1 alignment
TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT: PASS - UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2.resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 bridge alignment
TADSR_UNET_DOWNBLOCK2_ATTENTION1_BRIDGE_ALIGNMENT: PASS - down_blocks.2.attentions.1 aligned after complete local down_blocks.0/down_blocks.1/down_blocks.2.resnets.0/down_blocks.2.attentions.0/down_blocks.2.resnets.1 bridge; downsampler remains intentionally unopened
TADSR_UNET_DOWNBLOCK2_PRE_DOWNSAMPLER_ALIGNMENT: PASS - down_blocks.2 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 only
TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_AUDIT: PASS - official down_blocks.2.downsamplers.0 audit
TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_LORA_AUDIT: PASS - LoRA/effective weight state for down_blocks.2.downsamplers.0
TADSR_UNET_DOWNBLOCK2_LOCAL_FORWARD_AUDIT: PASS - manual local down_blocks.2 chain matches official block.forward
TADSR_UNET_DOWNBLOCK2_OUTPUT_STATES_AUDIT: PASS - official down_blocks.2 residual output_states tuple audited
TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_ORACLE_TENSORS: PASS - PyTorch oracle tensors for down_blocks.2.downsamplers.0 exported
TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_EFFECTIVE_WEIGHTS: PASS - effective static weights for down_blocks.2.downsamplers.0 exported locally
TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_PADDING_ALIGNMENT: PASS - down_blocks.2.downsampler padding alignment; NOT_APPLICABLE for direct conv path
TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_CONV_ALIGNMENT: PASS - down_blocks.2.downsampler conv output alignment
TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_ALIGNMENT: PASS - isolated down_blocks.2.downsampler alignment
TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_DOWNSAMPLER_ALIGNMENT: PASS - UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2 local chain through downsampler bridge alignment
TADSR_UNET_DOWNBLOCK2_OUTPUT_HIDDEN_ALIGNMENT: PASS - down_blocks.2 final hidden_states output alignment
TADSR_UNET_DOWNBLOCK2_OUTPUT_STATES_ALIGNMENT: PASS - down_blocks.2 residual output_states tuple alignment
TADSR_UNET_DOWNBLOCK2_ALIGNMENT: PASS - full local down_blocks.2 alignment through output_states tuple
TADSR_UNET_DOWNBLOCK2_BRIDGE_ALIGNMENT: PASS - complete local down_blocks.2 bridge aligned through downsampler and output_states; later UNet blocks remain unopened
TADSR_UNET_DOWNBLOCK3_AUDIT: PASS - official UNet down_blocks.3 topology audit
TADSR_UNET_DOWNBLOCK3_RESNET0_AUDIT: PASS - official down_blocks.3.resnets.0 audit
TADSR_UNET_DOWNBLOCK3_RESNET0_LORA_AUDIT: PASS - LoRA/effective weight state for down_blocks.3.resnets.0
TADSR_UNET_DOWNBLOCK3_RESNET0_ORACLE_TENSORS: PASS - PyTorch oracle tensors for down_blocks.3.resnets.0 exported
TADSR_UNET_DOWNBLOCK3_RESNET0_EFFECTIVE_WEIGHTS: PASS - effective static weights for down_blocks.3.resnets.0 exported locally
TADSR_UNET_DOWNBLOCK3_RESNET0_NORM1_ALIGNMENT: PASS - down_blocks.3.resnets.0 norm1 alignment
TADSR_UNET_DOWNBLOCK3_RESNET0_CONV1_ALIGNMENT: PASS - down_blocks.3.resnets.0 conv1 alignment
TADSR_UNET_DOWNBLOCK3_RESNET0_TEMB_PROJ_ALIGNMENT: PASS - down_blocks.3.resnets.0 time_emb_proj alignment
TADSR_UNET_DOWNBLOCK3_RESNET0_CONV2_ALIGNMENT: PASS - down_blocks.3.resnets.0 conv2 alignment
TADSR_UNET_DOWNBLOCK3_RESNET0_ALIGNMENT: PASS - down_blocks.3.resnets.0 local alignment
TADSR_UNET_DOWNBLOCK3_RESNET0_SYNTHETIC_ALIGNMENT: PASS - isolated synthetic hidden/temb -> down_blocks.3.resnets.0 alignment
TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_DOWNBLOCK3_RESNET0_ALIGNMENT: PASS - UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2 -> down_blocks.3.resnets.0 bridge alignment
TADSR_UNET_DOWNBLOCK3_RESNET0_SHORTCUT_ALIGNMENT: NOT_APPLICABLE - down_blocks.3.resnets.0 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent
TADSR_UNET_DOWNBLOCK3_RESNET0_BRIDGE_ALIGNMENT: PASS - down_blocks.3.resnets.0 aligned after complete local down_blocks.0/1/2 bridge; remaining down_blocks.3 modules are intentionally unopened
TADSR_UNET_DOWNBLOCK3_RESNET1_AUDIT: PASS - official down_blocks.3.resnets.1 audit
TADSR_UNET_DOWNBLOCK3_LOCAL_FORWARD_AUDIT: PASS - official down_blocks.3 local forward hidden output audited
TADSR_UNET_DOWNBLOCK3_OUTPUT_STATES_AUDIT: PASS - official down_blocks.3 residual output_states tuple audited
TADSR_UNET_DOWNBLOCK3_RESNET1_ORACLE_TENSORS: PASS - PyTorch oracle tensors for down_blocks.3.resnets.1 exported
TADSR_UNET_DOWNBLOCK3_RESNET1_EFFECTIVE_WEIGHTS: PASS - effective static weights for down_blocks.3.resnets.1 exported locally
TADSR_UNET_DOWNBLOCK3_RESNET1_NORM1_ALIGNMENT: PASS - down_blocks.3.resnets.1 norm1 alignment
TADSR_UNET_DOWNBLOCK3_RESNET1_CONV1_ALIGNMENT: PASS - down_blocks.3.resnets.1 conv1 alignment
TADSR_UNET_DOWNBLOCK3_RESNET1_TEMB_PROJ_ALIGNMENT: PASS - down_blocks.3.resnets.1 time_emb_proj alignment
TADSR_UNET_DOWNBLOCK3_RESNET1_CONV2_ALIGNMENT: PASS - down_blocks.3.resnets.1 conv2 alignment
TADSR_UNET_DOWNBLOCK3_RESNET1_ALIGNMENT: PASS - down_blocks.3.resnets.1 local alignment
TADSR_UNET_DOWNBLOCK3_RESNET1_SYNTHETIC_ALIGNMENT: PASS - isolated synthetic hidden/temb -> down_blocks.3.resnets.1 alignment
TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_DOWNBLOCK3_RESNET0_RESNET1_ALIGNMENT: PASS - UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2 -> down_blocks.3.resnets.0 -> resnets.1 bridge alignment
TADSR_UNET_DOWNBLOCK3_OUTPUT_HIDDEN_ALIGNMENT: PASS - down_blocks.3 final hidden_states output alignment
TADSR_UNET_DOWNBLOCK3_OUTPUT_STATES_ALIGNMENT: PASS - down_blocks.3 residual output_states tuple alignment
TADSR_UNET_DOWNBLOCK3_ALIGNMENT: PASS - full local down_blocks.3 alignment through output_states tuple
TADSR_UNET_DOWNBLOCK3_RESNET1_SHORTCUT_ALIGNMENT: NOT_APPLICABLE - down_blocks.3.resnets.1 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent
TADSR_UNET_DOWNBLOCK3_BRIDGE_ALIGNMENT: PASS - complete local down_blocks.3 bridge aligned through resnet0 -> resnet1 and output_states; mid_block remains unopened
TADSR_UNET_MIDBLOCK_AUDIT: PASS - official UNet mid_block topology audit
TADSR_UNET_MIDBLOCK_RESNET0_AUDIT: PASS - official mid_block.resnets.0 audit
TADSR_UNET_MIDBLOCK_RESNET0_LORA_AUDIT: PASS - LoRA/effective weight state for mid_block.resnets.0
TADSR_UNET_MIDBLOCK_RESNET0_ORACLE_TENSORS: PASS - PyTorch oracle tensors for mid_block.resnets.0 exported
TADSR_UNET_MIDBLOCK_RESNET0_EFFECTIVE_WEIGHTS: PASS - effective static weights for mid_block.resnets.0 exported locally
TADSR_UNET_MIDBLOCK_RESNET0_NORM1_ALIGNMENT: PASS - mid_block.resnets.0 norm1 alignment
TADSR_UNET_MIDBLOCK_RESNET0_CONV1_ALIGNMENT: PASS - mid_block.resnets.0 conv1 alignment
TADSR_UNET_MIDBLOCK_RESNET0_TEMB_PROJ_ALIGNMENT: PASS - mid_block.resnets.0 time_emb_proj alignment
TADSR_UNET_MIDBLOCK_RESNET0_CONV2_ALIGNMENT: PASS - mid_block.resnets.0 conv2 alignment
TADSR_UNET_MIDBLOCK_RESNET0_ALIGNMENT: PASS - mid_block.resnets.0 local alignment
TADSR_UNET_MIDBLOCK_RESNET0_SYNTHETIC_ALIGNMENT: PASS - isolated synthetic hidden/temb -> mid_block.resnets.0 alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_RESNET0_ALIGNMENT: PASS - UNet entry -> down_blocks.0/1/2/3 -> mid_block.resnets.0 bridge alignment
TADSR_UNET_MIDBLOCK_RESNET0_SHORTCUT_ALIGNMENT: NOT_APPLICABLE - mid_block.resnets.0 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent
TADSR_UNET_MIDBLOCK_RESNET0_BRIDGE_ALIGNMENT: PASS - mid_block.resnets.0 aligned after complete local down_blocks.0/1/2/3 bridge; attention/resnet1 remain unopened
TADSR_UNET_MIDBLOCK_ATTENTION0_AUDIT: PASS - official mid_block.attentions.0 audit
TADSR_UNET_MIDBLOCK_ATTENTION0_LORA_AUDIT: PASS - LoRA/effective weight state for mid_block.attentions.0
TADSR_UNET_MIDBLOCK_ATTENTION0_ORACLE_TENSORS: PASS - PyTorch oracle tensors for mid_block.attentions.0 exported
TADSR_UNET_MIDBLOCK_ATTENTION0_EFFECTIVE_WEIGHTS: PASS - effective static weights for mid_block.attentions.0 exported locally
TADSR_UNET_MIDBLOCK_ATTENTION0_NORM_ALIGNMENT: PASS - mid_block.attentions.0 top-level norm alignment
TADSR_UNET_MIDBLOCK_ATTENTION0_PROJ_IN_ALIGNMENT: PASS - mid_block.attentions.0 proj_in alignment
TADSR_UNET_MIDBLOCK_ATTENTION0_SEQUENCE_ALIGNMENT: PASS - mid_block.attentions.0 NCHW-to-sequence alignment
TADSR_UNET_MIDBLOCK_ATTENTION0_ATTN1_QKV_ALIGNMENT: PASS - mid_block.attentions.0 self-attention QKV alignment
TADSR_UNET_MIDBLOCK_ATTENTION0_ATTN1_ALIGNMENT: PASS - mid_block.attentions.0 self-attention output alignment
TADSR_UNET_MIDBLOCK_ATTENTION0_AFTER_ATTN1_ALIGNMENT: PASS - mid_block.attentions.0 after-attn1 residual alignment
TADSR_UNET_MIDBLOCK_ATTENTION0_ATTN2_QKV_ALIGNMENT: PASS - mid_block.attentions.0 cross-attention QKV alignment
TADSR_UNET_MIDBLOCK_ATTENTION0_ATTN2_ALIGNMENT: PASS - mid_block.attentions.0 cross-attention output alignment
TADSR_UNET_MIDBLOCK_ATTENTION0_AFTER_ATTN2_ALIGNMENT: PASS - mid_block.attentions.0 after-attn2 residual alignment
TADSR_UNET_MIDBLOCK_ATTENTION0_FF_ALIGNMENT: PASS - mid_block.attentions.0 feed-forward output alignment
TADSR_UNET_MIDBLOCK_ATTENTION0_TRANSFORMER0_ALIGNMENT: PASS - mid_block.attentions.0 transformer block alignment
TADSR_UNET_MIDBLOCK_ATTENTION0_ALIGNMENT: PASS - isolated mid_block.attentions.0 alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_RESNET0_ATTENTION0_ALIGNMENT: PASS - UNet entry -> down_blocks.0/1/2/3 -> mid_block.resnets.0 -> mid_block.attentions.0 bridge alignment
TADSR_UNET_MIDBLOCK_ATTENTION0_BRIDGE_ALIGNMENT: PASS - mid_block.attentions.0 aligned in isolated and complete pre-attention bridge paths; resnet1 remains unopened
TADSR_UNET_MIDBLOCK_PRE_RESNET1_ALIGNMENT: PASS - UNet path through mid_block.resnets.0 and mid_block.attentions.0 is aligned; mid_block.resnets.1 is evaluated separately below
TADSR_UNET_MIDBLOCK_RESNET1_AUDIT: PASS - official mid_block.resnets.1 audit
TADSR_UNET_MIDBLOCK_LOCAL_FORWARD_AUDIT: PASS - official local mid_block forward audit
TADSR_UNET_MIDBLOCK_RESNET1_LORA_AUDIT: PASS - LoRA/effective weight state for mid_block.resnets.1
TADSR_UNET_MIDBLOCK_RESNET1_ORACLE_TENSORS: PASS - PyTorch oracle tensors for mid_block.resnets.1 exported
TADSR_UNET_MIDBLOCK_RESNET1_EFFECTIVE_WEIGHTS: PASS - effective static weights for mid_block.resnets.1 exported locally
TADSR_UNET_MIDBLOCK_RESNET1_NORM1_ALIGNMENT: PASS - mid_block.resnets.1 norm1 alignment
TADSR_UNET_MIDBLOCK_RESNET1_CONV1_ALIGNMENT: PASS - mid_block.resnets.1 conv1 alignment
TADSR_UNET_MIDBLOCK_RESNET1_TEMB_PROJ_ALIGNMENT: PASS - mid_block.resnets.1 time_emb_proj alignment
TADSR_UNET_MIDBLOCK_RESNET1_CONV2_ALIGNMENT: PASS - mid_block.resnets.1 conv2 alignment
TADSR_UNET_MIDBLOCK_RESNET1_ALIGNMENT: PASS - isolated entry-input mid_block.resnets.1 alignment
TADSR_UNET_MIDBLOCK_RESNET1_SYNTHETIC_ALIGNMENT: PASS - synthetic hidden/temb -> mid_block.resnets.1 alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_RESNET0_ATTENTION0_RESNET1_ALIGNMENT: PASS - UNet entry -> down_blocks.0/1/2/3 -> mid_block.resnets.0 -> attention0 -> resnets.1 bridge alignment
TADSR_UNET_MIDBLOCK_OUTPUT_HIDDEN_ALIGNMENT: PASS - manual local mid_block hidden output vs official local mid_block forward
TADSR_UNET_MIDBLOCK_RESNET1_SHORTCUT_ALIGNMENT: NOT_APPLICABLE - mid_block.resnets.1 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent
TADSR_UNET_MIDBLOCK_OUTPUT_STATES_ALIGNMENT: NOT_APPLICABLE - official local mid_block output_states status; NOT_APPLICABLE when mid_block returns hidden_states only
TADSR_UNET_MIDBLOCK_BRIDGE_ALIGNMENT: PASS - complete local mid_block chain resnets.0 -> attentions.0 -> resnets.1 aligned; up_blocks remain unopened
TADSR_UNET_MIDBLOCK_ALIGNMENT: PASS - local mid_block hidden output aligned through resnets.1; this is not full UNet forward
TADSR_UNET_UPBLOCK0_AUDIT: PASS - official up_blocks.0 topology audit
TADSR_UNET_UPBLOCK0_RESNET0_AUDIT: PASS - official up_blocks.0.resnets.0 audit
TADSR_UNET_UPBLOCK0_RESNET0_RESIDUAL_CONTRACT_AUDIT: PASS - official residual pop/concat contract audit
TADSR_UNET_UPBLOCK0_RESNET0_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.0.resnets.0 exported
TADSR_UNET_UPBLOCK0_RESNET0_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.0.resnets.0 exported locally
TADSR_UNET_UPBLOCK0_RESNET0_RESIDUAL_CONCAT_ALIGNMENT: PASS - up_blocks.0.resnets.0 hidden/residual concat alignment
TADSR_UNET_UPBLOCK0_RESNET0_NORM1_ALIGNMENT: PASS - up_blocks.0.resnets.0 norm1 alignment
TADSR_UNET_UPBLOCK0_RESNET0_CONV1_ALIGNMENT: PASS - up_blocks.0.resnets.0 conv1 alignment
TADSR_UNET_UPBLOCK0_RESNET0_TEMB_PROJ_ALIGNMENT: PASS - up_blocks.0.resnets.0 time_emb_proj alignment
TADSR_UNET_UPBLOCK0_RESNET0_CONV2_ALIGNMENT: PASS - up_blocks.0.resnets.0 conv2 alignment
TADSR_UNET_UPBLOCK0_RESNET0_ALIGNMENT: PASS - isolated up_blocks.0.resnets.0 after official residual concat alignment
TADSR_UNET_UPBLOCK0_RESNET0_SYNTHETIC_ALIGNMENT: PASS - synthetic hidden/residual -> up_blocks.0.resnets.0 alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_RESNET0_ALIGNMENT: PASS - UNet entry -> down_blocks.0/1/2/3 -> mid_block -> up_blocks.0.resnets.0 bridge alignment
TADSR_UNET_UPBLOCK0_RESNET0_SHORTCUT_ALIGNMENT: PASS - up_blocks.0.resnets.0 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent
TADSR_UNET_UPBLOCK0_RESNET0_BRIDGE_ALIGNMENT: PASS - first up path leaf up_blocks.0.resnets.0 aligned after official residual consumption; full up_blocks.0 remains unopened
TADSR_UNET_UPBLOCK0_RESNET1_AUDIT: BLOCKED - official up_blocks.0.resnets.1 audit
TADSR_UNET_UPBLOCK0_RESNET1_RESIDUAL_CONTRACT_AUDIT: BLOCKED - official residual pop/concat contract audit for second upblock residual
TADSR_UNET_UPBLOCK0_RESNET1_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.0.resnets.1 exported
TADSR_UNET_UPBLOCK0_RESNET1_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.0.resnets.1 exported locally
TADSR_UNET_UPBLOCK0_RESNET1_RESIDUAL_CONCAT_ALIGNMENT: BLOCKED - up_blocks.0.resnets.1 hidden/residual concat alignment
TADSR_UNET_UPBLOCK0_RESNET1_NORM1_ALIGNMENT: BLOCKED - up_blocks.0.resnets.1 norm1 alignment
TADSR_UNET_UPBLOCK0_RESNET1_CONV1_ALIGNMENT: BLOCKED - up_blocks.0.resnets.1 conv1 alignment
TADSR_UNET_UPBLOCK0_RESNET1_TEMB_PROJ_ALIGNMENT: BLOCKED - up_blocks.0.resnets.1 time_emb_proj alignment
TADSR_UNET_UPBLOCK0_RESNET1_CONV2_ALIGNMENT: BLOCKED - up_blocks.0.resnets.1 conv2 alignment
TADSR_UNET_UPBLOCK0_RESNET1_ALIGNMENT: BLOCKED - isolated up_blocks.0.resnets.1 after official residual concat alignment
TADSR_UNET_UPBLOCK0_RESNET1_SYNTHETIC_ALIGNMENT: BLOCKED - synthetic hidden/residual -> up_blocks.0.resnets.1 alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_RESNET0_RESNET1_ALIGNMENT: BLOCKED - UNet entry -> down_blocks.0/1/2/3 -> mid_block -> up_blocks.0.resnets.0 -> resnets.1 bridge alignment
TADSR_UNET_UPBLOCK0_RESNET1_SHORTCUT_ALIGNMENT: BLOCKED - up_blocks.0.resnets.1 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent
TADSR_UNET_UPBLOCK0_RESNET1_BRIDGE_ALIGNMENT: PARTIAL - up_blocks.0.resnets.1 aligned after official second residual consumption; full up_blocks.0 remains unopened
TADSR_UNET_UPBLOCK0_PRE_RESNET2_ALIGNMENT: PARTIAL - up_blocks.0 path is aligned through resnets.0 -> resnets.1 only
TADSR_UNET_UPBLOCK0_RESNET2_AUDIT: PASS - official up_blocks.0.resnets.2 audit
TADSR_UNET_UPBLOCK0_RESNET2_RESIDUAL_CONTRACT_AUDIT: PASS - official residual pop/concat contract audit for third upblock residual
TADSR_UNET_UPBLOCK0_RESNET2_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.0.resnets.2 exported
TADSR_UNET_UPBLOCK0_RESNET2_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.0.resnets.2 exported locally
TADSR_UNET_UPBLOCK0_RESNET2_RESIDUAL_CONCAT_ALIGNMENT: PASS - up_blocks.0.resnets.2 hidden/residual concat alignment
TADSR_UNET_UPBLOCK0_RESNET2_NORM1_ALIGNMENT: PASS - up_blocks.0.resnets.2 norm1 alignment
TADSR_UNET_UPBLOCK0_RESNET2_CONV1_ALIGNMENT: PASS - up_blocks.0.resnets.2 conv1 alignment
TADSR_UNET_UPBLOCK0_RESNET2_TEMB_PROJ_ALIGNMENT: PASS - up_blocks.0.resnets.2 time_emb_proj alignment
TADSR_UNET_UPBLOCK0_RESNET2_CONV2_ALIGNMENT: PASS - up_blocks.0.resnets.2 conv2 alignment
TADSR_UNET_UPBLOCK0_RESNET2_ALIGNMENT: PASS - isolated up_blocks.0.resnets.2 after official residual concat alignment
TADSR_UNET_UPBLOCK0_RESNET2_SYNTHETIC_ALIGNMENT: PASS - synthetic hidden/residual -> up_blocks.0.resnets.2 alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_RESNET0_RESNET1_RESNET2_ALIGNMENT: PASS - UNet entry -> down_blocks.0/1/2/3 -> mid_block -> up_blocks.0.resnets.0 -> resnets.1 -> resnets.2 bridge alignment
TADSR_UNET_UPBLOCK0_RESNET2_SHORTCUT_ALIGNMENT: PASS - up_blocks.0.resnets.2 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent
TADSR_UNET_UPBLOCK0_RESNET2_BRIDGE_ALIGNMENT: PARTIAL - up_blocks.0.resnets.2 aligned after official third residual consumption; full up_blocks.0 remains unopened
TADSR_UNET_UPBLOCK0_PRE_UPSAMPLER_ALIGNMENT: PARTIAL - up_blocks.0 path is aligned through resnets.0 -> resnets.1 -> resnets.2 only
TADSR_UNET_UPBLOCK0_UPSAMPLER_AUDIT: PASS - official up_blocks.0.upsamplers.0 audit
TADSR_UNET_UPBLOCK0_LOCAL_FORWARD_AUDIT: PASS - safe official local up_blocks.0 hidden-state forward audit
TADSR_UNET_UPBLOCK0_UPSAMPLER_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.0.upsamplers.0 exported
TADSR_UNET_UPBLOCK0_UPSAMPLER_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.0.upsamplers.0 exported locally
TADSR_UNET_UPBLOCK0_LOCAL_FORWARD_ORACLE: PASS - PyTorch oracle for local up_blocks.0 hidden-state output exported
TADSR_UNET_UPBLOCK0_UPSAMPLER_INTERPOLATION_ALIGNMENT: PASS - nearest-2x interpolation input to upsampler conv alignment
TADSR_UNET_UPBLOCK0_UPSAMPLER_CONV_ALIGNMENT: PASS - effective upsampler conv output alignment
TADSR_UNET_UPBLOCK0_UPSAMPLER_ALIGNMENT: PASS - isolated up_blocks.0.upsamplers.0 alignment
TADSR_UNET_UPBLOCK0_UPSAMPLER_SYNTHETIC_ALIGNMENT: PASS - synthetic hidden -> up_blocks.0.upsamplers.0 alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_RESNETS012_UPSAMPLER_ALIGNMENT: PASS - UNet entry -> all down_blocks -> full local mid_block -> up_blocks.0.resnets.0/1/2 -> upsamplers.0 bridge alignment
TADSR_UNET_UPBLOCK0_OUTPUT_HIDDEN_ALIGNMENT: PASS - full local up_blocks.0 hidden-state output alignment
TADSR_UNET_UPBLOCK0_OUTPUT_STATES_ALIGNMENT: PASS - up_blocks.0 output-state contract alignment; official UpBlock2D returns hidden states only
TADSR_UNET_UPBLOCK0_BRIDGE_ALIGNMENT: PARTIAL - up_blocks.0 path is aligned through resnets.0 -> resnets.1 -> resnets.2 -> upsamplers.0
TADSR_UNET_UPBLOCK0_ALIGNMENT: PARTIAL - Full local up_blocks.0 hidden output is aligned; later up_blocks remain unopened
TADSR_UNET_UPBLOCK1_AUDIT: PASS - official up_blocks.1 topology audit through resnets.0 only
TADSR_UNET_UPBLOCK1_RESNET0_AUDIT: PASS - official up_blocks.1.resnets.0 audit
TADSR_UNET_UPBLOCK1_RESNET0_RESIDUAL_CONTRACT_AUDIT: PASS - official residual pop/concat contract audit after local up_blocks.0
TADSR_UNET_UPBLOCK1_RESNET0_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.1.resnets.0 exported
TADSR_UNET_UPBLOCK1_RESNET0_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.1.resnets.0 exported locally
TADSR_UNET_UPBLOCK1_RESNET0_RESIDUAL_CONCAT_ALIGNMENT: PASS - up_blocks.1.resnets.0 hidden/residual concat alignment
TADSR_UNET_UPBLOCK1_RESNET0_NORM1_ALIGNMENT: PASS - up_blocks.1.resnets.0 norm1 alignment
TADSR_UNET_UPBLOCK1_RESNET0_CONV1_ALIGNMENT: PASS - up_blocks.1.resnets.0 conv1 alignment
TADSR_UNET_UPBLOCK1_RESNET0_TEMB_PROJ_ALIGNMENT: PASS - up_blocks.1.resnets.0 time_emb_proj alignment
TADSR_UNET_UPBLOCK1_RESNET0_CONV2_ALIGNMENT: PASS - up_blocks.1.resnets.0 conv2 alignment
TADSR_UNET_UPBLOCK1_RESNET0_ALIGNMENT: PASS - isolated up_blocks.1.resnets.0 after official residual concat alignment
TADSR_UNET_UPBLOCK1_RESNET0_SYNTHETIC_ALIGNMENT: PASS - synthetic hidden/residual -> up_blocks.1.resnets.0 alignment
TADSR_UNET_UPBLOCK1_RESNET0_BRIDGE_ALIGNMENT: PASS - UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 bridge alignment
TADSR_UNET_UPBLOCK1_RESNET0_SHORTCUT_ALIGNMENT: PASS - up_blocks.1.resnets.0 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent
TADSR_UNET_UPBLOCK1_PRE_ATTENTION0_ALIGNMENT: PARTIAL - up_blocks.1 path is aligned through resnets.0 only
TADSR_UNET_UPBLOCK1_ATTENTION0_AUDIT: PASS - official up_blocks.1.attentions.0 topology/LoRA audit
TADSR_UNET_UPBLOCK1_ATTENTION0_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.1.attentions.0 exported
TADSR_UNET_UPBLOCK1_ATTENTION0_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.1.attentions.0 exported locally
TADSR_UNET_UPBLOCK1_ATTENTION0_NORM_ALIGNMENT: PASS - up_blocks.1.attentions.0 group norm alignment
TADSR_UNET_UPBLOCK1_ATTENTION0_SEQUENCE_ALIGNMENT: PASS - up_blocks.1.attentions.0 NCHW to sequence reshape alignment
TADSR_UNET_UPBLOCK1_ATTENTION0_PROJ_IN_ALIGNMENT: PASS - up_blocks.1.attentions.0 proj_in alignment
TADSR_UNET_UPBLOCK1_ATTENTION0_ATTN1_QKV_ALIGNMENT: PASS - up_blocks.1.attentions.0 self-attention q/k/v alignment
TADSR_UNET_UPBLOCK1_ATTENTION0_ATTN1_ALIGNMENT: PASS - up_blocks.1.attentions.0 self-attention output alignment
TADSR_UNET_UPBLOCK1_ATTENTION0_AFTER_ATTN1_ALIGNMENT: PASS - up_blocks.1.attentions.0 residual after attn1 alignment
TADSR_UNET_UPBLOCK1_ATTENTION0_ATTN2_QKV_ALIGNMENT: PASS - up_blocks.1.attentions.0 cross-attention q/k/v alignment
TADSR_UNET_UPBLOCK1_ATTENTION0_ATTN2_ALIGNMENT: PASS - up_blocks.1.attentions.0 cross-attention output alignment
TADSR_UNET_UPBLOCK1_ATTENTION0_AFTER_ATTN2_ALIGNMENT: PASS - up_blocks.1.attentions.0 residual after attn2 alignment
TADSR_UNET_UPBLOCK1_ATTENTION0_FF_ALIGNMENT: PASS - up_blocks.1.attentions.0 feed-forward output alignment
TADSR_UNET_UPBLOCK1_ATTENTION0_TRANSFORMER0_ALIGNMENT: PASS - up_blocks.1.attentions.0 transformer block output alignment
TADSR_UNET_UPBLOCK1_ATTENTION0_ALIGNMENT: PASS - isolated up_blocks.1.attentions.0 output alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_RESNET0_ATTENTION0_ALIGNMENT: PASS - UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 bridge alignment
TADSR_UNET_UPBLOCK1_ATTENTION0_BRIDGE_ALIGNMENT: PASS - up_blocks.1 path is aligned through resnets.0 -> attentions.0 only
TADSR_UNET_UPBLOCK1_PRE_RESNET1_ALIGNMENT: PASS - input to next unopened up_blocks.1.resnets.1 is now aligned
TADSR_UNET_UPBLOCK1_RESNET1_AUDIT: PASS - official up_blocks.1.resnets.1 audit
TADSR_UNET_UPBLOCK1_RESNET1_RESIDUAL_CONTRACT_AUDIT: PASS - official residual pop/concat contract audit after up_blocks.1.attentions.0
TADSR_UNET_UPBLOCK1_RESNET1_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.1.resnets.1 exported
TADSR_UNET_UPBLOCK1_RESNET1_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.1.resnets.1 exported locally
TADSR_UNET_UPBLOCK1_RESNET1_RESIDUAL_CONCAT_ALIGNMENT: PASS - up_blocks.1.resnets.1 hidden/residual concat alignment
TADSR_UNET_UPBLOCK1_RESNET1_NORM1_ALIGNMENT: PASS - up_blocks.1.resnets.1 norm1 alignment
TADSR_UNET_UPBLOCK1_RESNET1_CONV1_ALIGNMENT: PASS - up_blocks.1.resnets.1 conv1 alignment
TADSR_UNET_UPBLOCK1_RESNET1_TEMB_PROJ_ALIGNMENT: PASS - up_blocks.1.resnets.1 time_emb_proj alignment
TADSR_UNET_UPBLOCK1_RESNET1_CONV2_ALIGNMENT: PASS - up_blocks.1.resnets.1 conv2 alignment
TADSR_UNET_UPBLOCK1_RESNET1_ALIGNMENT: PASS - isolated up_blocks.1.resnets.1 after official residual concat alignment
TADSR_UNET_UPBLOCK1_RESNET1_SYNTHETIC_ALIGNMENT: PASS - synthetic hidden/residual -> up_blocks.1.resnets.1 alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_RESNET0_ATTENTION0_RESNET1_ALIGNMENT: PASS - UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 -> up_blocks.1.resnets.1 bridge alignment
TADSR_UNET_UPBLOCK1_RESNET1_SHORTCUT_ALIGNMENT: PASS - up_blocks.1.resnets.1 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent
TADSR_UNET_UPBLOCK1_RESNET1_BRIDGE_ALIGNMENT: PASS - up_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 only
TADSR_UNET_UPBLOCK1_PRE_ATTENTION1_ALIGNMENT: PASS - input to next unopened up_blocks.1.attentions.1 is now aligned
TADSR_UNET_UPBLOCK1_ATTENTION1_AUDIT: PASS - official up_blocks.1.attentions.1 topology/LoRA audit
TADSR_UNET_UPBLOCK1_ATTENTION1_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.1.attentions.1 exported
TADSR_UNET_UPBLOCK1_ATTENTION1_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.1.attentions.1 exported locally
TADSR_UNET_UPBLOCK1_ATTENTION1_NORM_ALIGNMENT: PASS - up_blocks.1.attentions.1 group norm alignment
TADSR_UNET_UPBLOCK1_ATTENTION1_SEQUENCE_ALIGNMENT: PASS - up_blocks.1.attentions.1 NCHW to sequence reshape alignment
TADSR_UNET_UPBLOCK1_ATTENTION1_PROJ_IN_ALIGNMENT: PASS - up_blocks.1.attentions.1 proj_in alignment
TADSR_UNET_UPBLOCK1_ATTENTION1_ATTN1_QKV_ALIGNMENT: PASS - up_blocks.1.attentions.1 self-attention q/k/v alignment
TADSR_UNET_UPBLOCK1_ATTENTION1_ATTN1_ALIGNMENT: PASS - up_blocks.1.attentions.1 self-attention output alignment
TADSR_UNET_UPBLOCK1_ATTENTION1_AFTER_ATTN1_ALIGNMENT: PASS - up_blocks.1.attentions.1 residual after attn1 alignment
TADSR_UNET_UPBLOCK1_ATTENTION1_ATTN2_QKV_ALIGNMENT: PASS - up_blocks.1.attentions.1 cross-attention q/k/v alignment
TADSR_UNET_UPBLOCK1_ATTENTION1_ATTN2_ALIGNMENT: PASS - up_blocks.1.attentions.1 cross-attention output alignment
TADSR_UNET_UPBLOCK1_ATTENTION1_AFTER_ATTN2_ALIGNMENT: PASS - up_blocks.1.attentions.1 residual after attn2 alignment
TADSR_UNET_UPBLOCK1_ATTENTION1_FF_ALIGNMENT: PASS - up_blocks.1.attentions.1 feed-forward output alignment
TADSR_UNET_UPBLOCK1_ATTENTION1_TRANSFORMER0_ALIGNMENT: PASS - up_blocks.1.attentions.1 transformer block output alignment
TADSR_UNET_UPBLOCK1_ATTENTION1_ALIGNMENT: PASS - isolated up_blocks.1.attentions.1 output alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT: PASS - UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 -> up_blocks.1.resnets.1 -> up_blocks.1.attentions.1 bridge alignment
TADSR_UNET_UPBLOCK1_ATTENTION1_BRIDGE_ALIGNMENT: PASS - up_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 only
TADSR_UNET_UPBLOCK1_PRE_RESNET2_ALIGNMENT: PASS - input to next unopened up_blocks.1.resnets.2 is now aligned
TADSR_UNET_UPBLOCK1_RESNET2_AUDIT: PASS - official up_blocks.1.resnets.2 audit after up_blocks.1.attentions.1
TADSR_UNET_UPBLOCK1_RESNET2_RESIDUAL_CONTRACT_AUDIT: PASS - official residual pop/concat contract audit after up_blocks.1.attentions.1
TADSR_UNET_UPBLOCK1_RESNET2_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.1.resnets.2 exported
TADSR_UNET_UPBLOCK1_RESNET2_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.1.resnets.2 exported locally
TADSR_UNET_UPBLOCK1_RESNET2_RESIDUAL_CONCAT_ALIGNMENT: PASS - up_blocks.1.resnets.2 hidden/residual concat alignment
TADSR_UNET_UPBLOCK1_RESNET2_NORM1_ALIGNMENT: PASS - up_blocks.1.resnets.2 norm1 alignment
TADSR_UNET_UPBLOCK1_RESNET2_CONV1_ALIGNMENT: PASS - up_blocks.1.resnets.2 conv1 alignment
TADSR_UNET_UPBLOCK1_RESNET2_TEMB_PROJ_ALIGNMENT: PASS - up_blocks.1.resnets.2 time_emb_proj alignment
TADSR_UNET_UPBLOCK1_RESNET2_CONV2_ALIGNMENT: PASS - up_blocks.1.resnets.2 conv2 alignment
TADSR_UNET_UPBLOCK1_RESNET2_ALIGNMENT: PASS - isolated up_blocks.1.resnets.2 after attention1 output and official residual concat alignment
TADSR_UNET_UPBLOCK1_RESNET2_SYNTHETIC_ALIGNMENT: PASS - synthetic hidden/residual -> up_blocks.1.resnets.2 alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ALIGNMENT: PASS - UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 -> up_blocks.1.resnets.1 -> up_blocks.1.attentions.1 -> up_blocks.1.resnets.2 bridge alignment
TADSR_UNET_UPBLOCK1_RESNET2_SHORTCUT_ALIGNMENT: PASS - up_blocks.1.resnets.2 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent
TADSR_UNET_UPBLOCK1_RESNET2_BRIDGE_ALIGNMENT: PASS - up_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 only
TADSR_UNET_UPBLOCK1_PRE_ATTENTION2_ALIGNMENT: PASS - input to next unopened up_blocks.1.attentions.2 is now aligned
TADSR_UNET_UPBLOCK1_ATTENTION2_AUDIT: PASS - official up_blocks.1.attentions.2 topology/LoRA audit
TADSR_UNET_UPBLOCK1_ATTENTION2_RESIDUAL_CONTRACT_AUDIT: PASS - official no-residual-consumption contract audit for up_blocks.1.attentions.2
TADSR_UNET_UPBLOCK1_ATTENTION2_LORA_AUDIT: PASS - LoRA/effective static weight audit for up_blocks.1.attentions.2
TADSR_UNET_UPBLOCK1_ATTENTION2_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.1.attentions.2 exported
TADSR_UNET_UPBLOCK1_ATTENTION2_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.1.attentions.2 exported locally
TADSR_UNET_UPBLOCK1_ATTENTION2_RESIDUAL_CONTRACT_ORACLE: PASS - oracle records attention2 consumes no accumulated residuals
TADSR_UNET_UPBLOCK1_ATTENTION2_NORM_ALIGNMENT: PASS - up_blocks.1.attentions.2 group norm alignment
TADSR_UNET_UPBLOCK1_ATTENTION2_SEQUENCE_ALIGNMENT: PASS - up_blocks.1.attentions.2 NCHW to sequence reshape alignment
TADSR_UNET_UPBLOCK1_ATTENTION2_PROJ_IN_ALIGNMENT: PASS - up_blocks.1.attentions.2 proj_in alignment
TADSR_UNET_UPBLOCK1_ATTENTION2_ATTN1_QKV_ALIGNMENT: PASS - up_blocks.1.attentions.2 self-attention q/k/v alignment
TADSR_UNET_UPBLOCK1_ATTENTION2_ATTN1_ALIGNMENT: PASS - up_blocks.1.attentions.2 self-attention output alignment
TADSR_UNET_UPBLOCK1_ATTENTION2_AFTER_ATTN1_ALIGNMENT: PASS - up_blocks.1.attentions.2 residual after attn1 alignment
TADSR_UNET_UPBLOCK1_ATTENTION2_ATTN2_QKV_ALIGNMENT: PASS - up_blocks.1.attentions.2 cross-attention q/k/v alignment
TADSR_UNET_UPBLOCK1_ATTENTION2_ATTN2_ALIGNMENT: PASS - up_blocks.1.attentions.2 cross-attention output alignment
TADSR_UNET_UPBLOCK1_ATTENTION2_AFTER_ATTN2_ALIGNMENT: PASS - up_blocks.1.attentions.2 residual after attn2 alignment
TADSR_UNET_UPBLOCK1_ATTENTION2_FF_ALIGNMENT: PASS - up_blocks.1.attentions.2 feed-forward output alignment
TADSR_UNET_UPBLOCK1_ATTENTION2_TRANSFORMER0_ALIGNMENT: PASS - up_blocks.1.attentions.2 transformer block output alignment
TADSR_UNET_UPBLOCK1_ATTENTION2_ALIGNMENT: PASS - isolated up_blocks.1.attentions.2 output alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ATTENTION2_ALIGNMENT: BLOCKED - UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 -> up_blocks.1.resnets.1 -> up_blocks.1.attentions.1 -> up_blocks.1.resnets.2 -> up_blocks.1.attentions.2 bridge alignment
TADSR_UNET_UPBLOCK1_ATTENTION2_BRIDGE_ALIGNMENT: PARTIAL - up_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 only
TADSR_UNET_UPBLOCK1_PRE_UPSAMPLER_ALIGNMENT: PARTIAL - input to next unopened up_blocks.1.upsamplers.0 is now aligned
TADSR_UNET_UPBLOCK1_UPSAMPLER_AUDIT: PASS - official up_blocks.1.upsamplers.0 topology/operation audit
TADSR_UNET_UPBLOCK1_UPSAMPLER_LORA_AUDIT: PASS - LoRA/effective static weight audit for up_blocks.1.upsamplers.0
TADSR_UNET_UPBLOCK1_UPSAMPLER_RESIDUAL_CONTRACT_AUDIT: PASS - official upsampler consumes no accumulated residuals
TADSR_UNET_UPBLOCK1_UPSAMPLER_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.1.upsamplers.0 exported
TADSR_UNET_UPBLOCK1_UPSAMPLER_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.1.upsamplers.0 exported locally
TADSR_UNET_UPBLOCK1_UPSAMPLER_RESIDUAL_CONTRACT_ORACLE: PASS - oracle records upsampler consumes no accumulated residuals
TADSR_UNET_UPBLOCK1_UPSAMPLER_INTERPOLATION_ALIGNMENT: PASS - up_blocks.1.upsamplers.0 nearest interpolation alignment
TADSR_UNET_UPBLOCK1_UPSAMPLER_CONV_ALIGNMENT: PASS - up_blocks.1.upsamplers.0 exported effective conv alignment
TADSR_UNET_UPBLOCK1_UPSAMPLER_ALIGNMENT: PASS - isolated up_blocks.1.upsamplers.0 output alignment
TADSR_UNET_UPBLOCK1_UPSAMPLER_SYNTHETIC_ALIGNMENT: PASS - synthetic hidden_states -> up_blocks.1.upsamplers.0 alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_THROUGH_UPSAMPLER_ALIGNMENT: PASS - UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1 through upsamplers.0 bridge alignment
TADSR_UNET_UPBLOCK1_UPSAMPLER_BRIDGE_ALIGNMENT: PARTIAL - up_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 -> upsamplers.0 only; full aggregate is intentionally deferred
TADSR_UNET_UPBLOCK1_LOCAL_FORWARD_AUDIT: PASS - official local up_blocks.1 forward equals manual resnet/attention/upsampler chain
TADSR_UNET_UPBLOCK1_RESIDUAL_CONTRACT_AUDIT: PASS - official local up_blocks.1 residual tuple contract audited
TADSR_UNET_UPBLOCK1_LOCAL_ORACLE_TENSORS: PASS - PyTorch oracle tensors for full local up_blocks.1 aggregate exported
TADSR_UNET_UPBLOCK1_OUTPUT_HIDDEN_ALIGNMENT: PASS - Jittor full local up_blocks.1 hidden output alignment
TADSR_UNET_UPBLOCK1_OUTPUT_STATES_ALIGNMENT: NOT_APPLICABLE - official local up_blocks.1 output_states are not returned in this config, or aligned if present
TADSR_UNET_UPBLOCK1_ALIGNMENT: PARTIAL - full local up_blocks.1 aggregate is aligned through output hidden; execution still stops before up_blocks.2
TADSR_UNET_UPBLOCK2_AUDIT: PASS - official up_blocks.2 topology audited only through resnets.0
TADSR_UNET_UPBLOCK2_RESNET0_AUDIT: PASS - official up_blocks.2.resnets.0 config audited
TADSR_UNET_UPBLOCK2_RESNET0_RESIDUAL_CONTRACT_AUDIT: PASS - official residual consumption for up_blocks.2.resnets.0 audited
TADSR_UNET_UPBLOCK2_RESNET0_LORA_AUDIT: PASS - LoRA/effective static weight audit for up_blocks.2.resnets.0
TADSR_UNET_UPBLOCK2_RESNET0_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.2.resnets.0 exported
TADSR_UNET_UPBLOCK2_RESNET0_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.2.resnets.0 exported locally
TADSR_UNET_UPBLOCK2_RESNET0_RESIDUAL_CONTRACT_ORACLE: PASS - oracle records the exact residual consumed by up_blocks.2.resnets.0
TADSR_UNET_UPBLOCK2_RESNET0_RESIDUAL_CONCAT_ALIGNMENT: PASS - up_blocks.2.resnets.0 residual concat alignment
TADSR_UNET_UPBLOCK2_RESNET0_NORM1_ALIGNMENT: PASS - up_blocks.2.resnets.0 norm1 alignment
TADSR_UNET_UPBLOCK2_RESNET0_CONV1_ALIGNMENT: PASS - up_blocks.2.resnets.0 conv1 alignment
TADSR_UNET_UPBLOCK2_RESNET0_TEMB_PROJ_ALIGNMENT: PASS - up_blocks.2.resnets.0 time embedding projection alignment
TADSR_UNET_UPBLOCK2_RESNET0_CONV2_ALIGNMENT: PASS - up_blocks.2.resnets.0 conv2 alignment
TADSR_UNET_UPBLOCK2_RESNET0_SHORTCUT_ALIGNMENT: PASS - up_blocks.2.resnets.0 shortcut alignment
TADSR_UNET_UPBLOCK2_RESNET0_ALIGNMENT: PASS - isolated entry hidden/residual -> up_blocks.2.resnets.0 output alignment
TADSR_UNET_UPBLOCK2_RESNET0_SYNTHETIC_ALIGNMENT: PASS - synthetic hidden/residual -> up_blocks.2.resnets.0 alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_RESNET0_ALIGNMENT: PASS - UNet entry -> down_blocks -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2.resnets.0 bridge alignment
TADSR_UNET_UPBLOCK2_RESNET0_BRIDGE_ALIGNMENT: PARTIAL - up_blocks.2 is aligned only through resnets.0 and stops before attentions.0
TADSR_UNET_UPBLOCK2_PRE_ATTENTION0_ALIGNMENT: PARTIAL - execution reaches the input boundary before up_blocks.2.attentions.0 without entering attention0
TADSR_UNET_UPBLOCK2_ATTENTION0_AUDIT: PASS - official up_blocks.2.attentions.0 topology/LoRA audit
TADSR_UNET_UPBLOCK2_ATTENTION0_RESIDUAL_CONTRACT_AUDIT: PASS - official no-residual-consumption contract audit for up_blocks.2.attentions.0
TADSR_UNET_UPBLOCK2_ATTENTION0_LORA_AUDIT: PASS - LoRA/effective static weight audit for up_blocks.2.attentions.0
TADSR_UNET_UPBLOCK2_ATTENTION0_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.2.attentions.0 exported
TADSR_UNET_UPBLOCK2_ATTENTION0_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.2.attentions.0 exported locally
TADSR_UNET_UPBLOCK2_ATTENTION0_RESIDUAL_CONTRACT_ORACLE: PASS - oracle records attention0 consumes no accumulated residuals
TADSR_UNET_UPBLOCK2_ATTENTION0_NORM_ALIGNMENT: PASS - up_blocks.2.attentions.0 group norm alignment
TADSR_UNET_UPBLOCK2_ATTENTION0_SEQUENCE_ALIGNMENT: PASS - up_blocks.2.attentions.0 NCHW to sequence reshape alignment
TADSR_UNET_UPBLOCK2_ATTENTION0_PROJ_IN_ALIGNMENT: PASS - up_blocks.2.attentions.0 proj_in alignment
TADSR_UNET_UPBLOCK2_ATTENTION0_ATTN1_QKV_ALIGNMENT: PASS - up_blocks.2.attentions.0 self-attention q/k/v alignment
TADSR_UNET_UPBLOCK2_ATTENTION0_ATTN1_ALIGNMENT: PASS - up_blocks.2.attentions.0 self-attention output alignment
TADSR_UNET_UPBLOCK2_ATTENTION0_AFTER_ATTN1_ALIGNMENT: PASS - up_blocks.2.attentions.0 residual after attn1 alignment
TADSR_UNET_UPBLOCK2_ATTENTION0_ATTN2_QKV_ALIGNMENT: PASS - up_blocks.2.attentions.0 cross-attention q/k/v alignment
TADSR_UNET_UPBLOCK2_ATTENTION0_ATTN2_ALIGNMENT: PASS - up_blocks.2.attentions.0 cross-attention output alignment
TADSR_UNET_UPBLOCK2_ATTENTION0_AFTER_ATTN2_ALIGNMENT: PASS - up_blocks.2.attentions.0 residual after attn2 alignment
TADSR_UNET_UPBLOCK2_ATTENTION0_FF_ALIGNMENT: PASS - up_blocks.2.attentions.0 feed-forward output alignment
TADSR_UNET_UPBLOCK2_ATTENTION0_TRANSFORMER0_ALIGNMENT: PASS - up_blocks.2.attentions.0 transformer block output alignment
TADSR_UNET_UPBLOCK2_ATTENTION0_ALIGNMENT: PASS - isolated up_blocks.2.attentions.0 output alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_RESNET0_ATTENTION0_ALIGNMENT: PASS - UNet entry -> down_blocks -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2.resnets.0 -> up_blocks.2.attentions.0 bridge alignment
TADSR_UNET_UPBLOCK2_ATTENTION0_BRIDGE_ALIGNMENT: PARTIAL - up_blocks.2 is aligned through resnets.0 -> attentions.0 only and stops before resnets.1
TADSR_UNET_UPBLOCK2_PRE_RESNET1_ALIGNMENT: PARTIAL - execution reaches the input boundary before up_blocks.2.resnets.1 without entering resnet1
TADSR_UNET_UPBLOCK2_RESNET1_AUDIT: PASS - official up_blocks.2.resnets.1 config audited
TADSR_UNET_UPBLOCK2_RESNET1_RESIDUAL_CONTRACT_AUDIT: PASS - official residual consumption for up_blocks.2.resnets.1 audited
TADSR_UNET_UPBLOCK2_RESNET1_LORA_AUDIT: PASS - LoRA/effective static weight audit for up_blocks.2.resnets.1
TADSR_UNET_UPBLOCK2_RESNET1_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.2.resnets.1 exported
TADSR_UNET_UPBLOCK2_RESNET1_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.2.resnets.1 exported locally
TADSR_UNET_UPBLOCK2_RESNET1_RESIDUAL_CONTRACT_ORACLE: PASS - oracle records the exact residual consumed by up_blocks.2.resnets.1
TADSR_UNET_UPBLOCK2_RESNET1_RESIDUAL_CONCAT_ALIGNMENT: PASS - up_blocks.2.resnets.1 residual concat alignment
TADSR_UNET_UPBLOCK2_RESNET1_NORM1_ALIGNMENT: PASS - up_blocks.2.resnets.1 norm1 alignment
TADSR_UNET_UPBLOCK2_RESNET1_CONV1_ALIGNMENT: PASS - up_blocks.2.resnets.1 conv1 alignment
TADSR_UNET_UPBLOCK2_RESNET1_TEMB_PROJ_ALIGNMENT: PASS - up_blocks.2.resnets.1 time embedding projection alignment
TADSR_UNET_UPBLOCK2_RESNET1_CONV2_ALIGNMENT: PASS - up_blocks.2.resnets.1 conv2 alignment
TADSR_UNET_UPBLOCK2_RESNET1_SHORTCUT_ALIGNMENT: PASS - up_blocks.2.resnets.1 shortcut alignment
TADSR_UNET_UPBLOCK2_RESNET1_ALIGNMENT: PASS - isolated attention0 hidden/residual -> up_blocks.2.resnets.1 output alignment
TADSR_UNET_UPBLOCK2_RESNET1_SYNTHETIC_ALIGNMENT: PASS - synthetic hidden/residual -> up_blocks.2.resnets.1 alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_RESNET0_ATTENTION0_RESNET1_ALIGNMENT: PASS - UNet entry -> down_blocks -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2.resnets.0 -> up_blocks.2.attentions.0 -> up_blocks.2.resnets.1 bridge alignment
TADSR_UNET_UPBLOCK2_RESNET1_BRIDGE_ALIGNMENT: PARTIAL - up_blocks.2 is aligned through resnets.0 -> attentions.0 -> resnets.1 only and stops before attentions.1
TADSR_UNET_UPBLOCK2_PRE_ATTENTION1_ALIGNMENT: PARTIAL - execution reaches the input boundary before up_blocks.2.attentions.1 without entering attention1
TADSR_UNET_UPBLOCK2_ATTENTION1_AUDIT: PASS - official up_blocks.2.attentions.1 topology/LoRA audit
TADSR_UNET_UPBLOCK2_ATTENTION1_RESIDUAL_CONTRACT_AUDIT: PASS - official no-residual-consumption contract audit for up_blocks.2.attentions.1
TADSR_UNET_UPBLOCK2_ATTENTION1_LORA_AUDIT: PASS - LoRA/effective static weight audit for up_blocks.2.attentions.1
TADSR_UNET_UPBLOCK2_ATTENTION1_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.2.attentions.1 exported
TADSR_UNET_UPBLOCK2_ATTENTION1_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.2.attentions.1 exported locally
TADSR_UNET_UPBLOCK2_ATTENTION1_RESIDUAL_CONTRACT_ORACLE: PASS - oracle records attention1 consumes no accumulated residuals
TADSR_UNET_UPBLOCK2_ATTENTION1_NORM_ALIGNMENT: PASS - up_blocks.2.attentions.1 group norm alignment
TADSR_UNET_UPBLOCK2_ATTENTION1_SEQUENCE_ALIGNMENT: PASS - up_blocks.2.attentions.1 NCHW to sequence reshape alignment
TADSR_UNET_UPBLOCK2_ATTENTION1_PROJ_IN_ALIGNMENT: PASS - up_blocks.2.attentions.1 proj_in alignment
TADSR_UNET_UPBLOCK2_ATTENTION1_ATTN1_QKV_ALIGNMENT: PASS - up_blocks.2.attentions.1 self-attention q/k/v alignment
TADSR_UNET_UPBLOCK2_ATTENTION1_ATTN1_ALIGNMENT: PASS - up_blocks.2.attentions.1 self-attention output alignment
TADSR_UNET_UPBLOCK2_ATTENTION1_AFTER_ATTN1_ALIGNMENT: PASS - up_blocks.2.attentions.1 residual after attn1 alignment
TADSR_UNET_UPBLOCK2_ATTENTION1_ATTN2_QKV_ALIGNMENT: PASS - up_blocks.2.attentions.1 cross-attention q/k/v alignment
TADSR_UNET_UPBLOCK2_ATTENTION1_ATTN2_ALIGNMENT: PASS - up_blocks.2.attentions.1 cross-attention output alignment
TADSR_UNET_UPBLOCK2_ATTENTION1_AFTER_ATTN2_ALIGNMENT: PASS - up_blocks.2.attentions.1 residual after attn2 alignment
TADSR_UNET_UPBLOCK2_ATTENTION1_FF_ALIGNMENT: PASS - up_blocks.2.attentions.1 feed-forward output alignment
TADSR_UNET_UPBLOCK2_ATTENTION1_TRANSFORMER0_ALIGNMENT: PASS - up_blocks.2.attentions.1 transformer block output alignment
TADSR_UNET_UPBLOCK2_ATTENTION1_ALIGNMENT: PASS - isolated up_blocks.2.attentions.1 output alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT: PASS - UNet entry -> down_blocks -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2.resnets.0 -> up_blocks.2.attentions.0 -> up_blocks.2.resnets.1 -> up_blocks.2.attentions.1 bridge alignment
TADSR_UNET_UPBLOCK2_ATTENTION1_BRIDGE_ALIGNMENT: PARTIAL - up_blocks.2 is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 only and stops before resnets.2
TADSR_UNET_UPBLOCK2_PRE_RESNET2_ALIGNMENT: PARTIAL - execution reaches the input boundary before up_blocks.2.resnets.2 without entering resnet2
TADSR_UNET_UPBLOCK2_RESNET2_AUDIT: PASS - official up_blocks.2.resnets.2 config audited
TADSR_UNET_UPBLOCK2_RESNET2_RESIDUAL_CONTRACT_AUDIT: PASS - official residual consumption for up_blocks.2.resnets.2 audited
TADSR_UNET_UPBLOCK2_RESNET2_LORA_AUDIT: PASS - LoRA/effective static weight audit for up_blocks.2.resnets.2
TADSR_UNET_UPBLOCK2_RESNET2_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.2.resnets.2 exported
TADSR_UNET_UPBLOCK2_RESNET2_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.2.resnets.2 exported locally
TADSR_UNET_UPBLOCK2_RESNET2_RESIDUAL_CONTRACT_ORACLE: PASS - oracle records the exact residual consumed by up_blocks.2.resnets.2
TADSR_UNET_UPBLOCK2_RESNET2_RESIDUAL_CONCAT_ALIGNMENT: PASS - up_blocks.2.resnets.2 residual concat alignment
TADSR_UNET_UPBLOCK2_RESNET2_NORM1_ALIGNMENT: PASS - up_blocks.2.resnets.2 norm1 alignment
TADSR_UNET_UPBLOCK2_RESNET2_CONV1_ALIGNMENT: PASS - up_blocks.2.resnets.2 conv1 alignment
TADSR_UNET_UPBLOCK2_RESNET2_TEMB_PROJ_ALIGNMENT: PASS - up_blocks.2.resnets.2 time embedding projection alignment
TADSR_UNET_UPBLOCK2_RESNET2_CONV2_ALIGNMENT: PASS - up_blocks.2.resnets.2 conv2 alignment
TADSR_UNET_UPBLOCK2_RESNET2_SHORTCUT_ALIGNMENT: PASS - up_blocks.2.resnets.2 shortcut alignment
TADSR_UNET_UPBLOCK2_RESNET2_ALIGNMENT: PASS - isolated attention1 hidden/residual -> up_blocks.2.resnets.2 output alignment
TADSR_UNET_UPBLOCK2_RESNET2_SYNTHETIC_ALIGNMENT: PASS - synthetic hidden/residual -> up_blocks.2.resnets.2 alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ALIGNMENT: BLOCKED - UNet entry -> down_blocks -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2.resnets.0 -> up_blocks.2.attentions.0 -> up_blocks.2.resnets.1 -> up_blocks.2.attentions.1 -> up_blocks.2.resnets.2 bridge alignment
TADSR_UNET_UPBLOCK2_RESNET2_BRIDGE_ALIGNMENT: PARTIAL - up_blocks.2 is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 only and stops before attentions.2
TADSR_UNET_UPBLOCK2_PRE_ATTENTION2_ALIGNMENT: PARTIAL - execution reaches the input boundary before up_blocks.2.attentions.2 without entering attention2
TADSR_UNET_UPBLOCK2_ATTENTION2_AUDIT: PASS - official up_blocks.2.attentions.2 topology/LoRA audit
TADSR_UNET_UPBLOCK2_ATTENTION2_RESIDUAL_CONTRACT_AUDIT: PASS - official no-residual-consumption contract audit for up_blocks.2.attentions.2
TADSR_UNET_UPBLOCK2_ATTENTION2_LORA_AUDIT: PASS - LoRA/effective static weight audit for up_blocks.2.attentions.2
TADSR_UNET_UPBLOCK2_ATTENTION2_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.2.attentions.2 exported
TADSR_UNET_UPBLOCK2_ATTENTION2_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.2.attentions.2 exported locally
TADSR_UNET_UPBLOCK2_ATTENTION2_RESIDUAL_CONTRACT_ORACLE: PASS - oracle records attention2 consumes no accumulated residuals
TADSR_UNET_UPBLOCK2_ATTENTION2_NORM_ALIGNMENT: PASS - up_blocks.2.attentions.2 group norm alignment
TADSR_UNET_UPBLOCK2_ATTENTION2_SEQUENCE_ALIGNMENT: PASS - up_blocks.2.attentions.2 NCHW to sequence reshape alignment
TADSR_UNET_UPBLOCK2_ATTENTION2_PROJ_IN_ALIGNMENT: PASS - up_blocks.2.attentions.2 proj_in alignment
TADSR_UNET_UPBLOCK2_ATTENTION2_ATTN1_QKV_ALIGNMENT: PASS - up_blocks.2.attentions.2 self-attention q/k/v alignment
TADSR_UNET_UPBLOCK2_ATTENTION2_ATTN1_ALIGNMENT: PASS - up_blocks.2.attentions.2 self-attention output alignment
TADSR_UNET_UPBLOCK2_ATTENTION2_AFTER_ATTN1_ALIGNMENT: PASS - up_blocks.2.attentions.2 residual after attn1 alignment
TADSR_UNET_UPBLOCK2_ATTENTION2_ATTN2_QKV_ALIGNMENT: PASS - up_blocks.2.attentions.2 cross-attention q/k/v alignment
TADSR_UNET_UPBLOCK2_ATTENTION2_ATTN2_ALIGNMENT: PASS - up_blocks.2.attentions.2 cross-attention output alignment
TADSR_UNET_UPBLOCK2_ATTENTION2_AFTER_ATTN2_ALIGNMENT: PASS - up_blocks.2.attentions.2 residual after attn2 alignment
TADSR_UNET_UPBLOCK2_ATTENTION2_FF_ALIGNMENT: PASS - up_blocks.2.attentions.2 feed-forward output alignment
TADSR_UNET_UPBLOCK2_ATTENTION2_TRANSFORMER0_ALIGNMENT: PASS - up_blocks.2.attentions.2 transformer block output alignment
TADSR_UNET_UPBLOCK2_ATTENTION2_ALIGNMENT: PASS - isolated up_blocks.2.attentions.2 output alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ATTENTION2_ALIGNMENT: BLOCKED - UNet entry -> down_blocks -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2.resnets.0 -> up_blocks.2.attentions.0 -> up_blocks.2.resnets.1 -> up_blocks.2.attentions.1 -> up_blocks.2.resnets.2 -> up_blocks.2.attentions.2 bridge alignment
TADSR_UNET_UPBLOCK2_ATTENTION2_BRIDGE_ALIGNMENT: PARTIAL - up_blocks.2 is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 only and stops before upsamplers.0
TADSR_UNET_UPBLOCK2_PRE_UPSAMPLER_ALIGNMENT: PARTIAL - execution reaches the input boundary before up_blocks.2.upsamplers.0 without entering upsampler
TADSR_UNET_UPBLOCK2_UPSAMPLER_AUDIT: PASS - official up_blocks.2.upsamplers.0 topology/operation audit
TADSR_UNET_UPBLOCK2_UPSAMPLER_LORA_AUDIT: PASS - LoRA/effective static weight audit for up_blocks.2.upsamplers.0
TADSR_UNET_UPBLOCK2_UPSAMPLER_RESIDUAL_CONTRACT_AUDIT: PASS - official upsampler consumes no accumulated residuals
TADSR_UNET_UPBLOCK2_UPSAMPLER_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.2.upsamplers.0 exported
TADSR_UNET_UPBLOCK2_UPSAMPLER_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.2.upsamplers.0 exported locally
TADSR_UNET_UPBLOCK2_UPSAMPLER_RESIDUAL_CONTRACT_ORACLE: PASS - oracle records upsampler consumes no accumulated residuals
TADSR_UNET_UPBLOCK2_UPSAMPLER_INTERPOLATION_ALIGNMENT: PASS - up_blocks.2.upsamplers.0 nearest interpolation alignment
TADSR_UNET_UPBLOCK2_UPSAMPLER_CONV_ALIGNMENT: PASS - up_blocks.2.upsamplers.0 exported effective conv alignment
TADSR_UNET_UPBLOCK2_UPSAMPLER_ALIGNMENT: PASS - isolated up_blocks.2.upsamplers.0 output alignment
TADSR_UNET_UPBLOCK2_UPSAMPLER_SYNTHETIC_ALIGNMENT: PASS - synthetic hidden_states -> up_blocks.2.upsamplers.0 alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_THROUGH_UPSAMPLER_ALIGNMENT: PASS - UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> up_blocks.2 through upsamplers.0 bridge alignment
TADSR_UNET_UPBLOCK2_UPSAMPLER_BRIDGE_ALIGNMENT: PARTIAL - up_blocks.2 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 -> upsamplers.0 only; full aggregate is checked separately
TADSR_UNET_UPBLOCK2_LOCAL_FORWARD_AUDIT: PASS - official full local up_blocks.2 forward output matches manual resnet/attention/upsampler chain
TADSR_UNET_UPBLOCK2_RESIDUAL_CONTRACT_AUDIT: PASS - official residual consumption contract for full local up_blocks.2 audited
TADSR_UNET_UPBLOCK2_LOCAL_ORACLE_TENSORS: PASS - PyTorch oracle tensors for full local up_blocks.2 aggregate exported
TADSR_UNET_UPBLOCK2_OUTPUT_HIDDEN_ALIGNMENT: PASS - full local up_blocks.2 hidden-state output alignment
TADSR_UNET_UPBLOCK2_OUTPUT_STATES_ALIGNMENT: NOT_APPLICABLE - up_blocks.2 output-state contract alignment; official CrossAttnUpBlock2D returns hidden states only in this path
TADSR_UNET_UPBLOCK2_ALIGNMENT: PARTIAL - full local up_blocks.2 aggregate is aligned through output hidden; execution still stops before up_blocks.3
TADSR_UNET_UPBLOCK3_AUDIT: PASS - official up_blocks.3 topology audit
TADSR_UNET_UPBLOCK3_RESNET0_AUDIT: PASS - official up_blocks.3.resnets.0 module audit
TADSR_UNET_UPBLOCK3_RESNET0_RESIDUAL_CONTRACT_AUDIT: PASS - official residual consumption contract for up_blocks.3.resnets.0
TADSR_UNET_UPBLOCK3_RESNET0_LORA_AUDIT: PASS - LoRA/effective static weight audit for up_blocks.3.resnets.0
TADSR_UNET_UPBLOCK3_RESNET0_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.3.resnets.0 exported
TADSR_UNET_UPBLOCK3_RESNET0_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.3.resnets.0 exported locally
TADSR_UNET_UPBLOCK3_RESNET0_RESIDUAL_CONTRACT_ORACLE: PASS - oracle records up_blocks.3.resnets.0 residual pop/concat contract
TADSR_UNET_UPBLOCK3_RESNET0_RESIDUAL_CONCAT_ALIGNMENT: PASS - up_blocks.3.resnets.0 residual concat alignment
TADSR_UNET_UPBLOCK3_RESNET0_NORM1_ALIGNMENT: PASS - up_blocks.3.resnets.0 norm1 alignment
TADSR_UNET_UPBLOCK3_RESNET0_CONV1_ALIGNMENT: PASS - up_blocks.3.resnets.0 conv1 alignment
TADSR_UNET_UPBLOCK3_RESNET0_TEMB_PROJ_ALIGNMENT: PASS - up_blocks.3.resnets.0 time embedding projection alignment
TADSR_UNET_UPBLOCK3_RESNET0_CONV2_ALIGNMENT: PASS - up_blocks.3.resnets.0 conv2 alignment
TADSR_UNET_UPBLOCK3_RESNET0_SHORTCUT_ALIGNMENT: PASS - up_blocks.3.resnets.0 shortcut alignment
TADSR_UNET_UPBLOCK3_RESNET0_ALIGNMENT: PASS - isolated up_blocks.3.resnets.0 output alignment
TADSR_UNET_UPBLOCK3_RESNET0_SYNTHETIC_ALIGNMENT: PASS - synthetic hidden/residual/temb -> up_blocks.3.resnets.0 alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_UPBLOCK3_RESNET0_ALIGNMENT: PASS - UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 bridge alignment
TADSR_UNET_UPBLOCK3_RESNET0_BRIDGE_ALIGNMENT: PARTIAL - up_blocks.3 path is aligned through resnets.0 only and deliberately stops before the next up_blocks.3 module
TADSR_UNET_UPBLOCK3_PRE_ATTENTION0_ALIGNMENT: PARTIAL - execution reaches the input boundary before up_blocks.3.attentions.0 without entering attention0
TADSR_UNET_UPBLOCK3_ATTENTION0_AUDIT: PASS - official up_blocks.3.attentions.0 top-level audit
TADSR_UNET_UPBLOCK3_ATTENTION0_TRANSFORMER_AUDIT: PASS - official transformer block audit for up_blocks.3.attentions.0
TADSR_UNET_UPBLOCK3_ATTENTION0_LORA_AUDIT: PASS - LoRA/effective static weight audit for up_blocks.3.attentions.0
TADSR_UNET_UPBLOCK3_ATTENTION0_RESIDUAL_CONTRACT_AUDIT: PASS - official residual contract: attention0 consumes no residual
TADSR_UNET_UPBLOCK3_ATTENTION0_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.3.attentions.0 exported
TADSR_UNET_UPBLOCK3_ATTENTION0_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.3.attentions.0 exported locally
TADSR_UNET_UPBLOCK3_ATTENTION0_RESIDUAL_CONTRACT_ORACLE: PASS - oracle records unchanged residual tuple across up_blocks.3.attentions.0
TADSR_UNET_UPBLOCK3_ATTENTION0_NORM_ALIGNMENT: PASS - up_blocks.3.attentions.0 top norm alignment
TADSR_UNET_UPBLOCK3_ATTENTION0_PROJ_IN_ALIGNMENT: PASS - up_blocks.3.attentions.0 proj_in alignment
TADSR_UNET_UPBLOCK3_ATTENTION0_SEQUENCE_ALIGNMENT: PASS - up_blocks.3.attentions.0 NCHW-to-sequence contract alignment
TADSR_UNET_UPBLOCK3_ATTENTION0_ATTN1_QKV_ALIGNMENT: PASS - up_blocks.3.attentions.0 self-attention Q/K/V alignment
TADSR_UNET_UPBLOCK3_ATTENTION0_ATTN1_ALIGNMENT: PASS - up_blocks.3.attentions.0 self-attention output alignment
TADSR_UNET_UPBLOCK3_ATTENTION0_AFTER_ATTN1_ALIGNMENT: PASS - up_blocks.3.attentions.0 residual after self-attention alignment
TADSR_UNET_UPBLOCK3_ATTENTION0_ATTN2_QKV_ALIGNMENT: PASS - up_blocks.3.attentions.0 cross-attention Q/K/V alignment
TADSR_UNET_UPBLOCK3_ATTENTION0_ATTN2_ALIGNMENT: PASS - up_blocks.3.attentions.0 cross-attention output alignment
TADSR_UNET_UPBLOCK3_ATTENTION0_AFTER_ATTN2_ALIGNMENT: PASS - up_blocks.3.attentions.0 residual after cross-attention alignment
TADSR_UNET_UPBLOCK3_ATTENTION0_FF_ALIGNMENT: PASS - up_blocks.3.attentions.0 feed-forward alignment
TADSR_UNET_UPBLOCK3_ATTENTION0_TRANSFORMER0_ALIGNMENT: PASS - up_blocks.3.attentions.0 transformer block output alignment
TADSR_UNET_UPBLOCK3_ATTENTION0_ALIGNMENT: PASS - isolated up_blocks.3.attentions.0 output alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_UPBLOCK3_RESNET0_ATTENTION0_ALIGNMENT: PASS - UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 -> up_blocks.3.attentions.0 bridge alignment
TADSR_UNET_UPBLOCK3_ATTENTION0_BRIDGE_ALIGNMENT: PARTIAL - up_blocks.3 path is aligned through attentions.0 only and deliberately stops before the next up_blocks.3 module
TADSR_UNET_UPBLOCK3_PRE_RESNET1_ALIGNMENT: PARTIAL - execution reaches the input boundary before up_blocks.3.resnets.1 without entering resnet1
TADSR_UNET_UPBLOCK3_RESNET1_AUDIT: PASS - official up_blocks.3.resnets.1 module audit
TADSR_UNET_UPBLOCK3_RESNET1_RESIDUAL_CONTRACT_AUDIT: PASS - official residual consumption contract for up_blocks.3.resnets.1
TADSR_UNET_UPBLOCK3_RESNET1_LORA_AUDIT: PASS - LoRA/effective static weight audit for up_blocks.3.resnets.1
TADSR_UNET_UPBLOCK3_RESNET1_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.3.resnets.1 exported
TADSR_UNET_UPBLOCK3_RESNET1_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.3.resnets.1 exported locally
TADSR_UNET_UPBLOCK3_RESNET1_RESIDUAL_CONTRACT_ORACLE: PASS - oracle records up_blocks.3.resnets.1 residual pop/concat contract
TADSR_UNET_UPBLOCK3_RESNET1_RESIDUAL_CONCAT_ALIGNMENT: PASS - up_blocks.3.resnets.1 residual concat alignment
TADSR_UNET_UPBLOCK3_RESNET1_NORM1_ALIGNMENT: PASS - up_blocks.3.resnets.1 norm1 alignment
TADSR_UNET_UPBLOCK3_RESNET1_CONV1_ALIGNMENT: PASS - up_blocks.3.resnets.1 conv1 alignment
TADSR_UNET_UPBLOCK3_RESNET1_TEMB_PROJ_ALIGNMENT: PASS - up_blocks.3.resnets.1 time embedding projection alignment
TADSR_UNET_UPBLOCK3_RESNET1_CONV2_ALIGNMENT: PASS - up_blocks.3.resnets.1 conv2 alignment
TADSR_UNET_UPBLOCK3_RESNET1_SHORTCUT_ALIGNMENT: PASS - up_blocks.3.resnets.1 shortcut alignment
TADSR_UNET_UPBLOCK3_RESNET1_ALIGNMENT: PASS - isolated up_blocks.3.resnets.1 output alignment
TADSR_UNET_UPBLOCK3_RESNET1_SYNTHETIC_ALIGNMENT: PASS - synthetic hidden/residual/temb -> up_blocks.3.resnets.1 alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_UPBLOCK3_RESNET0_ATTENTION0_RESNET1_ALIGNMENT: PASS - UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 -> up_blocks.3.attentions.0 -> up_blocks.3.resnets.1 bridge alignment
TADSR_UNET_UPBLOCK3_RESNET1_BRIDGE_ALIGNMENT: PARTIAL - up_blocks.3 path is aligned through resnets.0 -> attentions.0 -> resnets.1 only and deliberately stops before the next up_blocks.3 module
TADSR_UNET_UPBLOCK3_PRE_ATTENTION1_ALIGNMENT: PARTIAL - execution reaches the input boundary before up_blocks.3.attentions.1 without entering attention1
TADSR_UNET_UPBLOCK3_ATTENTION1_AUDIT: PASS - official up_blocks.3.attentions.1 top-level audit
TADSR_UNET_UPBLOCK3_ATTENTION1_TRANSFORMER_AUDIT: PASS - official transformer block audit for up_blocks.3.attentions.1
TADSR_UNET_UPBLOCK3_ATTENTION1_LORA_AUDIT: PASS - LoRA/effective static weight audit for up_blocks.3.attentions.1
TADSR_UNET_UPBLOCK3_ATTENTION1_RESIDUAL_CONTRACT_AUDIT: PASS - official residual contract: attention1 consumes no residual
TADSR_UNET_UPBLOCK3_ATTENTION1_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.3.attentions.1 exported
TADSR_UNET_UPBLOCK3_ATTENTION1_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.3.attentions.1 exported locally
TADSR_UNET_UPBLOCK3_ATTENTION1_RESIDUAL_CONTRACT_ORACLE: PASS - oracle records unchanged residual tuple across up_blocks.3.attentions.1
TADSR_UNET_UPBLOCK3_ATTENTION1_NORM_ALIGNMENT: PASS - up_blocks.3.attentions.1 top norm alignment
TADSR_UNET_UPBLOCK3_ATTENTION1_PROJ_IN_ALIGNMENT: PASS - up_blocks.3.attentions.1 proj_in alignment
TADSR_UNET_UPBLOCK3_ATTENTION1_SEQUENCE_ALIGNMENT: PASS - up_blocks.3.attentions.1 NCHW-to-sequence contract alignment
TADSR_UNET_UPBLOCK3_ATTENTION1_ATTN1_QKV_ALIGNMENT: PASS - up_blocks.3.attentions.1 self-attention Q/K/V alignment
TADSR_UNET_UPBLOCK3_ATTENTION1_ATTN1_ALIGNMENT: PASS - up_blocks.3.attentions.1 self-attention output alignment
TADSR_UNET_UPBLOCK3_ATTENTION1_AFTER_ATTN1_ALIGNMENT: PASS - up_blocks.3.attentions.1 residual after self-attention alignment
TADSR_UNET_UPBLOCK3_ATTENTION1_ATTN2_QKV_ALIGNMENT: PASS - up_blocks.3.attentions.1 cross-attention Q/K/V alignment
TADSR_UNET_UPBLOCK3_ATTENTION1_ATTN2_ALIGNMENT: PASS - up_blocks.3.attentions.1 cross-attention output alignment
TADSR_UNET_UPBLOCK3_ATTENTION1_AFTER_ATTN2_ALIGNMENT: PASS - up_blocks.3.attentions.1 residual after cross-attention alignment
TADSR_UNET_UPBLOCK3_ATTENTION1_FF_ALIGNMENT: PASS - up_blocks.3.attentions.1 feed-forward alignment
TADSR_UNET_UPBLOCK3_ATTENTION1_TRANSFORMER0_ALIGNMENT: PASS - up_blocks.3.attentions.1 transformer block output alignment
TADSR_UNET_UPBLOCK3_ATTENTION1_ALIGNMENT: PASS - isolated up_blocks.3.attentions.1 output alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_UPBLOCK3_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT: BLOCKED - UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 -> up_blocks.3.attentions.0 -> up_blocks.3.resnets.1 -> up_blocks.3.attentions.1 bridge alignment
TADSR_UNET_UPBLOCK3_ATTENTION1_BRIDGE_ALIGNMENT: PARTIAL - up_blocks.3 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 only and deliberately stops before the next up_blocks.3 module
TADSR_UNET_UPBLOCK3_PRE_RESNET2_ALIGNMENT: PARTIAL - execution reaches the input boundary before up_blocks.3.resnets.2 without entering resnet2
TADSR_UNET_UPBLOCK3_RESNET2_AUDIT: PASS - official up_blocks.3.resnets.2 module audit
TADSR_UNET_UPBLOCK3_RESNET2_RESIDUAL_CONTRACT_AUDIT: PASS - official residual consumption contract for up_blocks.3.resnets.2
TADSR_UNET_UPBLOCK3_RESNET2_LORA_AUDIT: PASS - LoRA/effective static weight audit for up_blocks.3.resnets.2
TADSR_UNET_UPBLOCK3_RESNET2_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.3.resnets.2 exported
TADSR_UNET_UPBLOCK3_RESNET2_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.3.resnets.2 exported locally
TADSR_UNET_UPBLOCK3_RESNET2_RESIDUAL_CONTRACT_ORACLE: PASS - oracle records up_blocks.3.resnets.2 residual pop/concat contract
TADSR_UNET_UPBLOCK3_RESNET2_RESIDUAL_CONCAT_ALIGNMENT: PASS - up_blocks.3.resnets.2 residual concat alignment
TADSR_UNET_UPBLOCK3_RESNET2_NORM1_ALIGNMENT: PASS - up_blocks.3.resnets.2 norm1 alignment
TADSR_UNET_UPBLOCK3_RESNET2_CONV1_ALIGNMENT: PASS - up_blocks.3.resnets.2 conv1 alignment
TADSR_UNET_UPBLOCK3_RESNET2_TEMB_PROJ_ALIGNMENT: PASS - up_blocks.3.resnets.2 time embedding projection alignment
TADSR_UNET_UPBLOCK3_RESNET2_CONV2_ALIGNMENT: PASS - up_blocks.3.resnets.2 conv2 alignment
TADSR_UNET_UPBLOCK3_RESNET2_SHORTCUT_ALIGNMENT: PASS - up_blocks.3.resnets.2 shortcut alignment
TADSR_UNET_UPBLOCK3_RESNET2_ALIGNMENT: PASS - isolated up_blocks.3.resnets.2 output alignment
TADSR_UNET_UPBLOCK3_RESNET2_SYNTHETIC_ALIGNMENT: PASS - synthetic hidden/residual/temb -> up_blocks.3.resnets.2 alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_UPBLOCK3_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ALIGNMENT: BLOCKED - UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 -> up_blocks.3.attentions.0 -> up_blocks.3.resnets.1 -> up_blocks.3.attentions.1 -> up_blocks.3.resnets.2 bridge alignment
TADSR_UNET_UPBLOCK3_RESNET2_BRIDGE_ALIGNMENT: PARTIAL - up_blocks.3 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 only and deliberately stops before the next up_blocks.3 module
TADSR_UNET_UPBLOCK3_PRE_ATTENTION2_ALIGNMENT: PARTIAL - execution reaches the input boundary before up_blocks.3.attentions.2 without entering attention2
TADSR_UNET_UPBLOCK3_ATTENTION2_AUDIT: PASS - official up_blocks.3.attentions.2 module audit
TADSR_UNET_UPBLOCK3_ATTENTION2_TRANSFORMER_AUDIT: PASS - official Transformer2DModel/BasicTransformerBlock audit for up_blocks.3.attentions.2
TADSR_UNET_UPBLOCK3_ATTENTION2_LORA_AUDIT: PASS - LoRA/effective static weight audit for up_blocks.3.attentions.2
TADSR_UNET_UPBLOCK3_ATTENTION2_RESIDUAL_CONTRACT_AUDIT: PASS - official residual non-consumption contract for up_blocks.3.attentions.2
TADSR_UNET_UPBLOCK3_ATTENTION2_ORACLE_TENSORS: PASS - PyTorch oracle tensors for up_blocks.3.attentions.2 exported
TADSR_UNET_UPBLOCK3_ATTENTION2_EFFECTIVE_WEIGHTS: PASS - effective static weights for up_blocks.3.attentions.2 exported locally
TADSR_UNET_UPBLOCK3_ATTENTION2_RESIDUAL_CONTRACT_ORACLE: PASS - oracle records up_blocks.3.attentions.2 residual non-consumption contract
TADSR_UNET_UPBLOCK3_ATTENTION2_NORM_ALIGNMENT: PASS - attention2 top-level GroupNorm alignment
TADSR_UNET_UPBLOCK3_ATTENTION2_PROJ_IN_ALIGNMENT: PASS - attention2 proj_in alignment
TADSR_UNET_UPBLOCK3_ATTENTION2_SEQUENCE_ALIGNMENT: PASS - attention2 NCHW-to-sequence alignment
TADSR_UNET_UPBLOCK3_ATTENTION2_ATTN1_QKV_ALIGNMENT: PASS - attention2 transformer0 self-attention QKV alignment
TADSR_UNET_UPBLOCK3_ATTENTION2_ATTN1_ALIGNMENT: PASS - attention2 transformer0 self-attention output alignment
TADSR_UNET_UPBLOCK3_ATTENTION2_AFTER_ATTN1_ALIGNMENT: PASS - attention2 transformer0 after-attn1 residual alignment
TADSR_UNET_UPBLOCK3_ATTENTION2_ATTN2_QKV_ALIGNMENT: PASS - attention2 transformer0 cross-attention QKV alignment
TADSR_UNET_UPBLOCK3_ATTENTION2_ATTN2_ALIGNMENT: PASS - attention2 transformer0 cross-attention output alignment
TADSR_UNET_UPBLOCK3_ATTENTION2_AFTER_ATTN2_ALIGNMENT: PASS - attention2 transformer0 after-attn2 residual alignment
TADSR_UNET_UPBLOCK3_ATTENTION2_FF_ALIGNMENT: PASS - attention2 feed-forward output alignment
TADSR_UNET_UPBLOCK3_ATTENTION2_TRANSFORMER0_ALIGNMENT: PASS - attention2 full transformer block alignment
TADSR_UNET_UPBLOCK3_ATTENTION2_ALIGNMENT: PASS - isolated up_blocks.3.attentions.2 output alignment
TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_UPBLOCK3_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ATTENTION2_ALIGNMENT: BLOCKED - UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 bridge alignment
TADSR_UNET_UPBLOCK3_ATTENTION2_BRIDGE_ALIGNMENT: PARTIAL - up_blocks.3 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 and deliberately stops before output tail/full local aggregate
TADSR_UNET_UPBLOCK3_PRE_LOCAL_AGGREGATE_ALIGNMENT: PARTIAL - execution reaches the boundary before output tail / full local up_blocks.3 aggregate without entering full UNet forward
TADSR_UNET_UPBLOCK3_LOCAL_FORWARD_AUDIT: PASS - official local up_blocks.3 forward agrees with the audited manual resnet/attention chain
TADSR_UNET_UPBLOCK3_RESIDUAL_CONTRACT_AUDIT: PASS - official up_blocks.3 local residual tuple consumption contract audited
TADSR_UNET_UPBLOCK3_LOCAL_ORACLE_TENSORS: PASS - PyTorch oracle tensors for full local up_blocks.3 aggregate exported
TADSR_UNET_UPBLOCK3_RESIDUAL_CONTRACT_ORACLE: PASS - oracle records up_blocks.3 residual consumption and zero residuals after attention2
TADSR_UNET_OUTPUT_TAIL_BOUNDARY_AUDIT: PASS - official output-tail boundary after up_blocks.3 audited without executing the tail
TADSR_UNET_UPBLOCK3_OUTPUT_HIDDEN_ALIGNMENT: PASS - Jittor full local up_blocks.3 hidden output matches the PyTorch local oracle
TADSR_UNET_UPBLOCK3_OUTPUT_STATES_ALIGNMENT: NOT_APPLICABLE - official up_blocks.3 output_states contract; NOT_APPLICABLE if it returns a hidden tensor only
TADSR_UNET_OUTPUT_TAIL_BOUNDARY_ALIGNMENT: PASS - Jittor bridge stops at the audited output-tail boundary without running conv_norm_out/conv_act/conv_out
TADSR_UNET_UPBLOCK3_ALIGNMENT: PARTIAL - up_blocks.3 leaves plus full local aggregate hidden output are aligned and the output-tail boundary is audited
TADSR_UNET_OUTPUT_TAIL_AUDIT: PASS - official output tail conv_norm_out -> conv_act -> conv_out audit
TADSR_UNET_OUTPUT_TAIL_TOPOLOGY_AUDIT: PASS - official output tail topology/config audit
TADSR_UNET_OUTPUT_TAIL_LORA_AUDIT: PASS - output tail conv_out LoRA/effective static weight audit
TADSR_UNET_OUTPUT_TAIL_LOCAL_EXECUTION_AUDIT: PASS - official local output tail execution without official unet.forward
TADSR_UNET_OUTPUT_TAIL_ORACLE_TENSORS: PASS - PyTorch oracle tensors for output tail exported
TADSR_UNET_OUTPUT_TAIL_EFFECTIVE_WEIGHTS: PASS - conv_norm_out raw affine parameters and conv_out LoRA-merged effective weights exported
TADSR_UNET_OUTPUT_TAIL_NORM_ALIGNMENT: PASS - conv_norm_out GroupNorm alignment
TADSR_UNET_OUTPUT_TAIL_ACT_ALIGNMENT: PASS - conv_act SiLU alignment
TADSR_UNET_OUTPUT_TAIL_CONV_OUT_ALIGNMENT: PASS - conv_out LoRA-merged effective convolution alignment
TADSR_UNET_OUTPUT_TAIL_ALIGNMENT: PASS - isolated output tail alignment from the PyTorch up_blocks.3 output tensor
TADSR_UNET_OUTPUT_TAIL_SYNTHETIC_ALIGNMENT: PASS - synthetic hidden tensor -> output tail alignment
TADSR_UNET_ENTRY_TO_OUTPUT_TAIL_ALIGNMENT: PASS - entry -> all down_blocks -> full local mid_block -> full local up_blocks.0/1/2/3 -> output tail bridge alignment
TADSR_UNET_MANUAL_BLOCKS_TO_TAIL_ALIGNMENT: PASS - manual block composition through output tail is aligned without official UNet.forward
TADSR_UNET_MANUAL_FULL_WRAPPER_AUDIT: PASS - official manual wrapper contract audit without official UNet.forward
TADSR_UNET_MANUAL_FULL_WRAPPER_CONTRACT_AUDIT: PASS - manual wrapper input/output contract audit
TADSR_UNET_MANUAL_FULL_CHAIN_TO_TAIL_AUDIT: PASS - official module chain audit through output tail without full forward
TADSR_UNET_MANUAL_FULL_WRAPPER_ORACLE_TENSORS: PASS - PyTorch oracle tensors for alignment-only manual full-chain wrapper exported
TADSR_UNET_MANUAL_WRAPPER_ENTRY_ALIGNMENT: PASS - manual wrapper entry path alignment
TADSR_UNET_MANUAL_WRAPPER_DOWNBLOCKS_ALIGNMENT: PASS - manual wrapper down_blocks alignment
TADSR_UNET_MANUAL_WRAPPER_MIDBLOCK_ALIGNMENT: PASS - manual wrapper mid_block alignment
TADSR_UNET_MANUAL_WRAPPER_UPBLOCKS_ALIGNMENT: PASS - manual wrapper up_blocks.0/1/2/3 alignment
TADSR_UNET_MANUAL_WRAPPER_OUTPUT_TAIL_ALIGNMENT: PASS - manual wrapper output-tail alignment
TADSR_UNET_MANUAL_FULL_WRAPPER_ALIGNMENT: PASS - alignment-only manual UNet chain wrapper output matches the PyTorch manual oracle
TADSR_UNET_MANUAL_FULL_CHAIN_ALIGNMENT: PASS - manual full chain from UNet inputs through output tail is aligned; this is not official full forward
TADSR_UNET_OFFICIAL_FULL_FORWARD_AUDIT: PASS - official PyTorch UNet.forward contract audit
TADSR_UNET_FULL_FORWARD_CONTRACT_AUDIT: PASS - official full forward input/output and return contract audit
TADSR_UNET_MANUAL_VS_OFFICIAL_FULL_FORWARD_AUDIT: PASS - PyTorch manual wrapper output matches official full forward output
TADSR_UNET_OFFICIAL_FULL_FORWARD_ORACLE_TENSORS: PASS - official full forward oracle tensors exported
TADSR_UNET_MANUAL_VS_OFFICIAL_FULL_FORWARD_ALIGNMENT: PASS - manual wrapper oracle matches official full forward oracle
TADSR_UNET_JITTOR_VS_OFFICIAL_FULL_FORWARD_ALIGNMENT: PASS - Jittor alignment-only full forward output matches official PyTorch UNet.forward oracle
TADSR_UNET_JITTOR_VS_MANUAL_WRAPPER_ALIGNMENT: PASS - Jittor alignment-only full forward output matches PyTorch manual wrapper oracle
TADSR_UNET_FULL_FORWARD_RETURN_CONTRACT_ALIGNMENT: PASS - alignment-only return_dict/tensor contract matches the test contract
TADSR_UNET_UPBLOCKS_ALIGNMENT: NOT_COMPLETE - UNet up_blocks.0/1/2/3 local aggregate path is aligned; output tail is checked separately and full UNet forward remains unopened
TADSR_UNET_CROSS_ATTENTION_ALIGNMENT: NOT_COMPLETE - all local UNet cross-attention modules through up_blocks.3 are aligned; global full-forward integration remains incomplete until full forward wrapper is compared
TADSR_UNET_FULL_FORWARD_ALIGNMENT: PASS - UNet full forward numerical alignment against official PyTorch UNet.forward; not full TADSR inference
TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL - LoRA-bearing modules are exported as static effective weights; generic runtime LoRA is not implemented
TADSR_LORA_POLICY_AUDIT: PASS - official TADSR LoRA/PEFT source usage, active module inventory, and inference-time adapter policy audited
TADSR_STATIC_EFFECTIVE_LORA_POLICY_AUDIT: PASS - static effective weights are the selected alignment policy when official inference uses fixed active adapters
TADSR_RUNTIME_DYNAMIC_LORA_REQUIREMENT_AUDIT: PASS - generic runtime LoRA is required only if official inference dynamically changes adapters or scale
TADSR_LORA_MODULE_INVENTORY_AUDIT: PASS - active LoRA A/B module inventory from the official TADSR checkpoint and source policy audit
TIME_VAE_LORA_EFFECTIVE_WEIGHTS_AUDIT: PASS - official active TimeVAE LoRA pair inventory is exported as static effective weights
TIME_VAE_LORA_EFFECTIVE_WEIGHTS_EXPORT: PASS - converted_timevae_lora_effective_weights.npz exists as a local git-ignored static effective-weight artifact
TIME_VAE_LORA_EFFECTIVE_WEIGHT_MANUAL_VERIFY: PASS - official active LoRA module forward is compared with manual effective-weight forward for each TimeVAE pair
TIME_VAE_LORA_EFFECTIVE_ARTIFACT_COVERAGE: PASS - all active TimeVAE LoRA pairs have static effective-weight artifact metadata and verification evidence
TIME_VAE_ACTIVE_LORA_MODULE_COVERAGE: PASS - TimeVAE active LoRA module coverage in the project-wide static effective-weight policy audit
TADSR_STATIC_EFFECTIVE_LORA_COVERAGE_AUDIT: PASS - Jittor static effective-weight artifacts are checked against official active LoRA modules
TADSR_ACTIVE_LORA_MODULE_COVERAGE: PASS - active official LoRA modules covered by exported effective weights or explicitly reported as partial/missing
TADSR_EFFECTIVE_WEIGHT_ARTIFACT_COVERAGE: PASS - effective-weight artifacts exist and large NPZ files remain local artifacts rather than committed payloads
TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION: NOT_IMPLEMENTED_BY_DESIGN - generic runtime LoRA adapter loading/switching/scale is deferred and not claimed complete
TADSR_LORA_POLICY_CONTRACT_TEST: PASS - metadata-only contract test for LoRA policy/coverage evidence and full-inference guard
TIME_VAE_LORA_EFFECTIVE_WEIGHT_COVERAGE_TEST: PASS - metadata-only contract test for TimeVAE LoRA effective artifact coverage and full-inference guard
TIME_VAE_ENCODER_TO_QUANT_ALIGNMENT: PASS - TimeAware VAE encoder-side path to quant_conv aligned; deterministic decoder stack through tail aligned separately
TIME_VAE_FULL_ALIGNMENT: NOT_COMPLETE - TimeVAE full-boundary alignment is PASS, but full TimeVAE alignment remains NOT_COMPLETE because audit metadata says the boundary is not sufficient for full TADSR pipeline: VAEHook tiling, internal stochastic sampling policy, runtime LoRA, and full inference remain unopened.
JITTOR_FULL_INFERENCE: NOT_COMPLETE - full VAE/UNet/LoRA inference is intentionally NotImplemented
JITTOR_MIGRATION_REPORT: PASS - migration evidence report exists
JITTOR_FULL_PORT: PARTIAL - skeleton/check CLI only; full VAE/UNet not complete
FINAL_README: PASS - README has Jittor migration evidence sections
TADSR_FINAL_EVIDENCE_MANIFEST: PASS - final evidence manifest indexes committed audit/report files
TADSR_PRODUCTION_CLI_DESIGN_AUDIT: PASS - production CLI design memo exists and keeps full inference guarded
TADSR_FINAL_SUBMISSION_CHECKLIST: PASS - final submission checklist exists with status, commands, evidence and limitations
TADSR_REPORTING_READINESS_AUDIT: PASS - README/docs/report consistently document completed boundaries and honest gaps
TADSR_LARGE_ARTIFACT_POLICY_AUDIT: PASS - large tensor artifacts are ignored; metadata/report files are committed
TADSR_FINAL_PACKAGING_READINESS: PASS - final evidence package is ready for presentation/repository handoff while full inference remains guarded
TADSR_FINAL_PRESENTATION_PACKAGE: PASS - final PPT outline and presentation handoff docs are ready
TADSR_VIDEO_SCRIPT_READY: PASS - final video script is ready and avoids full-inference/image-generation claims
TADSR_DEMO_RUNBOOK_READY: PASS - demo runbook lists reproducible audit/alignment commands and expected markers
TADSR_REPOSITORY_HANDOFF_READY: PASS - repository handoff guide maps code, evidence, large-file policy and limitations
TADSR_FINAL_HANDOFF_READINESS: PASS - repository is ready for final presentation/video recording and submission handoff
TADSR_FINAL_PPT_READY: PASS - final PPTX presentation exists and is non-empty
TADSR_FINAL_PDF_READY: PASS - final PDF presentation export exists and is non-empty
TADSR_FINAL_VIDEO_RECORDING_PACKAGE_READY: PASS - video recording guide exists with safe demo commands and honest limitations
TADSR_FINAL_SUBMISSION_README_READY: PASS - submission README exists and indexes final deliverables/evidence
TADSR_FINAL_DELIVERABLE_SIZE_AUDIT: PASS - final deliverables stay below the 100 MB submission budget
TADSR_FINAL_DELIVERABLES_INCLUDE_SMOKE_TRAINING: PASS - final PPT/video/submission README include small-data PyTorch-vs-Jittor smoke-training evidence
TADSR_FINAL_PPT_INCLUDES_SMOKE_TRAINING: PASS - final PPT Markdown/PPTX source includes the small-data smoke-training evidence slide
TADSR_VIDEO_SCRIPT_INCLUDES_SMOKE_TRAINING: PASS - final video script/recording guide includes the smoke-training segment and demo command
TADSR_SUBMISSION_README_INCLUDES_SMOKE_TRAINING: PASS - submission README indexes smoke-training logs, loss curves, prediction visualizations and multi-seed evidence
TADSR_FINAL_DELIVERABLES_READY: PASS - PPT/PDF/video guide/submission README are ready and free of misleading full-inference claims
TADSR_FINAL_SUBMISSION_READY: PASS - final deliverables are ready for submission while full inference remains explicitly guarded
TADSR_FINAL_SUBMISSION_CONTENT_VALIDATION: PASS - PPT/video/submission wording contains required final markers and avoids misleading claims
TADSR_FINAL_PPT_CONTENT_GUARDRAIL: PASS - PPT Markdown and outline preserve honest full-inference limitations
TADSR_FINAL_VIDEO_SCRIPT_GUARDRAIL: PASS - video script and recording guide avoid full-inference/image-generation/runtime-LoRA over-claims
TADSR_FINAL_SUBMISSION_README_GUARDRAIL: PASS - submission README indexes evidence and keeps guarded-scope wording
TADSR_REPOSITORY_HANDOFF_VALIDATION: PASS - repository handoff files, git readability and large artifact policy are validated
TADSR_GITHUB_SUBMISSION_GUIDE_READY: PASS - GitHub handoff guide exists with remote/push commands and no automatic push policy
TADSR_TRACKED_LARGE_ARTIFACT_AUDIT: PASS - no new, staged or out-of-policy .npy/.npz files are present; historical oracle tensors are separately reported
TADSR_VIDEO_RECORDING_PREFLIGHT_CHECKLIST: PASS - recording checklist exists with windows, commands, limits and timing
TADSR_RELEASE_CANDIDATE_QA_READINESS: PASS - release-candidate QA package is ready for human GitHub handoff and video recording
TADSR_SMOKE_TRAINING_DATA_PREP: PASS - deterministic output-tail conv_out feature-target pairs metadata exists; raw .npy tensors stay ignored
TADSR_SMOKE_TRAINING_TRAIN_VAL_SPLIT: PASS - smoke training data has a deterministic 24/8 train-validation split over 32 output-tail samples
TADSR_PYTORCH_SMOKE_TRAINING: PASS - PyTorch reference output-tail conv_out smoke training ran with real optimizer steps
TADSR_PYTORCH_SMOKE_TRAINING_LOSS_DECREASE: PASS - PyTorch smoke-training loss decreased without NaN/Inf
TADSR_JITTOR_SMOKE_TRAINING: PASS - Jittor output-tail conv_out smoke training ran with real optimizer steps
TADSR_JITTOR_SMOKE_TRAINING_LOSS_DECREASE: PASS - Jittor smoke-training loss decreased without NaN/Inf
TADSR_SMOKE_TRAINING_LOSS_LOG: PASS - PyTorch and Jittor smoke-training loss.csv files exist
TADSR_SMOKE_TRAINING_VALIDATION_LOSS_LOG: PASS - PyTorch and Jittor validation_loss.csv files exist for small-data validation evidence
TADSR_SMOKE_TRAINING_LOSS_CURVE: PASS - loss_curve.png generated from both smoke-training logs
TADSR_SMOKE_TRAINING_TRAIN_VAL_LOSS_CURVE: PASS - train/validation loss curve visualizes both PyTorch and Jittor smoke training
TADSR_SMOKE_TRAINING_LOSS_GAP_CURVE: PASS - absolute and relative PyTorch/Jittor loss gap curves are generated
TADSR_SMOKE_TRAINING_PERFORMANCE_LOG: PASS - performance_log.csv exists for both PyTorch and Jittor smoke training
TADSR_SMOKE_TRAINING_PERFORMANCE_VISUALIZATION: PASS - step-time and samples/sec visualizations exist for PyTorch/Jittor smoke training
TADSR_SMOKE_TRAINING_PYTORCH_JITTOR_LOSS_ALIGNMENT: PASS - PyTorch and Jittor output-tail smoke losses both decrease with aligned trend
TADSR_SMOKE_TRAINING_PREDICTION_ALIGNMENT: PASS - PyTorch and Jittor final validation prediction tensors align for the output-tail smoke task
TADSR_SMOKE_TRAINING_VALIDATION_ALIGNMENT: PASS - validation loss curves and prediction summaries align between PyTorch and Jittor
TADSR_SMOKE_TRAINING_TENSOR_VISUALIZATION: PASS - prediction/target/error tensor heatmaps are generated as diagnostics, not restored images
TADSR_SMOKE_TRAINING_MULTI_SEED_STABILITY: PASS - multi-seed small-data output-tail training stability summary exists
TADSR_SMOKE_TRAINING_ARTIFACTS_TEST: PASS - smoke-training artifacts are present and do not claim full training/inference/image/video generation
TADSR_SMALL_DATA_TRAINING_READINESS: PASS - small-data training pipeline evidence is ready; this is output-tail smoke training, not full TADSR training
TADSR_GITHUB_HEAD_ARTIFACT_AUDIT: PASS - tracked tensor artifacts are counted and classified without deleting evidence files
TADSR_GIT_HISTORY_LARGE_BLOB_AUDIT: PASS - Git history blobs are scanned for >100MB hard-limit and >50MB warning risks
TADSR_GITHUB_DELIVERABLE_SIZE_AUDIT: PASS - final deliverables remain below the 100MB submission budget
TADSR_GITHUB_WORKTREE_LARGE_ARTIFACT_AUDIT: PASS - worktree/staged tensor artifact risks are reported without adding new tensor files
TADSR_EVIDENCE_DEPENDENCY_AUDIT: PASS - final evidence manifest depends on metadata/reports rather than requiring raw oracle tensors
TADSR_GITHUB_RELEASE_SLIMMING_DECISION: PASS - release slimming decision memo exists and records direct-push vs cleanup options
TADSR_GITHUB_RELEASE_READINESS_AUDIT: PASS - GitHub release readiness audit reports no hard-limit risk or explains remaining size risk
TADSR_GITHUB_RELEASE_AFTER_PHASE5B_READY: PASS - Phase 5-B summary and diagnostic plan are present in GitHub release readiness evidence
TADSR_JITTOR_MIGRATION_FEASIBILITY_VALIDATION: PASS - matrix-style migration feasibility validation exists and passes
TADSR_MODULE_COVERAGE_MATRIX: PASS - UNet / TimeVAE / LoRA / scheduler / training / deliverables coverage matrix is ready
TADSR_WEIGHT_LOADING_COVERAGE_MATRIX: PASS - weight loading and static effective LoRA coverage matrix is ready
TADSR_LORA_EFFECTIVE_WEIGHT_COVERAGE_MATRIX: PASS - all active static effective LoRA modules are covered by evidence
TADSR_NUMERICAL_ALIGNMENT_COVERAGE_MATRIX: PASS - numerical alignment evidence matrix is ready
TADSR_INTEGRATION_PATH_COVERAGE_MATRIX: PASS - UNet + TimeVAE + LoRA + scheduler + minimal integration paths are summarized
TADSR_TRAINING_PATH_FEASIBILITY_MATRIX: PASS - small-data PyTorch-vs-Jittor training feasibility matrix is ready
TADSR_BOUNDARY_LEVEL_REPRODUCTION: PASS - boundary-level Jittor migration evidence is sufficient for the final report
TADSR_SMALL_DATA_TRAINING_ALIGNMENT: PASS - small-data PyTorch-vs-Jittor loss/prediction/multi-seed alignment is recorded
TADSR_FULL_INFERENCE_GUARD_VALIDATION: PASS - full inference CLI still raises NotImplementedError and remains guarded
TADSR_FULL_INFERENCE_GAP_ANALYSIS: PASS - full inference gap is analyzed and documented while JITTOR_FULL_INFERENCE stays NOT_COMPLETE
TADSR_TIMEVAE_FULL_ALIGNMENT_GAP_ANALYSIS: PASS - TimeVAE full-alignment gap is analyzed while TIME_VAE_FULL_ALIGNMENT stays NOT_COMPLETE
TADSR_LORA_RUNTIME_GAP_ANALYSIS: PASS - static effective LoRA vs dynamic runtime LoRA boundary is documented
TADSR_RUNTIME_LORA_LAYER_FORMULA_ALIGNMENT: PASS - fixed-adapter/fixed-scale LoRA formula equivalence is checked without claiming runtime adapter switching
TADSR_SUBMISSION_FACING_STATUS_SUMMARY: PASS - teacher-facing final status summary exists
TADSR_ENVIRONMENT_BLOCKERS_EXPLAINED: PASS - resource and environment blockers are explicitly explained in the teacher-facing summary
TADSR_GAP_ANALYSIS_READINESS: PASS - all known gaps are analyzed, scoped, and guarded against false completion claims
TADSR_PRODUCTION_COMPLETION_READINESS: PASS - production completion branch/readiness checks are generated without opening full inference
TADSR_PRODUCTION_COMPLETION_BRANCH_READY: PASS - codex/tadsr-production-completion branch and submission-freeze ref are available
TADSR_PRODUCTION_COMPLETION_BASELINE_PRESERVED: PASS - baseline submission markers and NotImplemented full-inference guard are preserved
TADSR_TIMEVAE_FULL_PRODUCTION_PATH_AUDIT: PASS - TimeVAE production path audit report exists; PARTIAL means live official runtime remains a blocker
TADSR_OFFICIAL_RUNTIME_LORA_BEHAVIOR_AUDIT: PASS - official runtime LoRA behavior audit report exists; PARTIAL means existing reports were used
TADSR_DYNAMIC_RUNTIME_LORA_NOT_REQUIRED_FOR_OFFICIAL_INFERENCE: PASS - existing fixed-adapter evidence indicates dynamic LoRA is not required for the audited fixed inference path
TADSR_FULL_INFERENCE_CONTROLLED_VALIDATION_PLAN: PASS - controlled full-inference validation plan exists; stages were not executed in this phase
TADSR_PRODUCTION_COMPLETION_BLOCKER_REPORT: PASS - blocker report documents official-env and future-stage limits without failing current submission
TADSR_PRODUCTION_COMPLETION_PHASE1_VALIDATION: PASS_WITH_BLOCKERS - Phase 1 validation summarizes readiness/audits/plan/blockers without upgrading NOT_COMPLETE markers
TADSR_PRODUCTION_OFFICIAL_ENV_RESOLUTION: PASS - Phase 2 official repo/weights/python availability is resolved without loading models
TADSR_TIMEVAE_PRODUCTION_ORACLE_METADATA: PASS - TimeVAE production oracle metadata report exists or is explicitly blocked by official environment
TADSR_TIMEVAE_METADATA_PARTIAL_DIAGNOSIS: PASS - TimeVAE metadata PARTIAL reason, missing fields and next repair actions are reported
TADSR_OFFICIAL_DIFFUSERS_OVERLAY_READY: PASS - isolated diffusers overlay readiness is recorded without modifying the official strict venv
TADSR_OFFICIAL_RUNTIME_DEPENDENCY_REPAIR_READINESS: PASS - official dependency repair readiness is recorded from overlay dry-run or execution
TADSR_OFFICIAL_RUNTIME_DEPENDENCY_OVERLAY_ACTIVE: PASS - official dependency diagnosis records whether PYTHONPATH overlay was active for imports
TADSR_OFFICIAL_RUNTIME_DEPENDENCY_DIAGNOSIS: PASS - official Python runtime dependency diagnosis is recorded without modifying the environment
TADSR_TIMEVAE_METADATA_REPAIR_ATTEMPT: PASS - TimeVAE production metadata repair attempt is recorded without running full inference
TADSR_TIMEVAE_METADATA_FIELDS_COMPLETE: PASS - TimeVAE required metadata field completeness is evaluated explicitly
TADSR_TIMEVAE_METADATA_READY_FOR_ONE_STEP: PASS - TimeVAE metadata readiness for controlled one-step tensor alignment is evaluated separately
TADSR_TIMEVAE_LIVE_METADATA_COMPLETION: PASS - controlled TimeVAE live encode/decode metadata completion is validated from JSON only
TADSR_TIMEVAE_LIVE_ENCODE_METADATA: PASS - controlled live TimeVAE encode/posterior/latent metadata fields are complete
TADSR_TIMEVAE_LIVE_DECODE_METADATA: PASS - controlled live TimeVAE decode/clamp metadata fields are complete
TADSR_TIMEVAE_LIVE_SAFETY_FLAGS: PASS - controlled live TimeVAE metadata export proves no full inference, scheduler loop, UNet call, image/video generation or raw tensor commit
TADSR_TIMEVAE_PRODUCTION_ALIGNMENT_PREFLIGHT: PASS - Jittor TimeVAE production alignment preflight checks metadata contracts without upgrading full alignment
TADSR_FULL_INFERENCE_METADATA_DRY_RUN_CONTRACT: PARTIAL - full-inference stage contract is documented without running denoising loop or image generation
TADSR_PRODUCTION_COMPLETION_PHASE2_VALIDATION: PASS_WITH_BLOCKERS - Phase 2 validation aggregates official env resolution, TimeVAE metadata, LoRA audit and metadata contract
TADSR_PRODUCTION_COMPLETION_PHASE3_VALIDATION: PASS - Phase 3 validation aggregates live official-env readiness, TimeVAE metadata export, LoRA live audit and the full-inference metadata contract
TADSR_TIMEVAE_LIVE_METADATA_EXPORT: PASS - live TimeVAE metadata export is PASS only when official runtime metadata oracle is actually available
TADSR_LORA_LIVE_RUNTIME_AUDIT: PASS - live LoRA runtime audit is PASS only when official runtime evidence is available; PARTIAL means existing reports were used
TADSR_RUNTIME_LORA_DECISION_FINALIZED_FOR_FIXED_INFERENCE: PASS - fixed-adapter official inference path does not require dynamic runtime LoRA; dynamic runtime LoRA remains future work
TADSR_SMOKE_TRAINING_SUBMISSION_SUMMARY: PASS - small-data PyTorch-vs-Jittor smoke-training evidence is consolidated for submission
TADSR_FULL_INFERENCE_CONTRACT_READY_FOR_ONE_STEP: PASS - one-step tensor alignment readiness is separate from full inference execution and remains blocked until Phase 3 prerequisites pass
TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT_PLAN: PASS - controlled one-step tensor alignment plan is generated only after the live metadata and readiness gates pass
TADSR_PHASE3B_LINUX_MANUAL_RUNBOOK_READY: PASS - Linux Phase 3-B manual execution runbook exists and contains safe instructions
TADSR_PHASE3B_LINUX_LIVE_AUDIT_SCRIPT_READY: PASS - Linux one-click Phase 3-B live audit script exists and remains metadata/audit-only
TADSR_PHASE3B_RESULT_PACKAGER_READY: PASS - Phase 3-B result packager exists and only packages JSON/Markdown/log/txt evidence
TADSR_PHASE3B_RESULT_IMPORT_VALIDATOR_READY: PASS - Windows/local Phase 3-B result import validator exists and rejects raw tensors
TADSR_PHASE3B_MANUAL_EXECUTION_KIT_READY: PASS - manual Linux execution kit is ready while live audit remains blocked by authentication / official runtime gates
TADSR_PHASE3B_FINAL_MANUAL_HANDOFF_READY: PASS - manual handoff validation confirms Linux runbook, env template, live-audit script, packager and importer are ready
TADSR_PHASE3B_LIVE_RESULTS_IMPORT_VALIDATION: PASS - live result import validation is PASS only after a Linux package is safely imported
TADSR_PHASE3B_LIVE_RESULTS_NO_RAW_TENSORS: PASS - live result import package contains no .npy/.npz/local_tensors/weights/checkpoints
TADSR_PHASE3D_LIVE_RESULTS_PACKAGE_FOUND: PASS - Phase 3-D found the Linux live-audit result package in an allowed local candidate path
TADSR_PHASE3D_LIVE_RESULTS_PACKAGE_SECURITY: PASS - Phase 3-D package security passes only after safe dry-run / no-raw-tensor validation
TADSR_PHASE3D_IMPORT_GATE: PASS - Phase 3-D import gate passes only after live results are safely imported
TADSR_PHASE3D_ONE_STEP_GATE_DECISION: PASS - one-step tensor alignment gate decision is separate from full inference and remains blocked until prerequisites pass
TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT_PROTOCOL_READY: PASS - controlled one-step protocol is ready only after import gate and one-step prerequisites pass
TADSR_FULL_INFERENCE_ONE_STEP_GATE_BLOCKER_SUMMARY: PASS - blocker summary explains why controlled one-step tensor alignment is not yet allowed
TADSR_ONE_STEP_OFFICIAL_PATH_AUDIT: PASS - official controlled one-step tensor path is audited without running the full denoising loop or image/video generation
TADSR_ONE_STEP_OFFICIAL_ORACLE_TENSORS: PASS - official PyTorch one-step oracle tensor metadata is exported; raw tensors remain local_tensors-only and ignored
TADSR_ONE_STEP_JITTOR_TENSOR_RUN: PASS - Jittor executes the corresponding controlled one-step tensor path when official local tensors are available
TADSR_ONE_STEP_TENSOR_STAGE_ALIGNMENT: PASS - per-stage shape/range/error/cosine alignment passes only for controlled one-step tensor stages
TADSR_ONE_STEP_TENSOR_ALIGNMENT_VALIDATION: PASS - metadata-only validator checks one-step artifacts, safety flags, and raw tensor policy
TADSR_ONE_STEP_NO_IMAGE_VIDEO_GUARD: PASS - one-step phase did not generate restored images or videos
TADSR_ONE_STEP_RAW_TENSOR_POLICY: PASS - one-step raw .npy/.npz tensors are not tracked or staged
TADSR_ONE_STEP_FULL_INFERENCE_GUARD_PRESERVED: PASS - production full-inference CLI remains guarded by NotImplementedError
TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT: PASS - controlled one-step tensor alignment status; this is not full TADSR inference completion
TADSR_OFFICIAL_ACTUAL_INFERENCE_PATH_AUDIT: PASS - official TADSR actual inference path is audited without executing production full inference
TADSR_OFFICIAL_INFERENCE_MULTISTEP_REQUIREMENT_AUDIT: PASS - single-step vs multi-step requirement is determined from the official actual path
TADSR_OFFICIAL_ACTUAL_PATH_SINGLE_STEP_CONTRACT: PASS - official actual inference path is documented as a single-step/get_x0_from_res contract when applicable
TADSR_POSTPROCESS_CONTRACT_AUDIT: PASS - postprocess/output tensor-to-image contract is documented without image/video generation
TADSR_OUTPUT_IMAGE_SAVE_POLICY_AUDIT: PASS - official image save policy is audited but not executed
TADSR_POSTPROCESS_NOT_EXECUTED_GUARD: PASS - postprocess image save remains documented_not_executed in this project
TADSR_TINY_MULTISTEP_ALIGNMENT_APPLICABILITY: NOT_REQUIRED_FOR_OFFICIAL_ACTUAL_INFERENCE - decision marker records whether tiny multi-step is required by the official actual path
TADSR_TINY_MULTISTEP_ALIGNMENT_DECISION: PASS - multi-step applicability decision is generated without running multi-step alignment
TADSR_EXPERIMENTAL_CLI_METADATA_ONLY_PLAN: PASS - future experimental metadata-only CLI plan is documented without implementation
TADSR_PHASE5B_SUBMISSION_FREEZE_SUMMARY: PASS - Phase 5-B final submission freeze summary exists and preserves honest scope
TADSR_DIAGNOSTIC_IMAGE_SMOKE_PLAN: PASS - future diagnostic image-smoke plan is documented without execution
TADSR_DIAGNOSTIC_IMAGE_SMOKE_EXECUTED: PASS - diagnostic image-smoke execution status is recorded honestly
TADSR_DIAGNOSTIC_IMAGE_SMOKE_READY: PASS - diagnostic image-smoke readiness is recorded as PASS or honest PARTIAL/BLOCKED
TADSR_DIAGNOSTIC_IMAGE_SMOKE_ALIGNMENT: PASS - diagnostic image-smoke alignment is PASS only when local tensors and PNG evidence exist
TADSR_DIAGNOSTIC_IMAGE_SMOKE_NOT_FULL_INFERENCE: PASS - diagnostic image-smoke explicitly does not run full inference
TADSR_DIAGNOSTIC_IMAGE_SMOKE_NO_RAW_TENSORS: PASS - diagnostic image-smoke does not stage raw .npy/.npz tensors
TADSR_DIAGNOSTIC_IMAGE_SMOKE_ARTIFACTS_READY: PASS - diagnostic image-smoke artifacts are available or honestly partial
TADSR_DIAGNOSTIC_IMAGE_SMOKE_VALIDATION: PASS - diagnostic image-smoke validator checks artifacts and guardrails
TADSR_DIAGNOSTIC_IMAGE_SMOKE_FALSE_CLAIM_GUARD: PASS - diagnostic image-smoke wording avoids false full-inference/image-generation claims
TADSR_DIAGNOSTIC_IMAGE_SMOKE_ARTIFACT_POLICY: PASS - diagnostic image-smoke artifact policy rejects staged raw tensors
TADSR_ONE_STEP_DIAGNOSTIC_CLI_READY: PASS - metadata-only one-step diagnostic CLI is ready
TADSR_ONE_STEP_DIAGNOSTIC_CLI_NOT_FULL_INFERENCE: PASS - one-step diagnostic CLI keeps full inference guarded
TADSR_TIMEVAE_FULL_ALIGNMENT_CLOSURE_SUMMARY: PASS - TimeVAE closure summary exists without upgrading full alignment
TADSR_TIMEVAE_FULL_ALIGNMENT_REMAINS_SCOPED: PASS - TimeVAE full alignment remains scoped and NOT_COMPLETE
TADSR_RUNTIME_LORA_FINAL_DECISION_PROOF: PASS - runtime LoRA final decision proof is documented
TADSR_DYNAMIC_RUNTIME_LORA_NOT_REQUIRED_FOR_AUDITED_PATH: PASS - dynamic runtime LoRA is not required for the audited fixed path
TADSR_OFFICIAL_ACTUAL_SINGLE_STEP_PATH_DOCUMENTED: PASS - official actual path is documented as single-step/get_x0_from_res
TADSR_MULTISTEP_NOT_REQUIRED_FOR_OFFICIAL_ACTUAL_PATH: PASS - tiny multi-step is documented as not required for the official actual path
TADSR_CONTROLLED_ONE_STEP_EVIDENCE_DOCUMENTED: PASS - controlled one-step tensor evidence and key metrics are documented
TADSR_FINAL_SUBMISSION_AFTER_PHASE5B_READY: PASS - final submission materials are ready after Phase 5-B without opening full inference
TADSR_FINAL_SUBMISSION_WITH_DIAGNOSTIC_SMOKE_READY: PASS - final submission materials include Phase 6-A diagnostic smoke/CLI/closure proof status
TADSR_COURSE_REQUIREMENT_COMPLIANCE: PASS - course requirement matrix covers the submitted evidence
TADSR_COURSE_REQUIREMENT_EVIDENCE_MATRIX: PASS - course requirement evidence matrix is generated
TADSR_COURSE_TRAINING_REQUIREMENT_EVIDENCE: PASS - training logs/loss curves/performance evidence are indexed for course requirements
TADSR_COURSE_VISUALIZATION_REQUIREMENT_EVIDENCE: PASS - visualization evidence is indexed for course requirements
TADSR_COURSE_GITHUB_REQUIREMENT_EVIDENCE: PASS - GitHub submission readiness evidence is indexed for course requirements
TADSR_COURSE_PPT_VIDEO_REQUIREMENT_EVIDENCE: PASS - PPT/PDF/video guide evidence is indexed for course requirements
TADSR_FINAL_EVIDENCE_INDEX: PASS - teacher-readable evidence index is generated
TADSR_FINAL_EVIDENCE_INDEX_TEACHER_READABLE: PASS - final evidence index is readable by course reviewers
TADSR_TRAINING_ALIGNMENT_EVIDENCE_VALIDATION: PASS - small-data training evidence completeness is audited
TADSR_TRAINING_LOSS_CURVE_EVIDENCE: PASS - training and validation loss curves are present
TADSR_TRAINING_PERFORMANCE_LOG_EVIDENCE: PASS - training performance logs are present
TADSR_TRAINING_OUTPUT_ALIGNMENT_EVIDENCE: PASS - training prediction/output alignment visualizations are present
TADSR_TRAINING_GRAD_PARAM_UPDATE_EVIDENCE: PARTIAL - parameter-update evidence exists; gradient norm may be optional if not logged originally
TADSR_TRAINING_EVIDENCE_TEACHER_READY: PASS - training evidence is ready for teacher review without claiming full TADSR training
TADSR_FINAL_CLAIMS_CONSISTENCY: PASS - final materials preserve scope consistency
TADSR_FINAL_FALSE_CLAIM_GUARD: PASS - final materials avoid misleading full-inference/image-generation/runtime-LoRA claims
TADSR_FINAL_SCOPE_CONSISTENCY: PASS - required NOT_COMPLETE / NOT_IMPLEMENTED statuses remain visible
TADSR_DEFENSE_RISK_RESPONSE_PACK: PASS - Chinese defense risk-response pack covers high-risk reviewer questions
TADSR_DEFENSE_SHORT_ANSWERS_READY: PASS - short Chinese defense answers are ready for live Q&A
TADSR_DEFENSE_LONG_ANSWERS_READY: PASS - long Chinese defense answers are ready for detailed reviewer follow-up
TADSR_DEFENSE_FALSE_CLAIM_GUARD: PASS - defense materials avoid unguarded full-inference/image-generation/runtime-LoRA overclaims
TADSR_DEFENSE_EVIDENCE_LOOKUP_READY: PASS - defense evidence lookup table maps claims to files and demo commands
TADSR_FINAL_RELEASE_CANDIDATE_SIGNOFF: PASS - final release candidate signoff report is present
TADSR_FINAL_RELEASE_CANDIDATE_TECHNICAL_EVIDENCE: PASS - final signoff summarizes core technical evidence
TADSR_FINAL_RELEASE_CANDIDATE_SCOPE_GUARD: PASS - final signoff preserves NOT_COMPLETE / NOT_IMPLEMENTED scope guards
TADSR_FINAL_LINKS_AND_PATHS_VALIDATION: PASS - final critical links and paths are validated
TADSR_FINAL_MARKDOWN_LINKS_VALIDATION: PASS - final Markdown links are scanned; noncritical warnings are recorded
TADSR_FINAL_PLACEHOLDER_SCAN: PASS - final materials are scanned for obvious placeholders
TADSR_FINAL_CHINESE_MATERIALS_VALIDATION: PASS - final reviewer-facing materials are primarily Chinese
TADSR_FINAL_MOJIBAKE_SCAN: PASS - final Chinese materials are scanned for obvious mojibake
TADSR_FINAL_CHINESE_READABILITY_READY: PASS - final Chinese materials are ready for human review
TADSR_FINAL_COMMAND_INDEX: PASS - final demo command index is present
TADSR_FINAL_DEMO_COMMANDS_READY: PASS - teacher-facing demo commands are grouped and explained
TADSR_FINAL_GITHUB_UPLOAD_CHECKLIST: PASS - GitHub / LMS upload checklist is present
TADSR_FINAL_UPLOAD_PACKAGE_READY: PASS - upload package safety checklist is ready
TADSR_CLEAN_PUBLIC_RELEASE_PACKAGE_BUILD: PASS - clean public release tree is built for manual GitHub upload
TADSR_CLEAN_PUBLIC_RELEASE_NO_RAW_TENSORS: PASS - clean public release tree excludes raw tensors and large private artifacts
TADSR_CLEAN_PUBLIC_RELEASE_SIZE_AUDIT: PASS - clean public release tree is below the course upload size budget
TADSR_CLEAN_PUBLIC_RELEASE_PACKAGE_VALIDATION: PASS - clean public release tree validates as GitHub-ready
TADSR_CLEAN_PUBLIC_RELEASE_GITHUB_READY: PASS - clean public release tree can be manually uploaded to GitHub
TADSR_CLEAN_PUBLIC_RELEASE_FALSE_CLAIM_GUARD: PASS - clean public release tree avoids misleading completion claims
TADSR_FINAL_GITHUB_URL_STATUS: PARTIAL_GITHUB_URL_PENDING - GitHub URL status is recorded; pending is acceptable before manual repo creation
TADSR_FINAL_GITHUB_URL_UPDATE_SCRIPT_READY: PASS - GitHub URL update script is ready and does not push or add remotes
TADSR_FINAL_HUMAN_SUBMISSION_INSTRUCTIONS: PASS - final human submission instructions are present
TADSR_FINAL_VIDEO_REHEARSAL_CHECKLIST: PASS - final video rehearsal checklist is present
TADSR_FINAL_VIDEO_RECORDING_READY: PASS - video recording flow is ready for human execution
TADSR_FINAL_SUBMISSION_FREEZE_TAG_DOC: PASS - final freeze/tag documentation is present without auto-tagging or pushing
TADSR_FINAL_HUMAN_SUBMISSION_LOCK_REPORT: PASS - final human submission lock report is present
TADSR_FINAL_HUMAN_SUBMISSION_LOCK_READY: PASS - final human submission lock is ready; only manual submission actions remain
TADSR_FINAL_VIDEO_SUBMISSION_CHECK: PASS - final video submission check is present
TADSR_FINAL_VIDEO_RECORDING_READY: PASS - video recording flow is ready for human execution
TADSR_FULL_ENGINEERING_COMPLETION_ROADMAP: PASS - future full-engineering completion roadmap is documented without execution
TADSR_CONTROLLED_EXTENSION_PLAN_READY: PASS - controlled extension plan is ready if future work is approved
TADSR_TEACHER_REVIEW_GUIDE_READY: PASS - teacher review guide exists and indexes fast review commands/files
TADSR_FINAL_DEFENSE_QA_READY: PASS - final defense Q&A exists for high-risk reviewer questions
TADSR_FINAL_PERFECTION_READINESS: PASS - course compliance, evidence index, training audit, claims guard, defense guide and roadmap are ready while guarded limitations remain honest
TADSR_FINAL_READY_FOR_HUMAN_SUBMISSION: PASS - release candidate signoff, links/path QA, Chinese readability QA, command index and GitHub upload checklist are ready
TADSR_FINAL_PUBLIC_RELEASE_AND_SUBMISSION_READY: PASS - clean public release tree, URL setter, human submission instructions, video rehearsal and freeze docs are ready
TADSR_FINAL_MANUAL_ACTIONS_ONLY_REMAINING: PASS - all technical and material checks are complete; remaining work is GitHub creation, URL writeback, video recording and course submission only
NEXT_ACTION: Manual actions only: create the GitHub repo, upload the clean public release, run set_final_github_url.py with the real URL, record the video, and submit PPT/PDF/video/GitHub URL. Do not expand technical scope further.
```

| Check | Status | Note | Next Action |
|---|---|---|---|
| JITTOR_TINY_TRAINING | PASS | tiny train log/loss curve/checkpoint present | run scripts/train_jittor_tiny.sh |
| JITTOR_TINY_TESTING | PASS | tiny test metrics and compare grid present | run scripts/test_jittor_tiny.sh |
| PYTORCH_TINY_ALIGNMENT | PASS | alignment report and compare curves present | run scripts/compare_jittor_pytorch.sh |
| PYTORCH_OFFICIAL_VENV | BLOCKED | \mnt\data\sj\venvs\tadsr_official_pytorch | bash scripts/install_official_env_matrix.sh |
| PYTORCH_OFFICIAL_ENV_MATRIX | PASS | at least one env fully passed | run bash scripts/install_official_env_matrix.sh or prepare offline_pack |
| PYTORCH_OFFICIAL_SELECTED_ENV | BLOCKED | selected_env.sh missing or points to no usable venv | inspect experiments/full_repro/pytorch_official/env_matrix/selected_env.sh |
| PYTORCH_OFFICIAL_ENV_STRICT_CU118 | PASS | strict-cu118 passed | check install_strict-cu118.log |
| PYTORCH_OFFICIAL_ENV_STRICT_PYPI | PASS | strict-pypi passed | check install_strict-pypi.log |
| PYTORCH_OFFICIAL_ENV_RELAXED_PYPI | PARTIAL | version mismatches: torch, torchvision | check install_relaxed-pypi.log |
| PYTORCH_OFFICIAL_REPO_IMPORTS | PASS | all official repo imports passed | python3 scripts/check_official_repo_imports.py |
| PYTORCH_OFFICIAL_DRYRUN | PASS | test_tadsr.py CLI checked | bash scripts/run_official_pytorch_dryrun.sh |
| PYTORCH_OFFICIAL_ASSETS | BLOCKED | missing: time_vae, unet, vae, text_encoder, tokenizer, scheduler, feature_extractor, bert-base-uncased, DAPE.pth, ram_swin_large_14m.pth, tadsr.pkl | upload weights to /mnt/data/sj/incoming/TADSR_assets/TADSR_weights |
| OFFICIAL_PYTORCH_ORACLE | PASS | outputs=4 inputs=4 | bash scripts/run_official_pytorch_smoke.sh |
| PYTORCH_OFFICIAL_BENCHMARK | BLOCKED_DATASET_MISSING | missing benchmark datasets: RealSR, DRealSR, RealLR200 | upload datasets and run scripts/run_official_pytorch_subset.sh |
| WHEELHOUSE | BLOCKED | 0 files in \mnt\data\sj\wheelhouse\tadsr_official_pytorch | build offline_pack on networked machine and unpack it |
| OFFLINE_PACK_GUIDE | PASS | offline_pack guide exists | restore offline_pack docs |
| JITTOR_GPU | BLOCKED | cuDNN8 required; cuDNN9 not acceptable | install cuDNN8 to /mnt/data/sj/local/cudnn8 |
| JITTOR_BASE_WEIGHT_CONVERSION | PASS | tadsr/DAPE/RAM conversion report | python3 tools/verify_converted_weights.py |
| JITTOR_DIFFUSERS_WEIGHT_CONVERSION | PASS | diffusers NPZ conversion verification | python3 tools/verify_diffusers_npz_weights.py |
| PYTORCH_ORACLE_TENSORS | PASS | preprocess/scheduler oracle tensors exported | python3 tools/export_pytorch_oracle_tensors.py --split smoke |
| TADSR_SCHEDULER_USAGE_AUDIT | PASS | official TADSR scheduler usage path audited without running full inference | python3 tools/audit_official_tadsr_scheduler_boundary.py |
| TADSR_SCHEDULER_CONFIG_AUDIT | PASS | official scheduler class/config/tensor contract audited | python3 tools/audit_official_tadsr_scheduler_boundary.py |
| TADSR_SCHEDULER_TIMESTEP_CONTRACT_AUDIT | PASS | official set_timesteps(1) contract audited | python3 tools/audit_official_tadsr_scheduler_boundary.py |
| TADSR_SCHEDULER_STEP_CONTRACT_AUDIT | PASS | official scheduler.step one-step contract audited for boundary testing only | python3 tools/audit_official_tadsr_scheduler_boundary.py |
| TADSR_SCHEDULER_ORACLE_TENSORS | PASS | official scheduler-only one-step oracle tensors exported; no denoising loop or VAE decode | python3 tools/export_tadsr_scheduler_boundary_oracle.py |
| TADSR_UNET_SCHEDULER_ONE_STEP_ORACLE | PASS | optional one-step scheduler oracle using existing UNet full-forward tensors | python3 tools/export_tadsr_scheduler_boundary_oracle.py |
| TADSR_SCHEDULER_TIMESTEPS_ALIGNMENT | PASS | Jittor scheduler boundary tester matches official timesteps | USE_CUDA=0 python3 tests_jittor_alignment/test_scheduler_boundary_alignment.py |
| TADSR_SCHEDULER_SCALE_MODEL_INPUT_ALIGNMENT | NOT_APPLICABLE_NOOP | scale_model_input is aligned or explicitly no-op for the audited DDPMScheduler path | USE_CUDA=0 python3 tests_jittor_alignment/test_scheduler_boundary_alignment.py |
| TADSR_SCHEDULER_STEP_ALIGNMENT | PASS | Jittor one-step scheduler boundary output matches official oracle | USE_CUDA=0 python3 tests_jittor_alignment/test_scheduler_boundary_alignment.py |
| TADSR_UNET_SCHEDULER_ONE_STEP_ALIGNMENT | PASS | optional existing-UNet-output plus scheduler one-step boundary matches official oracle | USE_CUDA=0 python3 tests_jittor_alignment/test_scheduler_boundary_alignment.py |
| TADSR_SCHEDULER_BOUNDARY_ALIGNMENT | PASS | scheduler boundary / minimal denoising-step contract is audited and aligned; full loop still unopened | USE_CUDA=0 python3 tests_jittor_alignment/test_scheduler_boundary_alignment.py |
| TADSR_MINIMAL_INTEGRATION_AUDIT | PASS | official minimal TADSR latent one-step path audited from source code; no production full inference | python3 tools/audit_official_tadsr_minimal_integration.py |
| TADSR_GET_X0_FROM_RES_AUDIT | PASS | official get_x0_from_res formula audited: latent / sqrt(alpha_prod_t) - model_pred | python3 tools/audit_official_tadsr_minimal_integration.py |
| TADSR_MINIMAL_LATENT_ONE_STEP_CONTRACT_AUDIT | PASS | minimal latent-only encode -> UNet -> x0 contract defined without full loop | python3 tools/audit_official_tadsr_minimal_integration.py |
| TADSR_MINIMAL_LATENT_ONE_STEP_ORACLE | PASS | official minimal latent-only oracle exported; no VAE decode or image output | python3 tools/export_tadsr_minimal_latent_integration_oracle.py |
| TADSR_GET_X0_FROM_RES_ORACLE | PASS | official get_x0_from_res oracle tensor exported | python3 tools/export_tadsr_minimal_latent_integration_oracle.py |
| TADSR_MINIMAL_DECODE_BOUNDARY_ORACLE | PASS | official minimal one-step decode/clamp tensor oracle exported; no image output or full inference | python3 tools/export_tadsr_minimal_latent_integration_oracle.py |
| TADSR_MINIMAL_VAE_ENCODE_ALIGNMENT | PASS | Jittor actual TimeVAE encode/sample/scale boundary matches minimal oracle | USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py |
| TADSR_MINIMAL_UNET_MODEL_PRED_ALIGNMENT | PASS | Jittor UNet full-forward model_pred matches minimal oracle | USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py |
| TADSR_GET_X0_FROM_RES_ALIGNMENT | PASS | Jittor alpha/get_x0_from_res output matches official oracle | USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py |
| TADSR_MINIMAL_LATENT_ONE_STEP_ALIGNMENT | PASS | minimal latent one-step outputs match official oracle | USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py |
| TADSR_MINIMAL_DECODE_INPUT_ALIGNMENT | PASS | Jittor decode_input tensor matches official x0/scaling_factor oracle | USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py |
| TADSR_MINIMAL_DECODED_OUTPUT_ALIGNMENT | PASS | Jittor TimeVAE actual decoder original_forward output matches official tensor oracle | USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py |
| TADSR_MINIMAL_FINAL_CLAMPED_OUTPUT_ALIGNMENT | PASS | Jittor decoded output clamped to [-1, 1] matches official tensor oracle | USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py |
| TADSR_MINIMAL_DECODE_BOUNDARY_ALIGNMENT | PASS | minimal one-step decode boundary is PASS only when decode_input, decoded_output and final_clamped_output all align | USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py |
| TADSR_MINIMAL_LATENT_INTEGRATION_DRY_RUN | PASS | minimal latent-only dry-run combines VAE encode, UNet model_pred and get_x0_from_res without opening full inference | USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py |
| TADSR_MINIMAL_ONE_STEP_DECODE_DRY_RUN | PASS | minimal one-step dry-run includes tensor-only decode/clamp boundary while still avoiding full inference | USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py |
| JITTOR_PREPROCESS_ALIGNMENT | PASS | preprocess tensor alignment | USE_CUDA=0 bash scripts/run_jittor_alignment_tests.sh |
| JITTOR_SCHEDULER_ALIGNMENT | PASS | scheduler tensor alignment | USE_CUDA=0 bash scripts/run_jittor_alignment_tests.sh |
| JITTOR_WEIGHT_LOADING_ALIGNMENT | PASS | converted NPZ loading check | USE_CUDA=0 bash scripts/run_jittor_alignment_tests.sh |
| JITTOR_LORA_MAPPING | PARTIAL | LoRA key table; forward not claimed | USE_CUDA=0 bash scripts/run_jittor_alignment_tests.sh |
| JITTOR_TIME_VAE_LOADING | PARTIAL | time-VAE loading checklist; forward not claimed | USE_CUDA=0 bash scripts/run_jittor_alignment_tests.sh |
| OFFICIAL_TIME_VAE_AUDIT | PASS | official TimeAwareAutoencoderKL module tree and state-key audit | python3 tools/audit_official_time_vae.py |
| TIME_VAE_ORACLE_TENSORS | PASS | PyTorch time-VAE block oracle tensors exported | python3 tools/export_time_vae_oracle_tensors.py |
| TIME_VAE_CONV_IN_ALIGNMENT | PASS | Jittor-side conv_in output aligned with PyTorch oracle | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_conv_in_alignment.py |
| TIME_VAE_FIRST_BLOCK_ALIGNMENT | PASS | first TimeAware ResnetBlock2D numerical alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_first_block_alignment.py |
| LORA_PAIR_ANALYSIS | PASS | LoRA A/B key-pair discovery | python3 tools/analyze_lora_pairs.py |
| LORA_MERGE_VALIDATION | PASS | at least one LoRA pair merge formula numerically aligned | python3 tools/export_lora_merge_oracle.py && python3 tests_jittor_alignment/test_lora_merge_formula.py |
| TIME_VAE_LORA_ALIGNMENT_REPORT | PASS | TimeAware VAE and LoRA progress report | python3 scripts/make_time_vae_lora_alignment_report.py |
| TIME_VAE_DOWNBLOCK0_AUDIT | PASS | official encoder.down_blocks.0 structure and key mapping audit | python3 tools/audit_official_time_vae_downblock0.py |
| TIME_VAE_RESNET1_ALIGNMENT | PASS | encoder.down_blocks.0.resnets.1 numerical alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_resnet1_alignment.py |
| TIME_VAE_DOWNSAMPLER_ALIGNMENT | PASS | encoder.down_blocks.0.downsamplers.0 numerical alignment or NOT_APPLICABLE | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downsampler_alignment.py |
| TIME_VAE_DOWNBLOCK0_ALIGNMENT | PASS | full encoder.down_blocks.0 stage numerical alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock0_alignment.py |
| TIME_VAE_DOWNBLOCK1_AUDIT | PASS | official encoder.down_blocks.1 structure and key mapping audit | python3 tools/audit_official_time_vae_downblock1.py |
| TIME_VAE_DOWNBLOCK1_ORACLE_TENSORS | PASS | PyTorch oracle tensors for encoder.down_blocks.1 exported | python3 tools/export_time_vae_downblock1_oracle.py |
| TIME_VAE_DOWNBLOCK1_RESNET0_ALIGNMENT | PASS | encoder.down_blocks.1.resnets.0 128->256 channel-change alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock1_resnet0_alignment.py |
| TIME_VAE_DOWNBLOCK1_RESNET1_ALIGNMENT | PASS | encoder.down_blocks.1.resnets.1 numerical alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock1_resnet1_alignment.py |
| TIME_VAE_DOWNBLOCK1_DOWNSAMPLER_ALIGNMENT | PASS | encoder.down_blocks.1.downsamplers.0 numerical alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock1_downsampler_alignment.py |
| TIME_VAE_DOWNBLOCK1_ALIGNMENT | PASS | full encoder.down_blocks.1 stage numerical alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock1_alignment.py |
| TIME_VAE_ENCODER_STAGE01_ALIGNMENT | PASS | encoder.conv_in + down_blocks.0 + down_blocks.1 composition alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_stage01_alignment.py |
| TIME_VAE_DOWNBLOCK2_AUDIT | PASS | official encoder.down_blocks.2 structure and key mapping audit | python3 tools/audit_official_time_vae_downblock.py --block-index 2 |
| TIME_VAE_DOWNBLOCK2_ORACLE_TENSORS | PASS | PyTorch oracle tensors for encoder.down_blocks.2 exported | python3 tools/export_time_vae_downblock_oracle.py --block-index 2 |
| TIME_VAE_DOWNBLOCK2_RESNET0_ALIGNMENT | PASS | encoder.down_blocks.2.resnets.0 256->512 channel-change alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock2_resnet0_alignment.py |
| TIME_VAE_DOWNBLOCK2_RESNET1_ALIGNMENT | PASS | encoder.down_blocks.2.resnets.1 numerical alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock2_resnet1_alignment.py |
| TIME_VAE_DOWNBLOCK2_DOWNSAMPLER_ALIGNMENT | PASS | encoder.down_blocks.2.downsamplers.0 numerical alignment or NOT_APPLICABLE | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock2_downsampler_alignment.py |
| TIME_VAE_DOWNBLOCK2_ALIGNMENT | PASS | full encoder.down_blocks.2 stage numerical alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock2_alignment.py |
| TIME_VAE_ENCODER_STAGE012_ALIGNMENT | PASS | encoder.conv_in + down_blocks.0 + down_blocks.1 + down_blocks.2 composition alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_stage012_alignment.py |
| TIME_VAE_DOWNBLOCK3_AUDIT | PASS | official encoder.down_blocks.3 structure and key mapping audit | python3 tools/audit_official_time_vae_downblock.py --block-index 3 |
| TIME_VAE_DOWNBLOCK3_ORACLE_TENSORS | PASS | PyTorch oracle tensors for encoder.down_blocks.3 exported | python3 tools/export_time_vae_downblock_oracle.py --block-index 3 |
| TIME_VAE_DOWNBLOCK3_RESNET0_ALIGNMENT | PASS | encoder.down_blocks.3.resnets.0 numerical alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock3_resnet0_alignment.py |
| TIME_VAE_DOWNBLOCK3_RESNET1_ALIGNMENT | PASS | encoder.down_blocks.3.resnets.1 numerical alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock3_resnet1_alignment.py |
| TIME_VAE_DOWNBLOCK3_DOWNSAMPLER_ALIGNMENT | NOT_APPLICABLE | encoder.down_blocks.3 downsampler status; NOT_APPLICABLE if absent | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock3_downsampler_alignment.py |
| TIME_VAE_DOWNBLOCK3_ALIGNMENT | PASS | full encoder.down_blocks.3 stage numerical alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock3_alignment.py |
| TIME_VAE_ENCODER_STAGE0123_ALIGNMENT | PASS | encoder.conv_in + down_blocks.0 + down_blocks.1 + down_blocks.2 + down_blocks.3 composition alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_stage0123_alignment.py |
| TIME_VAE_MIDBLOCK_AUDIT | PASS | official encoder.mid_block structure and key mapping audit | python3 tools/audit_official_time_vae_midblock.py |
| TIME_VAE_MIDBLOCK_ORACLE_TENSORS | PASS | PyTorch oracle tensors for encoder.mid_block exported | python3 tools/export_time_vae_midblock_oracle.py |
| TIME_VAE_MIDBLOCK_RESNET0_ALIGNMENT | PASS | encoder.mid_block.resnets.0 numerical alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_midblock_resnet0_alignment.py |
| TIME_VAE_MIDBLOCK_ATTENTION_ALIGNMENT | PASS | encoder.mid_block.attentions.0 numerical alignment or NOT_APPLICABLE | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_midblock_attention_alignment.py |
| TIME_VAE_MIDBLOCK_RESNET1_ALIGNMENT | PASS | encoder.mid_block.resnets.1 numerical alignment or NOT_APPLICABLE | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_midblock_resnet1_alignment.py |
| TIME_VAE_MIDBLOCK_ALIGNMENT | PASS | full encoder.mid_block stage numerical alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_midblock_alignment.py |
| TIME_VAE_ENCODER_STAGE0123_MID_ALIGNMENT | PASS | encoder.conv_in + down_blocks.0..3 + mid_block composition alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_stage0123_mid_alignment.py |
| TIME_VAE_ENCODER_TAIL_AUDIT | PASS | official encoder tail and quant_conv structure/key mapping audit | python3 tools/audit_official_time_vae_encoder_tail.py |
| TIME_VAE_ENCODER_TAIL_ORACLE_TENSORS | PASS | PyTorch oracle tensors for encoder tail and quant_conv exported | python3 tools/export_time_vae_encoder_tail_oracle.py |
| TIME_VAE_ENCODER_TAIL_NORM_ALIGNMENT | PASS | encoder.conv_norm_out numerical alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_tail_norm_alignment.py |
| TIME_VAE_ENCODER_TAIL_ACT_ALIGNMENT | PASS | encoder.conv_act numerical alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_tail_act_alignment.py |
| TIME_VAE_ENCODER_TAIL_CONV_OUT_ALIGNMENT | PASS | encoder.conv_out numerical alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_tail_conv_out_alignment.py |
| TIME_VAE_ENCODER_TAIL_ALIGNMENT | PASS | full encoder tail numerical alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_tail_alignment.py |
| TIME_VAE_QUANT_CONV_ALIGNMENT | PASS | deterministic quant_conv moments tensor alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_quant_conv_alignment.py |
| TIME_VAE_ENCODER_STAGE0123_MID_TAIL_ALIGNMENT | PASS | encoder.conv_in + down_blocks.0..3 + mid_block + tail composition alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_stage0123_mid_tail_alignment.py |
| TIME_VAE_ENCODER_STAGE0123_MID_TAIL_QUANT_ALIGNMENT | PASS | encoder-side deterministic path through quant_conv alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_stage0123_mid_tail_quant_alignment.py |
| TIME_VAE_DECODER_ENTRY_AUDIT | PASS | official decoder entry, DGD split/mode and post_quant_conv audit | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_ENTRY_ORACLE_TENSORS | PASS | PyTorch decoder-entry oracle tensors exported | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_MOMENTS_SPLIT_ALIGNMENT | PASS | posterior moments mean/logvar split and clamp alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_POSTERIOR_MODE_ALIGNMENT | PASS | posterior mode(mean) alignment; no sampling | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_POST_QUANT_CONV_ALIGNMENT | PASS | post_quant_conv numerical alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_CONV_IN_ALIGNMENT | PASS | decoder.conv_in numerical alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_ENTRY_ALIGNMENT | PASS | post_quant_conv + decoder.conv_in synthetic latent path alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_QUANT_TO_DECODER_ENTRY_ALIGNMENT | PASS | quant moments -> mode -> decoder entry alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_ENCODER_TO_DECODER_ENTRY_ALIGNMENT | PASS | deterministic encoder-to-decoder-entry bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_ENTRY_BRIDGE_ALIGNMENT | PASS | posterior mode + post_quant_conv + decoder.conv_in deterministic bridge; decoder body is checked in later rows | continue with decoder.mid_block / up_blocks |
| TIME_VAE_DECODER_MIDBLOCK_AUDIT | PASS | official decoder.mid_block topology and key audit | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_MIDBLOCK_ORACLE_TENSORS | PASS | PyTorch decoder.mid_block oracle tensors exported | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_MIDBLOCK_RESNET0_ALIGNMENT | PASS | decoder.mid_block.resnets.0 alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_MIDBLOCK_ATTENTION_ALIGNMENT | PASS | decoder.mid_block.attentions.0 alignment or NOT_APPLICABLE | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_MIDBLOCK_RESNET1_ALIGNMENT | PASS | decoder.mid_block.resnets.1 alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_MIDBLOCK_ALIGNMENT | PASS | full decoder.mid_block deterministic alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_MIDBLOCK_SYNTHETIC_ALIGNMENT | PASS | isolated synthetic hidden -> decoder.mid_block alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_ENTRY_MIDBLOCK_ALIGNMENT | PASS | decoder entry -> decoder.mid_block bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_QUANT_TO_DECODER_MIDBLOCK_ALIGNMENT | PASS | quant moments -> posterior mode -> decoder.mid_block bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_ENCODER_TO_DECODER_MIDBLOCK_ALIGNMENT | PASS | encoder -> quant -> decoder.mid_block deterministic bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_MIDBLOCK_BRIDGE_ALIGNMENT | PASS | deterministic decoder.mid_block aligned through isolated, decoder-entry and encoder-to-decoder paths; later decoder stages are checked separately | continue with decoder.up_blocks |
| TIME_VAE_DECODER_UPBLOCKS_AUDIT | PASS | official decoder.up_blocks audit; up_blocks.0 focus plus later-block topology notes | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK0_AUDIT | PASS | official decoder.up_blocks.0 topology and key audit | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK0_ORACLE_TENSORS | PASS | PyTorch decoder.up_blocks.0 oracle tensors exported | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK0_RESNET0_ALIGNMENT | PASS | decoder.up_blocks.0.resnets.0 alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK0_RESNET1_ALIGNMENT | PASS | decoder.up_blocks.0.resnets.1 alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK0_RESNET2_ALIGNMENT | PASS | decoder.up_blocks.0.resnets.2 alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK0_UPSAMPLER0_ALIGNMENT | PASS | decoder.up_blocks.0.upsamplers.0 nearest-2x + conv alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK0_ALIGNMENT | PASS | full decoder.up_blocks.0 deterministic alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK0_SYNTHETIC_ALIGNMENT | PASS | isolated synthetic hidden -> decoder.up_blocks.0 alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_MIDBLOCK_UPBLOCK0_ALIGNMENT | PASS | decoder.mid_block -> decoder.up_blocks.0 bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_ENTRY_MIDBLOCK_UPBLOCK0_ALIGNMENT | PASS | decoder entry -> decoder.mid_block -> decoder.up_blocks.0 bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_QUANT_TO_DECODER_UPBLOCK0_ALIGNMENT | PASS | quant moments -> posterior mode -> decoder.up_blocks.0 bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_ENCODER_TO_DECODER_UPBLOCK0_ALIGNMENT | PASS | encoder -> quant -> decoder.up_blocks.0 deterministic bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK0_BRIDGE_ALIGNMENT | PASS | deterministic decoder.up_blocks.0 aligned through isolated, decoder-entry and encoder-to-decoder paths; later up blocks and tail are checked separately | continue with decoder.up_blocks.1 |
| TIME_VAE_DECODER_UPBLOCK1_AUDIT | PASS | official decoder.up_blocks.1 topology and key audit | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK1_ORACLE_TENSORS | PASS | PyTorch decoder.up_blocks.1 oracle tensors exported | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK1_RESNET0_ALIGNMENT | PASS | decoder.up_blocks.1.resnets.0 alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK1_RESNET1_ALIGNMENT | PASS | decoder.up_blocks.1.resnets.1 alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK1_RESNET2_ALIGNMENT | PASS | decoder.up_blocks.1.resnets.2 alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK1_UPSAMPLER0_ALIGNMENT | PASS | decoder.up_blocks.1.upsamplers.0 nearest-2x + conv alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK1_ALIGNMENT | PASS | full decoder.up_blocks.1 deterministic alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK1_SYNTHETIC_ALIGNMENT | PASS | isolated synthetic hidden -> decoder.up_blocks.1 pressure alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK0_UPBLOCK1_ALIGNMENT | PASS | decoder.up_blocks.0 -> decoder.up_blocks.1 bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_ENTRY_MIDBLOCK_UPBLOCKS01_ALIGNMENT | PASS | decoder entry -> decoder.mid_block -> decoder.up_blocks.0 -> decoder.up_blocks.1 bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_QUANT_TO_DECODER_UPBLOCK1_ALIGNMENT | PASS | quant moments -> posterior mode -> decoder.up_blocks.1 bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_ENCODER_TO_DECODER_UPBLOCK1_ALIGNMENT | PASS | encoder -> quant -> decoder.up_blocks.1 deterministic bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK1_BRIDGE_ALIGNMENT | PASS | deterministic decoder.up_blocks.1 aligned through isolated, decoder-entry, quant-to-decoder and encoder-to-decoder paths; later up blocks and tail are checked separately | continue with decoder.up_blocks.2 |
| TIME_VAE_DECODER_UPBLOCK2_AUDIT | PASS | official decoder.up_blocks.2 topology and key audit | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK2_ORACLE_TENSORS | PASS | PyTorch decoder.up_blocks.2 oracle tensors exported | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK2_RESNET0_ALIGNMENT | PASS | decoder.up_blocks.2.resnets.0 512->256 shortcut alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK2_RESNET1_ALIGNMENT | PASS | decoder.up_blocks.2.resnets.1 alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK2_RESNET2_ALIGNMENT | PASS | decoder.up_blocks.2.resnets.2 alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK2_UPSAMPLER0_ALIGNMENT | PASS | decoder.up_blocks.2.upsamplers.0 nearest-2x + conv alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK2_ALIGNMENT | PASS | full decoder.up_blocks.2 deterministic alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK2_SYNTHETIC_ALIGNMENT | PASS | isolated synthetic hidden -> decoder.up_blocks.2 pressure alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCKS012_ALIGNMENT | PASS | decoder.up_blocks.0 -> decoder.up_blocks.1 -> decoder.up_blocks.2 bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_ENTRY_MIDBLOCK_UPBLOCKS012_ALIGNMENT | PASS | decoder entry -> decoder.mid_block -> decoder.up_blocks.0 -> decoder.up_blocks.1 -> decoder.up_blocks.2 bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_QUANT_TO_DECODER_UPBLOCK2_ALIGNMENT | PASS | quant moments -> posterior mode -> decoder.up_blocks.2 bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_ENCODER_TO_DECODER_UPBLOCK2_ALIGNMENT | PASS | encoder -> quant -> decoder.up_blocks.2 deterministic bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK2_BRIDGE_ALIGNMENT | PASS | deterministic decoder.up_blocks.2 aligned through isolated, decoder-entry, quant-to-decoder and encoder-to-decoder paths; decoder.up_blocks.3 and tail are checked separately | continue with decoder.up_blocks.3 |
| TIME_VAE_DECODER_UPBLOCK3_AUDIT | PASS | official decoder.up_blocks.3 topology and key audit | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK3_ORACLE_TENSORS | PASS | PyTorch decoder.up_blocks.3 oracle tensors exported | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK3_RESNET0_ALIGNMENT | PASS | decoder.up_blocks.3.resnets.0 256->128 shortcut alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK3_RESNET1_ALIGNMENT | PASS | decoder.up_blocks.3.resnets.1 alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK3_RESNET2_ALIGNMENT | PASS | decoder.up_blocks.3.resnets.2 alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK3_UPSAMPLER0_ALIGNMENT | NOT_APPLICABLE | decoder.up_blocks.3 has no official upsampler; NOT_APPLICABLE is expected | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK3_ALIGNMENT | PASS | full decoder.up_blocks.3 deterministic alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK3_SYNTHETIC_ALIGNMENT | PASS | isolated synthetic hidden -> decoder.up_blocks.3 pressure alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCKS0123_ALIGNMENT | PASS | decoder.up_blocks.0 -> decoder.up_blocks.1 -> decoder.up_blocks.2 -> decoder.up_blocks.3 bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_ENTRY_MIDBLOCK_UPBLOCKS0123_ALIGNMENT | PASS | decoder entry -> decoder.mid_block -> decoder.up_blocks.0 -> decoder.up_blocks.1 -> decoder.up_blocks.2 -> decoder.up_blocks.3 bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_QUANT_TO_DECODER_UPBLOCK3_ALIGNMENT | PASS | quant moments -> posterior mode -> decoder.up_blocks.3 bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_ENCODER_TO_DECODER_UPBLOCK3_ALIGNMENT | PASS | encoder -> quant -> decoder.up_blocks.3 deterministic bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCK3_BRIDGE_ALIGNMENT | PASS | deterministic decoder.up_blocks.3 aligned through isolated, decoder-entry, quant-to-decoder and encoder-to-decoder paths | continue with decoder tail |
| TIME_VAE_DECODER_UPBLOCKS_ALIGNMENT | PASS | decoder up_blocks.0/1/2/3 are aligned as block-level and bridge-level components | continue with decoder tail |
| TIME_VAE_DECODER_TAIL_AUDIT | PASS | official decoder tail topology and key audit | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_TAIL_ORACLE_TENSORS | PASS | PyTorch decoder tail oracle tensors exported | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_TAIL_NORM_ALIGNMENT | PASS | decoder.conv_norm_out alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_TAIL_ACT_ALIGNMENT | PASS | decoder.conv_act SiLU alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_TAIL_CONV_OUT_ALIGNMENT | PASS | decoder.conv_out alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_TAIL_ALIGNMENT | PASS | full isolated decoder tail alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_TAIL_SYNTHETIC_ALIGNMENT | PASS | isolated synthetic hidden -> decoder tail alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_UPBLOCKS0123_TO_TAIL_ALIGNMENT | PASS | decoder.up_blocks.0~3 -> decoder tail bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_ENTRY_MIDBLOCK_UPBLOCKS0123_TAIL_ALIGNMENT | PASS | decoder entry -> mid_block -> up_blocks.0~3 -> tail bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_QUANT_TO_DECODER_TAIL_ALIGNMENT | PASS | quant moments -> posterior mode -> decoder tail bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_ENCODER_TO_DECODER_TAIL_ALIGNMENT | PASS | encoder -> quant -> deterministic decoder tail bridge alignment | USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TIME_VAE_DECODER_TAIL_BRIDGE_ALIGNMENT | PASS | deterministic decoder tail aligned through isolated, decoder-entry, quant-to-decoder and encoder-to-decoder paths | continue deterministic decoder wrapper / UNet only after this stays PASS |
| TIME_VAE_DETERMINISTIC_DECODER_ALIGNMENT | PASS | post_quant_conv -> decoder.conv_in -> decoder.mid_block -> decoder.up_blocks.0~3 -> decoder tail deterministic stack alignment | continue full UNet audit/export/port after deterministic decoder stack alignment |
| TIME_VAE_FULL_DECODER_ALIGNMENT | PASS | deterministic decoder stack module-level alignment only; not stochastic VAE sampling, not full VAE API, and not full TADSR inference | continue UNet/LoRA runtime before full inference |
| TIME_VAE_DETERMINISTIC_RECONSTRUCTION_ALIGNMENT | PASS | deterministic encoder -> quant_conv moments -> posterior mode -> decoder stack -> tail alignment; no random sampling | continue full UNet audit/export/port |
| TIME_VAE_FULL_API_AUDIT | PASS | official TimeAwareAutoencoderKL API/method/config audit for full boundary | python3 tools/audit_official_tadsr_timevae_full_boundary.py |
| TIME_VAE_PIPELINE_USAGE_AUDIT | PASS | official TADSR pipeline VAE usage audited: encode/sample/scale/decode/clamp plus tiled-hook note | python3 tools/audit_official_tadsr_timevae_full_boundary.py |
| TIME_VAE_FULL_BOUNDARY_CONTRACT_AUDIT | PASS | minimal deterministic TimeVAE boundary contract defined from official TADSR usage | python3 tools/audit_official_tadsr_timevae_full_boundary.py |
| TIME_VAE_FULL_BOUNDARY_ORACLE_TENSORS | PASS | official PyTorch oracle tensors for TimeVAE encode/sample/scale/decode/clamp boundary exported | python3 tools/export_tadsr_timevae_full_boundary_oracle.py |
| TIME_VAE_FULL_BOUNDARY_ENCODE_ALIGNMENT | PASS | Jittor encoder/quant moments alignment inside the TimeVAE full-boundary tester | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_full_boundary_alignment.py |
| TIME_VAE_FULL_BOUNDARY_DECODE_ALIGNMENT | PASS | Jittor decode path alignment inside the TimeVAE full-boundary tester | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_full_boundary_alignment.py |
| TIME_VAE_FULL_BOUNDARY_POSTPROCESS_ALIGNMENT | PASS | Jittor clamp[-1,1] postprocess alignment for the TimeVAE boundary | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_full_boundary_alignment.py |
| TIME_VAE_FULL_BOUNDARY_ALIGNMENT | PASS | alignment-only non-tiled encode/sample/scale/decode/clamp TimeVAE boundary; no scheduler/full inference | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_full_boundary_alignment.py |
| TIME_VAE_TILED_HOOK_AUDIT | PASS | official VAEHook patch targets and dispatch behavior audited; no Jittor tiled implementation claimed | python3 tools/audit_official_tadsr_vae_tiled_boundary.py |
| TIME_VAE_TILED_PIPELINE_USAGE_AUDIT | PASS | official TADSR_test VAEHook usage and encode/decode call paths audited | python3 tools/audit_official_tadsr_vae_tiled_boundary.py |
| TIME_VAE_TILED_BOUNDARY_CONTRACT_AUDIT | PASS | official tiled VAE boundary contract audited, including encoder tiling and decoder hook blocker | python3 tools/audit_official_tadsr_vae_tiled_boundary.py |
| TIME_VAE_TILED_ORACLE_FEASIBILITY | BLOCKED_DECODER_HOOK_CONTRACT | official tiled encode/decode oracle feasibility; BLOCKED is acceptable if contract is explicitly diagnosed | python3 tools/audit_official_tadsr_vae_tiled_boundary.py |
| TIME_VAE_TILED_ORACLE_TENSORS | BLOCKED_DECODER_HOOK_CONTRACT | official tiled VAE oracle tensors or blocked metadata exported without running scheduler/full inference | python3 tools/export_tadsr_vae_tiled_boundary_oracle.py |
| TIME_VAE_TILED_VS_NONTILED_REFERENCE_RECORDED | NOT_APPLICABLE | tiled-vs-non-tiled comparison recorded only when a truthful official tiled oracle exists | python3 tools/audit_official_tadsr_vae_tiled_boundary.py |
| TIME_VAE_ACTUAL_VAEHOOK_BEHAVIOR_DECISION | PASS | design decision: mirror official actual VAEHook behavior rather than corrected tiled decoder | docs/timevae_tiled_policy_decision.md |
| TIME_VAE_ACTUAL_VAEHOOK_BEHAVIOR_AUDIT | PASS | official actual VAEHook behavior audited: encoder hook can tile, decoder hook dispatches original_forward | python3 tools/audit_official_tadsr_vae_actual_hook_behavior.py |
| TIME_VAE_OFFICIAL_MIRROR_POLICY | PASS | mainline policy mirrors official actual behavior; corrected tiled decoder is beyond official and deferred | docs/timevae_tiled_policy_decision.md |
| TIME_VAE_DECODER_TILED_PATH_STATUS | BLOCKED_DECODER_HOOK_CONTRACT | official decoder tiled path remains blocked/not reachable because decoder hook has time_vae=False | python3 tools/audit_official_tadsr_vae_actual_hook_behavior.py |
| TIME_VAE_ACTUAL_ENCODER_TILED_ORACLE_FEASIBILITY | PASS | controlled official actual encoder hook/tiled path feasibility | python3 tools/audit_official_tadsr_vae_actual_hook_behavior.py |
| TIME_VAE_ACTUAL_VAEHOOK_ORACLE_TENSORS | PASS | official actual VAEHook behavior oracle tensors exported; not an ideal corrected tiled decoder oracle | python3 tools/export_tadsr_vae_actual_hook_behavior_oracle.py |
| TIME_VAE_ACTUAL_ENCODER_TILED_ORACLE_TENSORS | PASS | actual encoder hook oracle tensors with encoder tiled path trigger status | python3 tools/export_tadsr_vae_actual_hook_behavior_oracle.py |
| TIME_VAE_ACTUAL_DECODER_HOOK_ORIGINAL_FORWARD_ORACLE | PASS | actual decoder hook oracle confirms installed hook uses original_forward and not tiled decode | python3 tools/export_tadsr_vae_actual_hook_behavior_oracle.py |
| TIME_VAE_ACTUAL_VAEHOOK_ORACLE_CONTRACT_TEST | PASS | metadata-only contract test for official actual VAEHook behavior oracle | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_oracle_contract.py |
| TIME_VAE_ACTUAL_ENCODER_TILE_QUEUE_ALIGNMENT | PASS | Jittor actual-hook wrapper mirrors official encoder VAEHook task queue, tile split/crop/write semantics, and cross-tile GroupNorm bookkeeping | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_alignment.py |
| TIME_VAE_ACTUAL_ENCODER_TILED_RAW_OUTPUT_ALIGNMENT | PASS | Jittor tiled encoder raw output before quant_conv against official actual VAEHook raw encoder oracle | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_alignment.py |
| TIME_VAE_ACTUAL_ENCODER_TILED_MOMENTS_ALIGNMENT | PASS | Jittor actual-hook wrapper computes official tiled encoder output plus quant_conv moments parity | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_alignment.py |
| TIME_VAE_ACTUAL_ENCODER_TILED_POSTERIOR_ALIGNMENT | PASS | Jittor actual-hook wrapper posterior tensors against official actual encoder oracle | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_alignment.py |
| TIME_VAE_ACTUAL_ENCODER_TILED_ALIGNMENT | PASS | official-actual encoder hook behavior alignment through tiled task queue and quant_conv | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_alignment.py |
| TIME_VAE_ACTUAL_DECODER_ORIGINAL_FORWARD_ALIGNMENT | PASS | official-actual decoder hook dispatches to original_forward and aligns numerically | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_alignment.py |
| TIME_VAE_ACTUAL_DECODER_HOOK_BEHAVIOR_ALIGNMENT | PASS | decoder hook behavior parity: installed hook, time_vae=False, original_forward path | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_alignment.py |
| TIME_VAE_ACTUAL_VAEHOOK_BEHAVIOR_ALIGNMENT | PASS | Jittor official-actual VAEHook behavior wrapper/tester status; does not invent a corrected tiled decoder | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_alignment.py |
| TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT | PASS | actual hook full boundary from encode/sample/scale/decode/clamp; decoder remains official original_forward and full TADSR inference remains closed | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_alignment.py |
| TIME_VAE_TILED_DECODER_ALIGNMENT | NOT_APPLICABLE_BLOCKED_DECODER_HOOK_CONTRACT | not applicable for official actual behavior because decoder VAEHook uses original_forward, not vae_tile_forward | do not force decoder time_vae=True without a separate corrected-decoder experiment |
| TADSR_UNET_OVERVIEW_AUDIT | PASS | official UNet overview audit: config, input contract, LoRA state, entry modules | python3 tools/audit_official_tadsr_unet_entry.py && python3 tools/export_tadsr_unet_entry_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_AUDIT | PASS | official UNet entry audit: conv_in, time_proj, time_embedding | python3 tools/audit_official_tadsr_unet_entry.py && python3 tools/export_tadsr_unet_entry_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_ORACLE_TENSORS | PASS | PyTorch oracle tensors for UNet entry exported | python3 tools/audit_official_tadsr_unet_entry.py && python3 tools/export_tadsr_unet_entry_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_EFFECTIVE_WEIGHTS | PASS | effective static conv_in + time_embedding weights exported | python3 tools/audit_official_tadsr_unet_entry.py && python3 tools/export_tadsr_unet_entry_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_CENTER_INPUT_ALIGNMENT | PASS | center_input_sample bridge alignment | python3 tools/audit_official_tadsr_unet_entry.py && python3 tools/export_tadsr_unet_entry_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_CONV_IN_ALIGNMENT | PASS | UNet conv_in effective-weight alignment | python3 tools/audit_official_tadsr_unet_entry.py && python3 tools/export_tadsr_unet_entry_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_TIME_PROJ_ALIGNMENT | PASS | UNet Timesteps positional projection alignment | python3 tools/audit_official_tadsr_unet_entry.py && python3 tools/export_tadsr_unet_entry_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_TIME_EMBED_LINEAR1_ALIGNMENT | PASS | UNet time_embedding.linear_1 alignment | python3 tools/audit_official_tadsr_unet_entry.py && python3 tools/export_tadsr_unet_entry_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_TIME_EMBED_ACT_ALIGNMENT | PASS | UNet time_embedding SiLU activation alignment | python3 tools/audit_official_tadsr_unet_entry.py && python3 tools/export_tadsr_unet_entry_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_TIME_EMBED_ALIGNMENT | PASS | UNet full timestep embedding MLP alignment | python3 tools/audit_official_tadsr_unet_entry.py && python3 tools/export_tadsr_unet_entry_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_ALIGNMENT | PASS | UNet entry aggregate alignment before down_blocks.0 | python3 tools/audit_official_tadsr_unet_entry.py && python3 tools/export_tadsr_unet_entry_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_VAE_MODE_ALIGNMENT | NOT_APPLICABLE | optional VAE-mode latent bridge; NOT_APPLICABLE until full TimeAware VAE API opens | continue full VAE API before enabling this optional bridge |
| TADSR_UNET_ENTRY_ALIGNMENT_AGGREGATE | PASS | UNet center input, conv_in, time_proj and time_embedding aligned; full UNet blocks are still unopened | Continue TADSR UNet up_blocks.0.upsamplers.0 audit/export/port after up_blocks.0.resnets.2 alignment; keep full UNet forward and full inference NotImplemented until all up_blocks and LoRA runtime integration pass. |
| TADSR_UNET_DOWNBLOCK0_AUDIT | PASS | official UNet down_blocks.0 overview audit | python3 tools/audit_official_tadsr_unet_downblock0_resnet0.py && python3 tools/export_tadsr_unet_downblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_RESNET0_AUDIT | PASS | official down_blocks.0.resnets.0 audit | python3 tools/audit_official_tadsr_unet_downblock0_resnet0.py && python3 tools/export_tadsr_unet_downblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_RESNET0_ORACLE_TENSORS | PASS | PyTorch oracle tensors for down_blocks.0.resnets.0 exported | python3 tools/audit_official_tadsr_unet_downblock0_resnet0.py && python3 tools/export_tadsr_unet_downblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_RESNET0_EFFECTIVE_WEIGHTS | PASS | effective static weights for down_blocks.0.resnets.0 exported locally | python3 tools/audit_official_tadsr_unet_downblock0_resnet0.py && python3 tools/export_tadsr_unet_downblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_RESNET0_NORM1_ALIGNMENT | PASS | down_blocks.0.resnets.0 norm1 alignment | python3 tools/audit_official_tadsr_unet_downblock0_resnet0.py && python3 tools/export_tadsr_unet_downblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_RESNET0_CONV1_ALIGNMENT | PASS | down_blocks.0.resnets.0 conv1 alignment | python3 tools/audit_official_tadsr_unet_downblock0_resnet0.py && python3 tools/export_tadsr_unet_downblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_RESNET0_TEMB_PROJ_ALIGNMENT | PASS | down_blocks.0.resnets.0 time_emb_proj alignment | python3 tools/audit_official_tadsr_unet_downblock0_resnet0.py && python3 tools/export_tadsr_unet_downblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_RESNET0_CONV2_ALIGNMENT | PASS | down_blocks.0.resnets.0 conv2 alignment | python3 tools/audit_official_tadsr_unet_downblock0_resnet0.py && python3 tools/export_tadsr_unet_downblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_RESNET0_ALIGNMENT | PASS | entry hidden/temb -> down_blocks.0.resnets.0 alignment | python3 tools/audit_official_tadsr_unet_downblock0_resnet0.py && python3 tools/export_tadsr_unet_downblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_RESNET0_SYNTHETIC_ALIGNMENT | PASS | isolated synthetic hidden/temb -> down_blocks.0.resnets.0 alignment | python3 tools/audit_official_tadsr_unet_downblock0_resnet0.py && python3 tools/export_tadsr_unet_downblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_TO_DOWNBLOCK0_RESNET0_ALIGNMENT | PASS | UNet entry -> down_blocks.0.resnets.0 bridge alignment | python3 tools/audit_official_tadsr_unet_downblock0_resnet0.py && python3 tools/export_tadsr_unet_downblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_RESNET0_SHORTCUT_ALIGNMENT | NOT_APPLICABLE | down_blocks.0.resnets.0 shortcut alignment; NOT_APPLICABLE when identity shortcut is official | continue if shortcut appears in later channel-changing ResNet |
| TADSR_UNET_DOWNBLOCK0_RESNET0_BRIDGE_ALIGNMENT | PASS | first UNet down_blocks.0 leaf ResNet aligned in isolated and entry-bridge paths only; full down_blocks.0 remains unopened | Continue TADSR UNet up_blocks.0.upsamplers.0 audit/export/port after up_blocks.0.resnets.2 alignment; keep full UNet forward and full inference NotImplemented until all up_blocks and LoRA runtime integration pass. |
| TADSR_UNET_DOWNBLOCK0_ATTENTION0_AUDIT | PASS | official down_blocks.0.attentions.0 audit | python3 tools/audit_official_tadsr_unet_downblock0_attention0.py && python3 tools/export_tadsr_unet_downblock0_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION0_ORACLE_TENSORS | PASS | PyTorch oracle tensors for down_blocks.0.attentions.0 exported | python3 tools/audit_official_tadsr_unet_downblock0_attention0.py && python3 tools/export_tadsr_unet_downblock0_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION0_EFFECTIVE_WEIGHTS | PASS | effective static weights for down_blocks.0.attentions.0 exported locally | python3 tools/audit_official_tadsr_unet_downblock0_attention0.py && python3 tools/export_tadsr_unet_downblock0_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION0_NORM_ALIGNMENT | PASS | attention0 top-level GroupNorm alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention0.py && python3 tools/export_tadsr_unet_downblock0_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION0_PROJ_IN_ALIGNMENT | PASS | attention0 proj_in alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention0.py && python3 tools/export_tadsr_unet_downblock0_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION0_SEQUENCE_ALIGNMENT | PASS | attention0 NCHW-to-sequence alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention0.py && python3 tools/export_tadsr_unet_downblock0_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION0_ATTN1_QKV_ALIGNMENT | PASS | attention0 transformer0 attn1 QKV alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention0.py && python3 tools/export_tadsr_unet_downblock0_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION0_ATTN1_ALIGNMENT | PASS | attention0 transformer0 self-attention output alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention0.py && python3 tools/export_tadsr_unet_downblock0_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION0_AFTER_ATTN1_ALIGNMENT | PASS | attention0 transformer0 after-attn1 residual alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention0.py && python3 tools/export_tadsr_unet_downblock0_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION0_ATTN2_QKV_ALIGNMENT | PASS | attention0 transformer0 attn2 QKV alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention0.py && python3 tools/export_tadsr_unet_downblock0_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION0_ATTN2_ALIGNMENT | PASS | attention0 transformer0 cross-attention output alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention0.py && python3 tools/export_tadsr_unet_downblock0_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION0_AFTER_ATTN2_ALIGNMENT | PASS | attention0 transformer0 after-attn2 residual alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention0.py && python3 tools/export_tadsr_unet_downblock0_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION0_FF_ALIGNMENT | PASS | attention0 transformer0 feed-forward output alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention0.py && python3 tools/export_tadsr_unet_downblock0_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION0_TRANSFORMER0_ALIGNMENT | PASS | attention0 full transformer block alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention0.py && python3 tools/export_tadsr_unet_downblock0_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION0_ALIGNMENT | PASS | isolated down_blocks.0.attentions.0 alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention0.py && python3 tools/export_tadsr_unet_downblock0_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_RESNET0_ATTENTION0_ALIGNMENT | PASS | UNet entry -> resnet0 -> attention0 bridge alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention0.py && python3 tools/export_tadsr_unet_downblock0_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION0_BRIDGE_ALIGNMENT | PASS | down_blocks.0.attentions.0 aligned in isolated and entry-resnet0 bridge paths only; full down_blocks.0 remains unopened | Continue TADSR UNet up_blocks.0.upsamplers.0 audit/export/port after up_blocks.0.resnets.2 alignment; keep full UNet forward and full inference NotImplemented until all up_blocks and LoRA runtime integration pass. |
| TADSR_UNET_DOWNBLOCK0_RESNET1_AUDIT | PASS | official down_blocks.0.resnets.1 audit | python3 tools/audit_official_tadsr_unet_downblock0_resnet1.py && python3 tools/export_tadsr_unet_downblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_RESNET1_ORACLE_TENSORS | PASS | PyTorch oracle tensors for down_blocks.0.resnets.1 exported | python3 tools/audit_official_tadsr_unet_downblock0_resnet1.py && python3 tools/export_tadsr_unet_downblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_RESNET1_EFFECTIVE_WEIGHTS | PASS | effective static weights for down_blocks.0.resnets.1 exported locally | python3 tools/audit_official_tadsr_unet_downblock0_resnet1.py && python3 tools/export_tadsr_unet_downblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_RESNET1_NORM1_ALIGNMENT | PASS | down_blocks.0.resnets.1 norm1 alignment | python3 tools/audit_official_tadsr_unet_downblock0_resnet1.py && python3 tools/export_tadsr_unet_downblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_RESNET1_CONV1_ALIGNMENT | PASS | down_blocks.0.resnets.1 conv1 alignment | python3 tools/audit_official_tadsr_unet_downblock0_resnet1.py && python3 tools/export_tadsr_unet_downblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_RESNET1_TEMB_PROJ_ALIGNMENT | PASS | down_blocks.0.resnets.1 time_emb_proj alignment | python3 tools/audit_official_tadsr_unet_downblock0_resnet1.py && python3 tools/export_tadsr_unet_downblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_RESNET1_CONV2_ALIGNMENT | PASS | down_blocks.0.resnets.1 conv2 alignment | python3 tools/audit_official_tadsr_unet_downblock0_resnet1.py && python3 tools/export_tadsr_unet_downblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_RESNET1_ALIGNMENT | PASS | entry-attention hidden/temb -> down_blocks.0.resnets.1 alignment | python3 tools/audit_official_tadsr_unet_downblock0_resnet1.py && python3 tools/export_tadsr_unet_downblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_RESNET1_SYNTHETIC_ALIGNMENT | PASS | isolated synthetic hidden/temb -> down_blocks.0.resnets.1 alignment | python3 tools/audit_official_tadsr_unet_downblock0_resnet1.py && python3 tools/export_tadsr_unet_downblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_RESNET0_ATTENTION0_RESNET1_ALIGNMENT | PASS | UNet entry -> resnet0 -> attention0 -> resnet1 bridge alignment | python3 tools/audit_official_tadsr_unet_downblock0_resnet1.py && python3 tools/export_tadsr_unet_downblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_RESNET1_SHORTCUT_ALIGNMENT | NOT_APPLICABLE | down_blocks.0.resnets.1 shortcut alignment; NOT_APPLICABLE when identity shortcut is official | continue if shortcut appears in later channel-changing ResNet |
| TADSR_UNET_DOWNBLOCK0_RESNET1_BRIDGE_ALIGNMENT | PASS | down_blocks.0.resnets.1 aligned in isolated and entry-resnet0-attention0 bridge paths only; full down_blocks.0 remains unopened | Continue TADSR UNet up_blocks.0.upsamplers.0 audit/export/port after up_blocks.0.resnets.2 alignment; keep full UNet forward and full inference NotImplemented until all up_blocks and LoRA runtime integration pass. |
| TADSR_UNET_DOWNBLOCK0_ATTENTION1_AUDIT | PASS | official down_blocks.0.attentions.1 audit | python3 tools/audit_official_tadsr_unet_downblock0_attention1.py && python3 tools/export_tadsr_unet_downblock0_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION1_ORACLE_TENSORS | PASS | PyTorch oracle tensors for down_blocks.0.attentions.1 exported | python3 tools/audit_official_tadsr_unet_downblock0_attention1.py && python3 tools/export_tadsr_unet_downblock0_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION1_EFFECTIVE_WEIGHTS | PASS | effective static weights for down_blocks.0.attentions.1 exported locally | python3 tools/audit_official_tadsr_unet_downblock0_attention1.py && python3 tools/export_tadsr_unet_downblock0_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION1_NORM_ALIGNMENT | PASS | attention1 top-level GroupNorm alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention1.py && python3 tools/export_tadsr_unet_downblock0_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION1_PROJ_IN_ALIGNMENT | PASS | attention1 proj_in alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention1.py && python3 tools/export_tadsr_unet_downblock0_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION1_SEQUENCE_ALIGNMENT | PASS | attention1 NCHW-to-sequence alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention1.py && python3 tools/export_tadsr_unet_downblock0_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION1_ATTN1_QKV_ALIGNMENT | PASS | attention1 transformer0 attn1 QKV alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention1.py && python3 tools/export_tadsr_unet_downblock0_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION1_ATTN1_ALIGNMENT | PASS | attention1 transformer0 self-attention output alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention1.py && python3 tools/export_tadsr_unet_downblock0_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION1_AFTER_ATTN1_ALIGNMENT | PASS | attention1 transformer0 after-attn1 residual alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention1.py && python3 tools/export_tadsr_unet_downblock0_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION1_ATTN2_QKV_ALIGNMENT | PASS | attention1 transformer0 attn2 QKV alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention1.py && python3 tools/export_tadsr_unet_downblock0_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION1_ATTN2_ALIGNMENT | PASS | attention1 transformer0 cross-attention output alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention1.py && python3 tools/export_tadsr_unet_downblock0_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION1_AFTER_ATTN2_ALIGNMENT | PASS | attention1 transformer0 after-attn2 residual alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention1.py && python3 tools/export_tadsr_unet_downblock0_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION1_FF_ALIGNMENT | PASS | attention1 transformer0 feed-forward output alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention1.py && python3 tools/export_tadsr_unet_downblock0_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION1_TRANSFORMER0_ALIGNMENT | PASS | attention1 full transformer block alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention1.py && python3 tools/export_tadsr_unet_downblock0_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION1_ALIGNMENT | PASS | isolated down_blocks.0.attentions.1 alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention1.py && python3 tools/export_tadsr_unet_downblock0_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT | PASS | UNet entry -> resnet0 -> attention0 -> resnet1 -> attention1 bridge alignment | python3 tools/audit_official_tadsr_unet_downblock0_attention1.py && python3 tools/export_tadsr_unet_downblock0_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ATTENTION1_BRIDGE_ALIGNMENT | PASS | down_blocks.0.attentions.1 aligned in isolated and entry-resnet0-attention0-resnet1 bridge paths only; full down_blocks.0 remains unopened | Continue TADSR UNet up_blocks.0.upsamplers.0 audit/export/port after up_blocks.0.resnets.2 alignment; keep full UNet forward and full inference NotImplemented until all up_blocks and LoRA runtime integration pass. |
| TADSR_UNET_DOWNBLOCK0_PRE_DOWNSAMPLER_ALIGNMENT | PASS | down_blocks.0 path through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 is aligned | continue with down_blocks.0.downsamplers.0 if this is not PASS |
| TADSR_UNET_DOWNBLOCK0_DOWNSAMPLER_AUDIT | PASS | official down_blocks.0.downsamplers.0 audit | python3 tools/audit_official_tadsr_unet_downblock0_downsampler.py && python3 tools/export_tadsr_unet_downblock0_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_LOCAL_FORWARD_AUDIT | PASS | official local down_blocks.0 manual-chain audit | python3 tools/audit_official_tadsr_unet_downblock0_downsampler.py && python3 tools/export_tadsr_unet_downblock0_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_DOWNSAMPLER_ORACLE_TENSORS | PASS | PyTorch oracle tensors for down_blocks.0.downsamplers.0 and local down_blocks.0 exported | python3 tools/audit_official_tadsr_unet_downblock0_downsampler.py && python3 tools/export_tadsr_unet_downblock0_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_DOWNSAMPLER_EFFECTIVE_WEIGHTS | PASS | effective static weights for down_blocks.0.downsamplers.0 exported locally | python3 tools/audit_official_tadsr_unet_downblock0_downsampler.py && python3 tools/export_tadsr_unet_downblock0_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_DOWNSAMPLER_ALIGNMENT | PASS | isolated down_blocks.0.downsamplers.0 alignment | python3 tools/audit_official_tadsr_unet_downblock0_downsampler.py && python3 tools/export_tadsr_unet_downblock0_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_ALIGNMENT | PASS | full local UNet down_blocks.0 alignment: entry -> resnet0 -> attention0 -> resnet1 -> attention1 -> downsampler | python3 tools/audit_official_tadsr_unet_downblock0_downsampler.py && python3 tools/export_tadsr_unet_downblock0_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK0_BRIDGE_ALIGNMENT | PASS | complete local down_blocks.0 bridge aligned through downsampler; later UNet blocks remain unopened | continue with down_blocks.1.attentions.0 |
| TADSR_UNET_DOWNBLOCK1_AUDIT | PASS | official UNet down_blocks.1 overview audit, resnets.0 baseline context | python3 tools/audit_official_tadsr_unet_downblock1_resnet0.py && python3 tools/export_tadsr_unet_downblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_RESNET0_AUDIT | PASS | official down_blocks.1.resnets.0 audit | python3 tools/audit_official_tadsr_unet_downblock1_resnet0.py && python3 tools/export_tadsr_unet_downblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_RESNET0_ORACLE_TENSORS | PASS | PyTorch oracle tensors for down_blocks.1.resnets.0 exported | python3 tools/audit_official_tadsr_unet_downblock1_resnet0.py && python3 tools/export_tadsr_unet_downblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_RESNET0_EFFECTIVE_WEIGHTS | PASS | effective static weights for down_blocks.1.resnets.0 exported locally | python3 tools/audit_official_tadsr_unet_downblock1_resnet0.py && python3 tools/export_tadsr_unet_downblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_RESNET0_NORM1_ALIGNMENT | PASS | down_blocks.1.resnets.0 norm1 alignment | python3 tools/audit_official_tadsr_unet_downblock1_resnet0.py && python3 tools/export_tadsr_unet_downblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_RESNET0_CONV1_ALIGNMENT | PASS | down_blocks.1.resnets.0 conv1 alignment | python3 tools/audit_official_tadsr_unet_downblock1_resnet0.py && python3 tools/export_tadsr_unet_downblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_RESNET0_TEMB_PROJ_ALIGNMENT | PASS | down_blocks.1.resnets.0 time_emb_proj alignment | python3 tools/audit_official_tadsr_unet_downblock1_resnet0.py && python3 tools/export_tadsr_unet_downblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_RESNET0_CONV2_ALIGNMENT | PASS | down_blocks.1.resnets.0 conv2 alignment | python3 tools/audit_official_tadsr_unet_downblock1_resnet0.py && python3 tools/export_tadsr_unet_downblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_RESNET0_ALIGNMENT | PASS | down_blocks.1.resnets.0 local alignment | python3 tools/audit_official_tadsr_unet_downblock1_resnet0.py && python3 tools/export_tadsr_unet_downblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_RESNET0_SYNTHETIC_ALIGNMENT | PASS | isolated synthetic hidden/temb -> down_blocks.1.resnets.0 alignment | python3 tools/audit_official_tadsr_unet_downblock1_resnet0.py && python3 tools/export_tadsr_unet_downblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_RESNET0_ALIGNMENT | PASS | UNet entry -> full local down_blocks.0 -> down_blocks.1.resnets.0 bridge alignment | python3 tools/audit_official_tadsr_unet_downblock1_resnet0.py && python3 tools/export_tadsr_unet_downblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_RESNET0_SHORTCUT_ALIGNMENT | PASS | down_blocks.1.resnets.0 shortcut alignment; channel-changing ResNet should include conv_shortcut | continue only if shortcut remains PASS |
| TADSR_UNET_DOWNBLOCK1_RESNET0_BRIDGE_ALIGNMENT | PASS | down_blocks.1.resnets.0 aligned after complete local down_blocks.0 bridge; remaining down_blocks.1 modules are intentionally unopened | continue with down_blocks.1.attentions.0 only |
| TADSR_UNET_DOWNBLOCK1_ATTENTION0_AUDIT | PASS | official down_blocks.1.attentions.0 audit | python3 tools/audit_official_tadsr_unet_downblock1_attention0.py && python3 tools/export_tadsr_unet_downblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION0_ORACLE_TENSORS | PASS | PyTorch oracle tensors for down_blocks.1.attentions.0 exported | python3 tools/audit_official_tadsr_unet_downblock1_attention0.py && python3 tools/export_tadsr_unet_downblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION0_EFFECTIVE_WEIGHTS | PASS | effective static weights for down_blocks.1.attentions.0 exported locally | python3 tools/audit_official_tadsr_unet_downblock1_attention0.py && python3 tools/export_tadsr_unet_downblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION0_NORM_ALIGNMENT | PASS | down_blocks.1.attentions.0 top-level norm alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention0.py && python3 tools/export_tadsr_unet_downblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION0_PROJ_IN_ALIGNMENT | PASS | down_blocks.1.attentions.0 proj_in alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention0.py && python3 tools/export_tadsr_unet_downblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION0_SEQUENCE_ALIGNMENT | PASS | down_blocks.1.attentions.0 NCHW-to-sequence alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention0.py && python3 tools/export_tadsr_unet_downblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION0_ATTN1_QKV_ALIGNMENT | PASS | down_blocks.1.attentions.0 transformer0 attn1 QKV alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention0.py && python3 tools/export_tadsr_unet_downblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION0_ATTN1_ALIGNMENT | PASS | down_blocks.1.attentions.0 transformer0 self-attention output alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention0.py && python3 tools/export_tadsr_unet_downblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION0_AFTER_ATTN1_ALIGNMENT | PASS | down_blocks.1.attentions.0 after-attn1 residual alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention0.py && python3 tools/export_tadsr_unet_downblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION0_ATTN2_QKV_ALIGNMENT | PASS | down_blocks.1.attentions.0 transformer0 attn2 QKV alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention0.py && python3 tools/export_tadsr_unet_downblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION0_ATTN2_ALIGNMENT | PASS | down_blocks.1.attentions.0 transformer0 cross-attention output alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention0.py && python3 tools/export_tadsr_unet_downblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION0_AFTER_ATTN2_ALIGNMENT | PASS | down_blocks.1.attentions.0 after-attn2 residual alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention0.py && python3 tools/export_tadsr_unet_downblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION0_FF_ALIGNMENT | PASS | down_blocks.1.attentions.0 feed-forward output alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention0.py && python3 tools/export_tadsr_unet_downblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION0_TRANSFORMER0_ALIGNMENT | PASS | down_blocks.1.attentions.0 full transformer block alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention0.py && python3 tools/export_tadsr_unet_downblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION0_ALIGNMENT | PASS | isolated down_blocks.1.attentions.0 alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention0.py && python3 tools/export_tadsr_unet_downblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_RESNET0_ATTENTION0_ALIGNMENT | PASS | UNet entry -> down_blocks.0 -> down_blocks.1.resnets.0 -> attentions.0 bridge alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention0.py && python3 tools/export_tadsr_unet_downblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION0_BRIDGE_ALIGNMENT | PASS | down_blocks.1.attentions.0 aligned after complete local down_blocks.0 and down_blocks.1.resnets.0 bridge; remaining down_blocks.1 modules are intentionally unopened | continue with down_blocks.1.resnets.1 only |
| TADSR_UNET_DOWNBLOCK1_RESNET1_AUDIT | PASS | official down_blocks.1.resnets.1 audit | python3 tools/audit_official_tadsr_unet_downblock1_resnet1.py && python3 tools/export_tadsr_unet_downblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_RESNET1_ORACLE_TENSORS | PASS | PyTorch oracle tensors for down_blocks.1.resnets.1 exported | python3 tools/audit_official_tadsr_unet_downblock1_resnet1.py && python3 tools/export_tadsr_unet_downblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_RESNET1_EFFECTIVE_WEIGHTS | PASS | effective static weights for down_blocks.1.resnets.1 exported locally | python3 tools/audit_official_tadsr_unet_downblock1_resnet1.py && python3 tools/export_tadsr_unet_downblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_RESNET1_NORM1_ALIGNMENT | PASS | down_blocks.1.resnets.1 norm1 alignment | python3 tools/audit_official_tadsr_unet_downblock1_resnet1.py && python3 tools/export_tadsr_unet_downblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_RESNET1_CONV1_ALIGNMENT | PASS | down_blocks.1.resnets.1 conv1 alignment | python3 tools/audit_official_tadsr_unet_downblock1_resnet1.py && python3 tools/export_tadsr_unet_downblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_RESNET1_TEMB_PROJ_ALIGNMENT | PASS | down_blocks.1.resnets.1 time_emb_proj alignment | python3 tools/audit_official_tadsr_unet_downblock1_resnet1.py && python3 tools/export_tadsr_unet_downblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_RESNET1_CONV2_ALIGNMENT | PASS | down_blocks.1.resnets.1 conv2 alignment | python3 tools/audit_official_tadsr_unet_downblock1_resnet1.py && python3 tools/export_tadsr_unet_downblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_RESNET1_ALIGNMENT | PASS | down_blocks.1.resnets.1 local alignment | python3 tools/audit_official_tadsr_unet_downblock1_resnet1.py && python3 tools/export_tadsr_unet_downblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_RESNET1_SYNTHETIC_ALIGNMENT | PASS | isolated synthetic hidden/temb -> down_blocks.1.resnets.1 alignment | python3 tools/audit_official_tadsr_unet_downblock1_resnet1.py && python3 tools/export_tadsr_unet_downblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_RESNET0_ATTENTION0_RESNET1_ALIGNMENT | PASS | UNet entry -> down_blocks.0 -> down_blocks.1.resnets.0 -> attentions.0 -> resnets.1 bridge alignment | python3 tools/audit_official_tadsr_unet_downblock1_resnet1.py && python3 tools/export_tadsr_unet_downblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_RESNET1_SHORTCUT_ALIGNMENT | NOT_APPLICABLE | down_blocks.1.resnets.1 shortcut alignment; should be NOT_APPLICABLE/PASS depending on audited shortcut | continue only if shortcut remains PASS or NOT_APPLICABLE |
| TADSR_UNET_DOWNBLOCK1_RESNET1_BRIDGE_ALIGNMENT | PASS | down_blocks.1.resnets.1 aligned after complete local down_blocks.0, down_blocks.1.resnets.0 and down_blocks.1.attentions.0 bridge; remaining down_blocks.1 modules are intentionally unopened | continue with down_blocks.1.downsamplers.0 only |
| TADSR_UNET_DOWNBLOCK1_PRE_ATTENTION1_ALIGNMENT | PASS | down_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 only | continue with down_blocks.1.downsamplers.0 only |
| TADSR_UNET_DOWNBLOCK1_ATTENTION1_AUDIT | PASS | official down_blocks.1.attentions.1 audit | python3 tools/audit_official_tadsr_unet_downblock1_attention1.py && python3 tools/export_tadsr_unet_downblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION1_ORACLE_TENSORS | PASS | PyTorch oracle tensors for down_blocks.1.attentions.1 exported | python3 tools/audit_official_tadsr_unet_downblock1_attention1.py && python3 tools/export_tadsr_unet_downblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION1_EFFECTIVE_WEIGHTS | PASS | effective static weights for down_blocks.1.attentions.1 exported locally | python3 tools/audit_official_tadsr_unet_downblock1_attention1.py && python3 tools/export_tadsr_unet_downblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION1_NORM_ALIGNMENT | PASS | down_blocks.1.attentions.1 top-level norm alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention1.py && python3 tools/export_tadsr_unet_downblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION1_PROJ_IN_ALIGNMENT | PASS | down_blocks.1.attentions.1 proj_in alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention1.py && python3 tools/export_tadsr_unet_downblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION1_SEQUENCE_ALIGNMENT | PASS | down_blocks.1.attentions.1 NCHW-to-sequence alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention1.py && python3 tools/export_tadsr_unet_downblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION1_ATTN1_QKV_ALIGNMENT | PASS | down_blocks.1.attentions.1 transformer0 attn1 QKV alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention1.py && python3 tools/export_tadsr_unet_downblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION1_ATTN1_ALIGNMENT | PASS | down_blocks.1.attentions.1 transformer0 self-attention output alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention1.py && python3 tools/export_tadsr_unet_downblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION1_AFTER_ATTN1_ALIGNMENT | PASS | down_blocks.1.attentions.1 after-attn1 residual alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention1.py && python3 tools/export_tadsr_unet_downblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION1_ATTN2_QKV_ALIGNMENT | PASS | down_blocks.1.attentions.1 transformer0 attn2 QKV alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention1.py && python3 tools/export_tadsr_unet_downblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION1_ATTN2_ALIGNMENT | PASS | down_blocks.1.attentions.1 transformer0 cross-attention output alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention1.py && python3 tools/export_tadsr_unet_downblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION1_AFTER_ATTN2_ALIGNMENT | PASS | down_blocks.1.attentions.1 after-attn2 residual alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention1.py && python3 tools/export_tadsr_unet_downblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION1_FF_ALIGNMENT | PASS | down_blocks.1.attentions.1 feed-forward output alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention1.py && python3 tools/export_tadsr_unet_downblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION1_TRANSFORMER0_ALIGNMENT | PASS | down_blocks.1.attentions.1 full transformer block alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention1.py && python3 tools/export_tadsr_unet_downblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION1_ALIGNMENT | PASS | isolated down_blocks.1.attentions.1 alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention1.py && python3 tools/export_tadsr_unet_downblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT | PASS | UNet entry -> down_blocks.0 -> down_blocks.1 resnet0/attention0/resnet1/attention1 bridge alignment | python3 tools/audit_official_tadsr_unet_downblock1_attention1.py && python3 tools/export_tadsr_unet_downblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ATTENTION1_BRIDGE_ALIGNMENT | PASS | down_blocks.1.attentions.1 aligned after complete local down_blocks.0, down_blocks.1.resnets.0, attentions.0 and resnets.1 bridge; downsampler is checked separately below | continue with down_blocks.1.downsamplers.0 only |
| TADSR_UNET_DOWNBLOCK1_PRE_DOWNSAMPLER_ALIGNMENT | PASS | down_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 only | continue with down_blocks.1.downsamplers.0 only |
| TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_AUDIT | PASS | official down_blocks.1.downsamplers.0 audit | python3 tools/audit_official_tadsr_unet_downblock1_downsampler.py && python3 tools/export_tadsr_unet_downblock1_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_LORA_AUDIT | PASS | LoRA wrapper/static effective conv audit for down_blocks.1.downsamplers.0 | python3 tools/audit_official_tadsr_unet_downblock1_downsampler.py && python3 tools/export_tadsr_unet_downblock1_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_LOCAL_FORWARD_AUDIT | PASS | official manual-chain local down_blocks.1 audit | python3 tools/audit_official_tadsr_unet_downblock1_downsampler.py && python3 tools/export_tadsr_unet_downblock1_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_OUTPUT_STATES_AUDIT | PASS | official down_blocks.1 output_states order audit | python3 tools/audit_official_tadsr_unet_downblock1_downsampler.py && python3 tools/export_tadsr_unet_downblock1_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_ORACLE_TENSORS | PASS | PyTorch oracle tensors for down_blocks.1.downsamplers.0 and local down_blocks.1 exported | python3 tools/audit_official_tadsr_unet_downblock1_downsampler.py && python3 tools/export_tadsr_unet_downblock1_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_EFFECTIVE_WEIGHTS | PASS | effective static weights for down_blocks.1.downsamplers.0 exported locally | python3 tools/audit_official_tadsr_unet_downblock1_downsampler.py && python3 tools/export_tadsr_unet_downblock1_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_OUTPUT_STATES_ORACLE_TENSORS | PASS | PyTorch oracle tensors for down_blocks.1 output_states exported | python3 tools/audit_official_tadsr_unet_downblock1_downsampler.py && python3 tools/export_tadsr_unet_downblock1_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_ALIGNMENT | PASS | isolated down_blocks.1.downsamplers.0 alignment | python3 tools/audit_official_tadsr_unet_downblock1_downsampler.py && python3 tools/export_tadsr_unet_downblock1_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_RESNET0_ATTENTION0_RESNET1_ATTENTION1_DOWNSAMPLER_ALIGNMENT | PASS | UNet entry -> down_blocks.0 -> down_blocks.1 resnet0/attention0/resnet1/attention1/downsampler bridge alignment | python3 tools/audit_official_tadsr_unet_downblock1_downsampler.py && python3 tools/export_tadsr_unet_downblock1_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_ALIGNMENT | PASS | full local down_blocks.1 alignment through downsampler and output_states | python3 tools/audit_official_tadsr_unet_downblock1_downsampler.py && python3 tools/export_tadsr_unet_downblock1_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_PADDING_ALIGNMENT | NOT_APPLICABLE | official Downsample2D uses Conv2d padding=1 directly; no separate padding op is exported | continue with down_blocks.2 after down_blocks.1 stays PASS |
| TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_CONV_ALIGNMENT | PASS | effective static Conv2d alignment for down_blocks.1.downsamplers.0 | continue only if PASS |
| TADSR_UNET_DOWNBLOCK1_OUTPUT_HIDDEN_ALIGNMENT | PASS | down_blocks.1 hidden_states output equals official block.forward output | continue only if PASS |
| TADSR_UNET_DOWNBLOCK1_OUTPUT_STATES_ALIGNMENT | PASS | down_blocks.1 output_states tuple order and tensors match official block.forward | continue only if PASS |
| TADSR_UNET_DOWNBLOCK1_BRIDGE_ALIGNMENT | PASS | complete local down_blocks.1 bridge aligned through downsampler and output_states; later UNet blocks remain unopened | continue with down_blocks.2.resnets.0 |
| TADSR_UNET_DOWNBLOCK2_AUDIT | PASS | official UNet down_blocks.2 overview audit, resnets.0 baseline context | python3 tools/audit_official_tadsr_unet_downblock2_resnet0.py && python3 tools/export_tadsr_unet_downblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET0_AUDIT | PASS | official down_blocks.2.resnets.0 audit | python3 tools/audit_official_tadsr_unet_downblock2_resnet0.py && python3 tools/export_tadsr_unet_downblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET0_LORA_AUDIT | PASS | LoRA/effective weight state for down_blocks.2.resnets.0 | python3 tools/audit_official_tadsr_unet_downblock2_resnet0.py && python3 tools/export_tadsr_unet_downblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET0_ORACLE_TENSORS | PASS | PyTorch oracle tensors for down_blocks.2.resnets.0 exported | python3 tools/audit_official_tadsr_unet_downblock2_resnet0.py && python3 tools/export_tadsr_unet_downblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET0_EFFECTIVE_WEIGHTS | PASS | effective static weights for down_blocks.2.resnets.0 exported locally | python3 tools/audit_official_tadsr_unet_downblock2_resnet0.py && python3 tools/export_tadsr_unet_downblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET0_NORM1_ALIGNMENT | PASS | down_blocks.2.resnets.0 norm1 alignment | python3 tools/audit_official_tadsr_unet_downblock2_resnet0.py && python3 tools/export_tadsr_unet_downblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET0_CONV1_ALIGNMENT | PASS | down_blocks.2.resnets.0 conv1 alignment | python3 tools/audit_official_tadsr_unet_downblock2_resnet0.py && python3 tools/export_tadsr_unet_downblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET0_TEMB_PROJ_ALIGNMENT | PASS | down_blocks.2.resnets.0 time_emb_proj alignment | python3 tools/audit_official_tadsr_unet_downblock2_resnet0.py && python3 tools/export_tadsr_unet_downblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET0_CONV2_ALIGNMENT | PASS | down_blocks.2.resnets.0 conv2 alignment | python3 tools/audit_official_tadsr_unet_downblock2_resnet0.py && python3 tools/export_tadsr_unet_downblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET0_ALIGNMENT | PASS | down_blocks.2.resnets.0 local alignment | python3 tools/audit_official_tadsr_unet_downblock2_resnet0.py && python3 tools/export_tadsr_unet_downblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET0_SYNTHETIC_ALIGNMENT | PASS | isolated synthetic hidden/temb -> down_blocks.2.resnets.0 alignment | python3 tools/audit_official_tadsr_unet_downblock2_resnet0.py && python3 tools/export_tadsr_unet_downblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ALIGNMENT | PASS | UNet entry -> full local down_blocks.0 -> full local down_blocks.1 -> down_blocks.2.resnets.0 bridge alignment | python3 tools/audit_official_tadsr_unet_downblock2_resnet0.py && python3 tools/export_tadsr_unet_downblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET0_SHORTCUT_ALIGNMENT | PASS | down_blocks.2.resnets.0 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent | continue only if shortcut remains PASS or NOT_APPLICABLE |
| TADSR_UNET_DOWNBLOCK2_RESNET0_BRIDGE_ALIGNMENT | PASS | down_blocks.2.resnets.0 aligned after complete local down_blocks.0 and down_blocks.1 bridge; remaining down_blocks.2 modules are intentionally unopened | continue with down_blocks.2.attentions.0 only |
| TADSR_UNET_DOWNBLOCK2_ATTENTION0_AUDIT | PASS | official down_blocks.2.attentions.0 audit | python3 tools/audit_official_tadsr_unet_downblock2_attention0.py && python3 tools/export_tadsr_unet_downblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION0_ORACLE_TENSORS | PASS | PyTorch oracle tensors for down_blocks.2.attentions.0 exported | python3 tools/audit_official_tadsr_unet_downblock2_attention0.py && python3 tools/export_tadsr_unet_downblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION0_EFFECTIVE_WEIGHTS | PASS | effective static weights for down_blocks.2.attentions.0 exported locally | python3 tools/audit_official_tadsr_unet_downblock2_attention0.py && python3 tools/export_tadsr_unet_downblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION0_NORM_ALIGNMENT | PASS | down_blocks.2.attentions.0 top-level norm alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention0.py && python3 tools/export_tadsr_unet_downblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION0_PROJ_IN_ALIGNMENT | PASS | down_blocks.2.attentions.0 proj_in alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention0.py && python3 tools/export_tadsr_unet_downblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION0_SEQUENCE_ALIGNMENT | PASS | down_blocks.2.attentions.0 NCHW-to-sequence alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention0.py && python3 tools/export_tadsr_unet_downblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION0_ATTN1_QKV_ALIGNMENT | PASS | down_blocks.2.attentions.0 transformer0 attn1 QKV alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention0.py && python3 tools/export_tadsr_unet_downblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION0_ATTN1_ALIGNMENT | PASS | down_blocks.2.attentions.0 transformer0 self-attention output alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention0.py && python3 tools/export_tadsr_unet_downblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION0_AFTER_ATTN1_ALIGNMENT | PASS | down_blocks.2.attentions.0 after-attn1 residual alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention0.py && python3 tools/export_tadsr_unet_downblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION0_ATTN2_QKV_ALIGNMENT | PASS | down_blocks.2.attentions.0 transformer0 attn2 QKV alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention0.py && python3 tools/export_tadsr_unet_downblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION0_ATTN2_ALIGNMENT | PASS | down_blocks.2.attentions.0 transformer0 cross-attention output alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention0.py && python3 tools/export_tadsr_unet_downblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION0_AFTER_ATTN2_ALIGNMENT | PASS | down_blocks.2.attentions.0 after-attn2 residual alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention0.py && python3 tools/export_tadsr_unet_downblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION0_FF_ALIGNMENT | PASS | down_blocks.2.attentions.0 feed-forward output alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention0.py && python3 tools/export_tadsr_unet_downblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION0_TRANSFORMER0_ALIGNMENT | PASS | down_blocks.2.attentions.0 full transformer block alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention0.py && python3 tools/export_tadsr_unet_downblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION0_ALIGNMENT | PASS | isolated down_blocks.2.attentions.0 alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention0.py && python3 tools/export_tadsr_unet_downblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ATTENTION0_ALIGNMENT | PASS | UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2.resnets.0 -> attentions.0 bridge alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention0.py && python3 tools/export_tadsr_unet_downblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION0_BRIDGE_ALIGNMENT | PASS | down_blocks.2.attentions.0 aligned after complete local down_blocks.0/down_blocks.1/down_blocks.2.resnets.0 bridge; remaining down_blocks.2 modules are intentionally unopened | continue with down_blocks.2.resnets.1 only |
| TADSR_UNET_DOWNBLOCK2_PRE_RESNET1_ALIGNMENT | PASS | down_blocks.2 path is aligned through resnets.0 -> attentions.0 only | continue with down_blocks.2.resnets.1 only |
| TADSR_UNET_DOWNBLOCK2_RESNET1_AUDIT | PASS | official down_blocks.2.resnets.1 audit | python3 tools/audit_official_tadsr_unet_downblock2_resnet1.py && python3 tools/export_tadsr_unet_downblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET1_LORA_AUDIT | PASS | LoRA/effective weight state for down_blocks.2.resnets.1 | python3 tools/audit_official_tadsr_unet_downblock2_resnet1.py && python3 tools/export_tadsr_unet_downblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET1_ORACLE_TENSORS | PASS | PyTorch oracle tensors for down_blocks.2.resnets.1 exported | python3 tools/audit_official_tadsr_unet_downblock2_resnet1.py && python3 tools/export_tadsr_unet_downblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET1_EFFECTIVE_WEIGHTS | PASS | effective static weights for down_blocks.2.resnets.1 exported locally | python3 tools/audit_official_tadsr_unet_downblock2_resnet1.py && python3 tools/export_tadsr_unet_downblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET1_NORM1_ALIGNMENT | PASS | down_blocks.2.resnets.1 norm1 alignment | python3 tools/audit_official_tadsr_unet_downblock2_resnet1.py && python3 tools/export_tadsr_unet_downblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET1_CONV1_ALIGNMENT | PASS | down_blocks.2.resnets.1 conv1 alignment | python3 tools/audit_official_tadsr_unet_downblock2_resnet1.py && python3 tools/export_tadsr_unet_downblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET1_TEMB_PROJ_ALIGNMENT | PASS | down_blocks.2.resnets.1 time_emb_proj alignment | python3 tools/audit_official_tadsr_unet_downblock2_resnet1.py && python3 tools/export_tadsr_unet_downblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET1_CONV2_ALIGNMENT | PASS | down_blocks.2.resnets.1 conv2 alignment | python3 tools/audit_official_tadsr_unet_downblock2_resnet1.py && python3 tools/export_tadsr_unet_downblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET1_ALIGNMENT | PASS | down_blocks.2.resnets.1 local alignment | python3 tools/audit_official_tadsr_unet_downblock2_resnet1.py && python3 tools/export_tadsr_unet_downblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET1_SYNTHETIC_ALIGNMENT | PASS | isolated synthetic hidden/temb -> down_blocks.2.resnets.1 alignment | python3 tools/audit_official_tadsr_unet_downblock2_resnet1.py && python3 tools/export_tadsr_unet_downblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ATTENTION0_RESNET1_ALIGNMENT | PASS | UNet entry -> full local down_blocks.0/down_blocks.1 -> down_blocks.2.resnets.0 -> attentions.0 -> resnets.1 bridge alignment | python3 tools/audit_official_tadsr_unet_downblock2_resnet1.py && python3 tools/export_tadsr_unet_downblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_RESNET1_SHORTCUT_ALIGNMENT | NOT_APPLICABLE | down_blocks.2.resnets.1 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent | continue only if shortcut remains PASS or NOT_APPLICABLE |
| TADSR_UNET_DOWNBLOCK2_RESNET1_BRIDGE_ALIGNMENT | PASS | down_blocks.2.resnets.1 aligned after complete local down_blocks.0/down_blocks.1/down_blocks.2.resnets.0/down_blocks.2.attentions.0 bridge; remaining down_blocks.2 modules are intentionally unopened | continue with down_blocks.2.attentions.1 only |
| TADSR_UNET_DOWNBLOCK2_PRE_ATTENTION1_ALIGNMENT | PASS | down_blocks.2 path is aligned through resnets.0 -> attentions.0 -> resnets.1 only | continue with down_blocks.2.attentions.1 only |
| TADSR_UNET_DOWNBLOCK2_ATTENTION1_AUDIT | PASS | official down_blocks.2.attentions.1 audit | python3 tools/audit_official_tadsr_unet_downblock2_attention1.py && python3 tools/export_tadsr_unet_downblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION1_ORACLE_TENSORS | PASS | PyTorch oracle tensors for down_blocks.2.attentions.1 exported | python3 tools/audit_official_tadsr_unet_downblock2_attention1.py && python3 tools/export_tadsr_unet_downblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION1_EFFECTIVE_WEIGHTS | PASS | effective static weights for down_blocks.2.attentions.1 exported locally | python3 tools/audit_official_tadsr_unet_downblock2_attention1.py && python3 tools/export_tadsr_unet_downblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION1_NORM_ALIGNMENT | PASS | down_blocks.2.attentions.1 top-level norm alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention1.py && python3 tools/export_tadsr_unet_downblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION1_PROJ_IN_ALIGNMENT | PASS | down_blocks.2.attentions.1 proj_in alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention1.py && python3 tools/export_tadsr_unet_downblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION1_SEQUENCE_ALIGNMENT | PASS | down_blocks.2.attentions.1 NCHW-to-sequence alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention1.py && python3 tools/export_tadsr_unet_downblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION1_ATTN1_QKV_ALIGNMENT | PASS | down_blocks.2.attentions.1 transformer0 attn1 QKV alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention1.py && python3 tools/export_tadsr_unet_downblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION1_ATTN1_ALIGNMENT | PASS | down_blocks.2.attentions.1 transformer0 self-attention output alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention1.py && python3 tools/export_tadsr_unet_downblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION1_AFTER_ATTN1_ALIGNMENT | PASS | down_blocks.2.attentions.1 after-attn1 residual alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention1.py && python3 tools/export_tadsr_unet_downblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION1_ATTN2_QKV_ALIGNMENT | PASS | down_blocks.2.attentions.1 transformer0 attn2 QKV alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention1.py && python3 tools/export_tadsr_unet_downblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION1_ATTN2_ALIGNMENT | PASS | down_blocks.2.attentions.1 transformer0 cross-attention output alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention1.py && python3 tools/export_tadsr_unet_downblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION1_AFTER_ATTN2_ALIGNMENT | PASS | down_blocks.2.attentions.1 after-attn2 residual alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention1.py && python3 tools/export_tadsr_unet_downblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION1_FF_ALIGNMENT | PASS | down_blocks.2.attentions.1 feed-forward output alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention1.py && python3 tools/export_tadsr_unet_downblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION1_TRANSFORMER0_ALIGNMENT | PASS | down_blocks.2.attentions.1 full transformer block alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention1.py && python3 tools/export_tadsr_unet_downblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION1_ALIGNMENT | PASS | isolated down_blocks.2.attentions.1 alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention1.py && python3 tools/export_tadsr_unet_downblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT | PASS | UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2.resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 bridge alignment | python3 tools/audit_official_tadsr_unet_downblock2_attention1.py && python3 tools/export_tadsr_unet_downblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ATTENTION1_BRIDGE_ALIGNMENT | PASS | down_blocks.2.attentions.1 aligned after complete local down_blocks.0/down_blocks.1/down_blocks.2.resnets.0/down_blocks.2.attentions.0/down_blocks.2.resnets.1 bridge; downsampler remains intentionally unopened | continue with down_blocks.2.downsamplers.0 only |
| TADSR_UNET_DOWNBLOCK2_PRE_DOWNSAMPLER_ALIGNMENT | PASS | down_blocks.2 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 only | continue with down_blocks.2.downsamplers.0 only |
| TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_AUDIT | PASS | official down_blocks.2.downsamplers.0 audit | python3 tools/audit_official_tadsr_unet_downblock2_downsampler.py && python3 tools/export_tadsr_unet_downblock2_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_LORA_AUDIT | PASS | LoRA/effective weight state for down_blocks.2.downsamplers.0 | python3 tools/audit_official_tadsr_unet_downblock2_downsampler.py && python3 tools/export_tadsr_unet_downblock2_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_LOCAL_FORWARD_AUDIT | PASS | manual local down_blocks.2 chain matches official block.forward | python3 tools/audit_official_tadsr_unet_downblock2_downsampler.py && python3 tools/export_tadsr_unet_downblock2_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_OUTPUT_STATES_AUDIT | PASS | official down_blocks.2 residual output_states tuple audited | python3 tools/audit_official_tadsr_unet_downblock2_downsampler.py && python3 tools/export_tadsr_unet_downblock2_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_ORACLE_TENSORS | PASS | PyTorch oracle tensors for down_blocks.2.downsamplers.0 exported | python3 tools/audit_official_tadsr_unet_downblock2_downsampler.py && python3 tools/export_tadsr_unet_downblock2_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_EFFECTIVE_WEIGHTS | PASS | effective static weights for down_blocks.2.downsamplers.0 exported locally | python3 tools/audit_official_tadsr_unet_downblock2_downsampler.py && python3 tools/export_tadsr_unet_downblock2_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_PADDING_ALIGNMENT | PASS | down_blocks.2.downsampler padding alignment; NOT_APPLICABLE for direct conv path | python3 tools/audit_official_tadsr_unet_downblock2_downsampler.py && python3 tools/export_tadsr_unet_downblock2_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_CONV_ALIGNMENT | PASS | down_blocks.2.downsampler conv output alignment | python3 tools/audit_official_tadsr_unet_downblock2_downsampler.py && python3 tools/export_tadsr_unet_downblock2_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_ALIGNMENT | PASS | isolated down_blocks.2.downsampler alignment | python3 tools/audit_official_tadsr_unet_downblock2_downsampler.py && python3 tools/export_tadsr_unet_downblock2_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_DOWNSAMPLER_ALIGNMENT | PASS | UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2 local chain through downsampler bridge alignment | python3 tools/audit_official_tadsr_unet_downblock2_downsampler.py && python3 tools/export_tadsr_unet_downblock2_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_OUTPUT_HIDDEN_ALIGNMENT | PASS | down_blocks.2 final hidden_states output alignment | python3 tools/audit_official_tadsr_unet_downblock2_downsampler.py && python3 tools/export_tadsr_unet_downblock2_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_OUTPUT_STATES_ALIGNMENT | PASS | down_blocks.2 residual output_states tuple alignment | python3 tools/audit_official_tadsr_unet_downblock2_downsampler.py && python3 tools/export_tadsr_unet_downblock2_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_ALIGNMENT | PASS | full local down_blocks.2 alignment through output_states tuple | python3 tools/audit_official_tadsr_unet_downblock2_downsampler.py && python3 tools/export_tadsr_unet_downblock2_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK2_BRIDGE_ALIGNMENT | PASS | complete local down_blocks.2 bridge aligned through downsampler and output_states; later UNet blocks remain unopened | continue with down_blocks.3.resnets.0 only |
| TADSR_UNET_DOWNBLOCK3_AUDIT | PASS | official UNet down_blocks.3 topology audit | python3 tools/audit_official_tadsr_unet_downblock3_resnet0.py && python3 tools/export_tadsr_unet_downblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET0_AUDIT | PASS | official down_blocks.3.resnets.0 audit | python3 tools/audit_official_tadsr_unet_downblock3_resnet0.py && python3 tools/export_tadsr_unet_downblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET0_LORA_AUDIT | PASS | LoRA/effective weight state for down_blocks.3.resnets.0 | python3 tools/audit_official_tadsr_unet_downblock3_resnet0.py && python3 tools/export_tadsr_unet_downblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET0_ORACLE_TENSORS | PASS | PyTorch oracle tensors for down_blocks.3.resnets.0 exported | python3 tools/audit_official_tadsr_unet_downblock3_resnet0.py && python3 tools/export_tadsr_unet_downblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET0_EFFECTIVE_WEIGHTS | PASS | effective static weights for down_blocks.3.resnets.0 exported locally | python3 tools/audit_official_tadsr_unet_downblock3_resnet0.py && python3 tools/export_tadsr_unet_downblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET0_NORM1_ALIGNMENT | PASS | down_blocks.3.resnets.0 norm1 alignment | python3 tools/audit_official_tadsr_unet_downblock3_resnet0.py && python3 tools/export_tadsr_unet_downblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET0_CONV1_ALIGNMENT | PASS | down_blocks.3.resnets.0 conv1 alignment | python3 tools/audit_official_tadsr_unet_downblock3_resnet0.py && python3 tools/export_tadsr_unet_downblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET0_TEMB_PROJ_ALIGNMENT | PASS | down_blocks.3.resnets.0 time_emb_proj alignment | python3 tools/audit_official_tadsr_unet_downblock3_resnet0.py && python3 tools/export_tadsr_unet_downblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET0_CONV2_ALIGNMENT | PASS | down_blocks.3.resnets.0 conv2 alignment | python3 tools/audit_official_tadsr_unet_downblock3_resnet0.py && python3 tools/export_tadsr_unet_downblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET0_ALIGNMENT | PASS | down_blocks.3.resnets.0 local alignment | python3 tools/audit_official_tadsr_unet_downblock3_resnet0.py && python3 tools/export_tadsr_unet_downblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET0_SYNTHETIC_ALIGNMENT | PASS | isolated synthetic hidden/temb -> down_blocks.3.resnets.0 alignment | python3 tools/audit_official_tadsr_unet_downblock3_resnet0.py && python3 tools/export_tadsr_unet_downblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_DOWNBLOCK3_RESNET0_ALIGNMENT | PASS | UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2 -> down_blocks.3.resnets.0 bridge alignment | python3 tools/audit_official_tadsr_unet_downblock3_resnet0.py && python3 tools/export_tadsr_unet_downblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET0_SHORTCUT_ALIGNMENT | NOT_APPLICABLE | down_blocks.3.resnets.0 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent | continue only if shortcut remains PASS or NOT_APPLICABLE |
| TADSR_UNET_DOWNBLOCK3_RESNET0_BRIDGE_ALIGNMENT | PASS | down_blocks.3.resnets.0 aligned after complete local down_blocks.0/1/2 bridge; remaining down_blocks.3 modules are intentionally unopened | continue with down_blocks.3.resnets.1 only |
| TADSR_UNET_DOWNBLOCK3_RESNET1_AUDIT | PASS | official down_blocks.3.resnets.1 audit | python3 tools/audit_official_tadsr_unet_downblock3_resnet1.py && python3 tools/export_tadsr_unet_downblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_LOCAL_FORWARD_AUDIT | PASS | official down_blocks.3 local forward hidden output audited | python3 tools/audit_official_tadsr_unet_downblock3_resnet1.py && python3 tools/export_tadsr_unet_downblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_OUTPUT_STATES_AUDIT | PASS | official down_blocks.3 residual output_states tuple audited | python3 tools/audit_official_tadsr_unet_downblock3_resnet1.py && python3 tools/export_tadsr_unet_downblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET1_ORACLE_TENSORS | PASS | PyTorch oracle tensors for down_blocks.3.resnets.1 exported | python3 tools/audit_official_tadsr_unet_downblock3_resnet1.py && python3 tools/export_tadsr_unet_downblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET1_EFFECTIVE_WEIGHTS | PASS | effective static weights for down_blocks.3.resnets.1 exported locally | python3 tools/audit_official_tadsr_unet_downblock3_resnet1.py && python3 tools/export_tadsr_unet_downblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET1_NORM1_ALIGNMENT | PASS | down_blocks.3.resnets.1 norm1 alignment | python3 tools/audit_official_tadsr_unet_downblock3_resnet1.py && python3 tools/export_tadsr_unet_downblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET1_CONV1_ALIGNMENT | PASS | down_blocks.3.resnets.1 conv1 alignment | python3 tools/audit_official_tadsr_unet_downblock3_resnet1.py && python3 tools/export_tadsr_unet_downblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET1_TEMB_PROJ_ALIGNMENT | PASS | down_blocks.3.resnets.1 time_emb_proj alignment | python3 tools/audit_official_tadsr_unet_downblock3_resnet1.py && python3 tools/export_tadsr_unet_downblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET1_CONV2_ALIGNMENT | PASS | down_blocks.3.resnets.1 conv2 alignment | python3 tools/audit_official_tadsr_unet_downblock3_resnet1.py && python3 tools/export_tadsr_unet_downblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET1_ALIGNMENT | PASS | down_blocks.3.resnets.1 local alignment | python3 tools/audit_official_tadsr_unet_downblock3_resnet1.py && python3 tools/export_tadsr_unet_downblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET1_SYNTHETIC_ALIGNMENT | PASS | isolated synthetic hidden/temb -> down_blocks.3.resnets.1 alignment | python3 tools/audit_official_tadsr_unet_downblock3_resnet1.py && python3 tools/export_tadsr_unet_downblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_DOWNBLOCK3_RESNET0_RESNET1_ALIGNMENT | PASS | UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2 -> down_blocks.3.resnets.0 -> resnets.1 bridge alignment | python3 tools/audit_official_tadsr_unet_downblock3_resnet1.py && python3 tools/export_tadsr_unet_downblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_OUTPUT_HIDDEN_ALIGNMENT | PASS | down_blocks.3 final hidden_states output alignment | python3 tools/audit_official_tadsr_unet_downblock3_resnet1.py && python3 tools/export_tadsr_unet_downblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_OUTPUT_STATES_ALIGNMENT | PASS | down_blocks.3 residual output_states tuple alignment | python3 tools/audit_official_tadsr_unet_downblock3_resnet1.py && python3 tools/export_tadsr_unet_downblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_ALIGNMENT | PASS | full local down_blocks.3 alignment through output_states tuple | python3 tools/audit_official_tadsr_unet_downblock3_resnet1.py && python3 tools/export_tadsr_unet_downblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_DOWNBLOCK3_RESNET1_SHORTCUT_ALIGNMENT | NOT_APPLICABLE | down_blocks.3.resnets.1 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent | continue only if shortcut remains PASS or NOT_APPLICABLE |
| TADSR_UNET_DOWNBLOCK3_BRIDGE_ALIGNMENT | PASS | complete local down_blocks.3 bridge aligned through resnet0 -> resnet1 and output_states; mid_block remains unopened | continue with mid_block.resnets.0 only |
| TADSR_UNET_MIDBLOCK_AUDIT | PASS | official UNet mid_block topology audit | python3 tools/audit_official_tadsr_unet_midblock_resnet0.py && python3 tools/export_tadsr_unet_midblock_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET0_AUDIT | PASS | official mid_block.resnets.0 audit | python3 tools/audit_official_tadsr_unet_midblock_resnet0.py && python3 tools/export_tadsr_unet_midblock_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET0_LORA_AUDIT | PASS | LoRA/effective weight state for mid_block.resnets.0 | python3 tools/audit_official_tadsr_unet_midblock_resnet0.py && python3 tools/export_tadsr_unet_midblock_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET0_ORACLE_TENSORS | PASS | PyTorch oracle tensors for mid_block.resnets.0 exported | python3 tools/audit_official_tadsr_unet_midblock_resnet0.py && python3 tools/export_tadsr_unet_midblock_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET0_EFFECTIVE_WEIGHTS | PASS | effective static weights for mid_block.resnets.0 exported locally | python3 tools/audit_official_tadsr_unet_midblock_resnet0.py && python3 tools/export_tadsr_unet_midblock_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET0_NORM1_ALIGNMENT | PASS | mid_block.resnets.0 norm1 alignment | python3 tools/audit_official_tadsr_unet_midblock_resnet0.py && python3 tools/export_tadsr_unet_midblock_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET0_CONV1_ALIGNMENT | PASS | mid_block.resnets.0 conv1 alignment | python3 tools/audit_official_tadsr_unet_midblock_resnet0.py && python3 tools/export_tadsr_unet_midblock_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET0_TEMB_PROJ_ALIGNMENT | PASS | mid_block.resnets.0 time_emb_proj alignment | python3 tools/audit_official_tadsr_unet_midblock_resnet0.py && python3 tools/export_tadsr_unet_midblock_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET0_CONV2_ALIGNMENT | PASS | mid_block.resnets.0 conv2 alignment | python3 tools/audit_official_tadsr_unet_midblock_resnet0.py && python3 tools/export_tadsr_unet_midblock_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET0_ALIGNMENT | PASS | mid_block.resnets.0 local alignment | python3 tools/audit_official_tadsr_unet_midblock_resnet0.py && python3 tools/export_tadsr_unet_midblock_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET0_SYNTHETIC_ALIGNMENT | PASS | isolated synthetic hidden/temb -> mid_block.resnets.0 alignment | python3 tools/audit_official_tadsr_unet_midblock_resnet0.py && python3 tools/export_tadsr_unet_midblock_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_RESNET0_ALIGNMENT | PASS | UNet entry -> down_blocks.0/1/2/3 -> mid_block.resnets.0 bridge alignment | python3 tools/audit_official_tadsr_unet_midblock_resnet0.py && python3 tools/export_tadsr_unet_midblock_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET0_SHORTCUT_ALIGNMENT | NOT_APPLICABLE | mid_block.resnets.0 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent | continue only if shortcut remains PASS or NOT_APPLICABLE |
| TADSR_UNET_MIDBLOCK_RESNET0_BRIDGE_ALIGNMENT | PASS | mid_block.resnets.0 aligned after complete local down_blocks.0/1/2/3 bridge; attention/resnet1 remain unopened | continue with mid_block.attentions.0 if present |
| TADSR_UNET_MIDBLOCK_ATTENTION0_AUDIT | PASS | official mid_block.attentions.0 audit | python3 tools/audit_official_tadsr_unet_midblock_attention0.py && python3 tools/export_tadsr_unet_midblock_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_ATTENTION0_LORA_AUDIT | PASS | LoRA/effective weight state for mid_block.attentions.0 | python3 tools/audit_official_tadsr_unet_midblock_attention0.py && python3 tools/export_tadsr_unet_midblock_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_ATTENTION0_ORACLE_TENSORS | PASS | PyTorch oracle tensors for mid_block.attentions.0 exported | python3 tools/audit_official_tadsr_unet_midblock_attention0.py && python3 tools/export_tadsr_unet_midblock_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_ATTENTION0_EFFECTIVE_WEIGHTS | PASS | effective static weights for mid_block.attentions.0 exported locally | python3 tools/audit_official_tadsr_unet_midblock_attention0.py && python3 tools/export_tadsr_unet_midblock_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_ATTENTION0_NORM_ALIGNMENT | PASS | mid_block.attentions.0 top-level norm alignment | python3 tools/audit_official_tadsr_unet_midblock_attention0.py && python3 tools/export_tadsr_unet_midblock_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_ATTENTION0_PROJ_IN_ALIGNMENT | PASS | mid_block.attentions.0 proj_in alignment | python3 tools/audit_official_tadsr_unet_midblock_attention0.py && python3 tools/export_tadsr_unet_midblock_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_ATTENTION0_SEQUENCE_ALIGNMENT | PASS | mid_block.attentions.0 NCHW-to-sequence alignment | python3 tools/audit_official_tadsr_unet_midblock_attention0.py && python3 tools/export_tadsr_unet_midblock_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_ATTENTION0_ATTN1_QKV_ALIGNMENT | PASS | mid_block.attentions.0 self-attention QKV alignment | python3 tools/audit_official_tadsr_unet_midblock_attention0.py && python3 tools/export_tadsr_unet_midblock_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_ATTENTION0_ATTN1_ALIGNMENT | PASS | mid_block.attentions.0 self-attention output alignment | python3 tools/audit_official_tadsr_unet_midblock_attention0.py && python3 tools/export_tadsr_unet_midblock_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_ATTENTION0_AFTER_ATTN1_ALIGNMENT | PASS | mid_block.attentions.0 after-attn1 residual alignment | python3 tools/audit_official_tadsr_unet_midblock_attention0.py && python3 tools/export_tadsr_unet_midblock_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_ATTENTION0_ATTN2_QKV_ALIGNMENT | PASS | mid_block.attentions.0 cross-attention QKV alignment | python3 tools/audit_official_tadsr_unet_midblock_attention0.py && python3 tools/export_tadsr_unet_midblock_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_ATTENTION0_ATTN2_ALIGNMENT | PASS | mid_block.attentions.0 cross-attention output alignment | python3 tools/audit_official_tadsr_unet_midblock_attention0.py && python3 tools/export_tadsr_unet_midblock_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_ATTENTION0_AFTER_ATTN2_ALIGNMENT | PASS | mid_block.attentions.0 after-attn2 residual alignment | python3 tools/audit_official_tadsr_unet_midblock_attention0.py && python3 tools/export_tadsr_unet_midblock_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_ATTENTION0_FF_ALIGNMENT | PASS | mid_block.attentions.0 feed-forward output alignment | python3 tools/audit_official_tadsr_unet_midblock_attention0.py && python3 tools/export_tadsr_unet_midblock_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_ATTENTION0_TRANSFORMER0_ALIGNMENT | PASS | mid_block.attentions.0 transformer block alignment | python3 tools/audit_official_tadsr_unet_midblock_attention0.py && python3 tools/export_tadsr_unet_midblock_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_ATTENTION0_ALIGNMENT | PASS | isolated mid_block.attentions.0 alignment | python3 tools/audit_official_tadsr_unet_midblock_attention0.py && python3 tools/export_tadsr_unet_midblock_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_RESNET0_ATTENTION0_ALIGNMENT | PASS | UNet entry -> down_blocks.0/1/2/3 -> mid_block.resnets.0 -> mid_block.attentions.0 bridge alignment | python3 tools/audit_official_tadsr_unet_midblock_attention0.py && python3 tools/export_tadsr_unet_midblock_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_ATTENTION0_BRIDGE_ALIGNMENT | PASS | mid_block.attentions.0 aligned in isolated and complete pre-attention bridge paths; resnet1 remains unopened | continue with mid_block.resnets.1 |
| TADSR_UNET_MIDBLOCK_PRE_RESNET1_ALIGNMENT | PASS | UNet path through mid_block.resnets.0 and mid_block.attentions.0 is aligned; mid_block.resnets.1 is evaluated separately below | continue with mid_block.resnets.1 |
| TADSR_UNET_MIDBLOCK_RESNET1_AUDIT | PASS | official mid_block.resnets.1 audit | python3 tools/audit_official_tadsr_unet_midblock_resnet1.py && python3 tools/export_tadsr_unet_midblock_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_LOCAL_FORWARD_AUDIT | PASS | official local mid_block forward audit | python3 tools/audit_official_tadsr_unet_midblock_resnet1.py && python3 tools/export_tadsr_unet_midblock_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET1_LORA_AUDIT | PASS | LoRA/effective weight state for mid_block.resnets.1 | python3 tools/audit_official_tadsr_unet_midblock_resnet1.py && python3 tools/export_tadsr_unet_midblock_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET1_ORACLE_TENSORS | PASS | PyTorch oracle tensors for mid_block.resnets.1 exported | python3 tools/audit_official_tadsr_unet_midblock_resnet1.py && python3 tools/export_tadsr_unet_midblock_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET1_EFFECTIVE_WEIGHTS | PASS | effective static weights for mid_block.resnets.1 exported locally | python3 tools/audit_official_tadsr_unet_midblock_resnet1.py && python3 tools/export_tadsr_unet_midblock_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET1_NORM1_ALIGNMENT | PASS | mid_block.resnets.1 norm1 alignment | python3 tools/audit_official_tadsr_unet_midblock_resnet1.py && python3 tools/export_tadsr_unet_midblock_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET1_CONV1_ALIGNMENT | PASS | mid_block.resnets.1 conv1 alignment | python3 tools/audit_official_tadsr_unet_midblock_resnet1.py && python3 tools/export_tadsr_unet_midblock_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET1_TEMB_PROJ_ALIGNMENT | PASS | mid_block.resnets.1 time_emb_proj alignment | python3 tools/audit_official_tadsr_unet_midblock_resnet1.py && python3 tools/export_tadsr_unet_midblock_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET1_CONV2_ALIGNMENT | PASS | mid_block.resnets.1 conv2 alignment | python3 tools/audit_official_tadsr_unet_midblock_resnet1.py && python3 tools/export_tadsr_unet_midblock_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET1_ALIGNMENT | PASS | isolated entry-input mid_block.resnets.1 alignment | python3 tools/audit_official_tadsr_unet_midblock_resnet1.py && python3 tools/export_tadsr_unet_midblock_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET1_SYNTHETIC_ALIGNMENT | PASS | synthetic hidden/temb -> mid_block.resnets.1 alignment | python3 tools/audit_official_tadsr_unet_midblock_resnet1.py && python3 tools/export_tadsr_unet_midblock_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_RESNET0_ATTENTION0_RESNET1_ALIGNMENT | PASS | UNet entry -> down_blocks.0/1/2/3 -> mid_block.resnets.0 -> attention0 -> resnets.1 bridge alignment | python3 tools/audit_official_tadsr_unet_midblock_resnet1.py && python3 tools/export_tadsr_unet_midblock_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_OUTPUT_HIDDEN_ALIGNMENT | PASS | manual local mid_block hidden output vs official local mid_block forward | python3 tools/audit_official_tadsr_unet_midblock_resnet1.py && python3 tools/export_tadsr_unet_midblock_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_MIDBLOCK_RESNET1_SHORTCUT_ALIGNMENT | NOT_APPLICABLE | mid_block.resnets.1 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent | continue only if shortcut remains PASS or NOT_APPLICABLE |
| TADSR_UNET_MIDBLOCK_OUTPUT_STATES_ALIGNMENT | NOT_APPLICABLE | official local mid_block output_states status; NOT_APPLICABLE when mid_block returns hidden_states only | do not invent output_states |
| TADSR_UNET_MIDBLOCK_BRIDGE_ALIGNMENT | PASS | complete local mid_block chain resnets.0 -> attentions.0 -> resnets.1 aligned; up_blocks remain unopened | continue with up_blocks.0 if PASS |
| TADSR_UNET_MIDBLOCK_ALIGNMENT | PASS | local mid_block hidden output aligned through resnets.1; this is not full UNet forward | continue with up_blocks.0 |
| TADSR_UNET_UPBLOCK0_AUDIT | PASS | official up_blocks.0 topology audit | python3 tools/audit_official_tadsr_unet_upblock0_resnet0.py && python3 tools/export_tadsr_unet_upblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET0_AUDIT | PASS | official up_blocks.0.resnets.0 audit | python3 tools/audit_official_tadsr_unet_upblock0_resnet0.py && python3 tools/export_tadsr_unet_upblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET0_RESIDUAL_CONTRACT_AUDIT | PASS | official residual pop/concat contract audit | python3 tools/audit_official_tadsr_unet_upblock0_resnet0.py && python3 tools/export_tadsr_unet_upblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET0_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.0.resnets.0 exported | python3 tools/audit_official_tadsr_unet_upblock0_resnet0.py && python3 tools/export_tadsr_unet_upblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET0_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.0.resnets.0 exported locally | python3 tools/audit_official_tadsr_unet_upblock0_resnet0.py && python3 tools/export_tadsr_unet_upblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET0_RESIDUAL_CONCAT_ALIGNMENT | PASS | up_blocks.0.resnets.0 hidden/residual concat alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet0.py && python3 tools/export_tadsr_unet_upblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET0_NORM1_ALIGNMENT | PASS | up_blocks.0.resnets.0 norm1 alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet0.py && python3 tools/export_tadsr_unet_upblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET0_CONV1_ALIGNMENT | PASS | up_blocks.0.resnets.0 conv1 alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet0.py && python3 tools/export_tadsr_unet_upblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET0_TEMB_PROJ_ALIGNMENT | PASS | up_blocks.0.resnets.0 time_emb_proj alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet0.py && python3 tools/export_tadsr_unet_upblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET0_CONV2_ALIGNMENT | PASS | up_blocks.0.resnets.0 conv2 alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet0.py && python3 tools/export_tadsr_unet_upblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET0_ALIGNMENT | PASS | isolated up_blocks.0.resnets.0 after official residual concat alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet0.py && python3 tools/export_tadsr_unet_upblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET0_SYNTHETIC_ALIGNMENT | PASS | synthetic hidden/residual -> up_blocks.0.resnets.0 alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet0.py && python3 tools/export_tadsr_unet_upblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_RESNET0_ALIGNMENT | PASS | UNet entry -> down_blocks.0/1/2/3 -> mid_block -> up_blocks.0.resnets.0 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet0.py && python3 tools/export_tadsr_unet_upblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET0_SHORTCUT_ALIGNMENT | PASS | up_blocks.0.resnets.0 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent | continue only if shortcut remains PASS or NOT_APPLICABLE |
| TADSR_UNET_UPBLOCK0_RESNET0_BRIDGE_ALIGNMENT | PASS | first up path leaf up_blocks.0.resnets.0 aligned after official residual consumption; full up_blocks.0 remains unopened | continue with actual next module after up_blocks.0.resnets.0 |
| TADSR_UNET_UPBLOCK0_RESNET1_AUDIT | BLOCKED | official up_blocks.0.resnets.1 audit | python3 tools/audit_official_tadsr_unet_upblock0_resnet1.py && python3 tools/export_tadsr_unet_upblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET1_RESIDUAL_CONTRACT_AUDIT | BLOCKED | official residual pop/concat contract audit for second upblock residual | python3 tools/audit_official_tadsr_unet_upblock0_resnet1.py && python3 tools/export_tadsr_unet_upblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET1_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.0.resnets.1 exported | python3 tools/audit_official_tadsr_unet_upblock0_resnet1.py && python3 tools/export_tadsr_unet_upblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET1_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.0.resnets.1 exported locally | python3 tools/audit_official_tadsr_unet_upblock0_resnet1.py && python3 tools/export_tadsr_unet_upblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET1_RESIDUAL_CONCAT_ALIGNMENT | BLOCKED | up_blocks.0.resnets.1 hidden/residual concat alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet1.py && python3 tools/export_tadsr_unet_upblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET1_NORM1_ALIGNMENT | BLOCKED | up_blocks.0.resnets.1 norm1 alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet1.py && python3 tools/export_tadsr_unet_upblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET1_CONV1_ALIGNMENT | BLOCKED | up_blocks.0.resnets.1 conv1 alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet1.py && python3 tools/export_tadsr_unet_upblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET1_TEMB_PROJ_ALIGNMENT | BLOCKED | up_blocks.0.resnets.1 time_emb_proj alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet1.py && python3 tools/export_tadsr_unet_upblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET1_CONV2_ALIGNMENT | BLOCKED | up_blocks.0.resnets.1 conv2 alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet1.py && python3 tools/export_tadsr_unet_upblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET1_ALIGNMENT | BLOCKED | isolated up_blocks.0.resnets.1 after official residual concat alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet1.py && python3 tools/export_tadsr_unet_upblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET1_SYNTHETIC_ALIGNMENT | BLOCKED | synthetic hidden/residual -> up_blocks.0.resnets.1 alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet1.py && python3 tools/export_tadsr_unet_upblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_RESNET0_RESNET1_ALIGNMENT | BLOCKED | UNet entry -> down_blocks.0/1/2/3 -> mid_block -> up_blocks.0.resnets.0 -> resnets.1 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet1.py && python3 tools/export_tadsr_unet_upblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET1_SHORTCUT_ALIGNMENT | BLOCKED | up_blocks.0.resnets.1 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent | continue only if shortcut remains PASS or NOT_APPLICABLE |
| TADSR_UNET_UPBLOCK0_RESNET1_BRIDGE_ALIGNMENT | PARTIAL | up_blocks.0.resnets.1 aligned after official second residual consumption; full up_blocks.0 remains unopened | continue with actual next module after up_blocks.0.resnets.1 |
| TADSR_UNET_UPBLOCK0_PRE_RESNET2_ALIGNMENT | PARTIAL | up_blocks.0 path is aligned through resnets.0 -> resnets.1 only | continue with up_blocks.0.resnets.2 |
| TADSR_UNET_UPBLOCK0_RESNET2_AUDIT | PASS | official up_blocks.0.resnets.2 audit | python3 tools/audit_official_tadsr_unet_upblock0_resnet2.py && python3 tools/export_tadsr_unet_upblock0_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET2_RESIDUAL_CONTRACT_AUDIT | PASS | official residual pop/concat contract audit for third upblock residual | python3 tools/audit_official_tadsr_unet_upblock0_resnet2.py && python3 tools/export_tadsr_unet_upblock0_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET2_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.0.resnets.2 exported | python3 tools/audit_official_tadsr_unet_upblock0_resnet2.py && python3 tools/export_tadsr_unet_upblock0_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET2_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.0.resnets.2 exported locally | python3 tools/audit_official_tadsr_unet_upblock0_resnet2.py && python3 tools/export_tadsr_unet_upblock0_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET2_RESIDUAL_CONCAT_ALIGNMENT | PASS | up_blocks.0.resnets.2 hidden/residual concat alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet2.py && python3 tools/export_tadsr_unet_upblock0_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET2_NORM1_ALIGNMENT | PASS | up_blocks.0.resnets.2 norm1 alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet2.py && python3 tools/export_tadsr_unet_upblock0_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET2_CONV1_ALIGNMENT | PASS | up_blocks.0.resnets.2 conv1 alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet2.py && python3 tools/export_tadsr_unet_upblock0_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET2_TEMB_PROJ_ALIGNMENT | PASS | up_blocks.0.resnets.2 time_emb_proj alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet2.py && python3 tools/export_tadsr_unet_upblock0_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET2_CONV2_ALIGNMENT | PASS | up_blocks.0.resnets.2 conv2 alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet2.py && python3 tools/export_tadsr_unet_upblock0_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET2_ALIGNMENT | PASS | isolated up_blocks.0.resnets.2 after official residual concat alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet2.py && python3 tools/export_tadsr_unet_upblock0_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET2_SYNTHETIC_ALIGNMENT | PASS | synthetic hidden/residual -> up_blocks.0.resnets.2 alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet2.py && python3 tools/export_tadsr_unet_upblock0_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_RESNET0_RESNET1_RESNET2_ALIGNMENT | PASS | UNet entry -> down_blocks.0/1/2/3 -> mid_block -> up_blocks.0.resnets.0 -> resnets.1 -> resnets.2 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock0_resnet2.py && python3 tools/export_tadsr_unet_upblock0_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_RESNET2_SHORTCUT_ALIGNMENT | PASS | up_blocks.0.resnets.2 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent | continue only if shortcut remains PASS or NOT_APPLICABLE |
| TADSR_UNET_UPBLOCK0_RESNET2_BRIDGE_ALIGNMENT | PARTIAL | up_blocks.0.resnets.2 aligned after official third residual consumption; full up_blocks.0 remains unopened | continue with up_blocks.0.upsamplers.0 |
| TADSR_UNET_UPBLOCK0_PRE_UPSAMPLER_ALIGNMENT | PARTIAL | up_blocks.0 path is aligned through resnets.0 -> resnets.1 -> resnets.2 only | continue with up_blocks.0.upsamplers.0 |
| TADSR_UNET_UPBLOCK0_UPSAMPLER_AUDIT | PASS | official up_blocks.0.upsamplers.0 audit | python3 tools/audit_official_tadsr_unet_upblock0_upsampler.py && python3 tools/export_tadsr_unet_upblock0_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_LOCAL_FORWARD_AUDIT | PASS | safe official local up_blocks.0 hidden-state forward audit | python3 tools/audit_official_tadsr_unet_upblock0_upsampler.py && python3 tools/export_tadsr_unet_upblock0_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_UPSAMPLER_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.0.upsamplers.0 exported | python3 tools/audit_official_tadsr_unet_upblock0_upsampler.py && python3 tools/export_tadsr_unet_upblock0_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_UPSAMPLER_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.0.upsamplers.0 exported locally | python3 tools/audit_official_tadsr_unet_upblock0_upsampler.py && python3 tools/export_tadsr_unet_upblock0_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_LOCAL_FORWARD_ORACLE | PASS | PyTorch oracle for local up_blocks.0 hidden-state output exported | python3 tools/audit_official_tadsr_unet_upblock0_upsampler.py && python3 tools/export_tadsr_unet_upblock0_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_UPSAMPLER_INTERPOLATION_ALIGNMENT | PASS | nearest-2x interpolation input to upsampler conv alignment | python3 tools/audit_official_tadsr_unet_upblock0_upsampler.py && python3 tools/export_tadsr_unet_upblock0_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_UPSAMPLER_CONV_ALIGNMENT | PASS | effective upsampler conv output alignment | python3 tools/audit_official_tadsr_unet_upblock0_upsampler.py && python3 tools/export_tadsr_unet_upblock0_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_UPSAMPLER_ALIGNMENT | PASS | isolated up_blocks.0.upsamplers.0 alignment | python3 tools/audit_official_tadsr_unet_upblock0_upsampler.py && python3 tools/export_tadsr_unet_upblock0_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_UPSAMPLER_SYNTHETIC_ALIGNMENT | PASS | synthetic hidden -> up_blocks.0.upsamplers.0 alignment | python3 tools/audit_official_tadsr_unet_upblock0_upsampler.py && python3 tools/export_tadsr_unet_upblock0_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_RESNETS012_UPSAMPLER_ALIGNMENT | PASS | UNet entry -> all down_blocks -> full local mid_block -> up_blocks.0.resnets.0/1/2 -> upsamplers.0 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock0_upsampler.py && python3 tools/export_tadsr_unet_upblock0_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_OUTPUT_HIDDEN_ALIGNMENT | PASS | full local up_blocks.0 hidden-state output alignment | python3 tools/audit_official_tadsr_unet_upblock0_upsampler.py && python3 tools/export_tadsr_unet_upblock0_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_OUTPUT_STATES_ALIGNMENT | PASS | up_blocks.0 output-state contract alignment; official UpBlock2D returns hidden states only | python3 tools/audit_official_tadsr_unet_upblock0_upsampler.py && python3 tools/export_tadsr_unet_upblock0_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK0_BRIDGE_ALIGNMENT | PARTIAL | up_blocks.0 path is aligned through resnets.0 -> resnets.1 -> resnets.2 -> upsamplers.0 | continue with up_blocks.1.resnets.0 |
| TADSR_UNET_UPBLOCK0_ALIGNMENT | PARTIAL | Full local up_blocks.0 hidden output is aligned; later up_blocks remain unopened | continue with up_blocks.1.resnets.0 |
| TADSR_UNET_UPBLOCK1_AUDIT | PASS | official up_blocks.1 topology audit through resnets.0 only | python3 tools/audit_official_tadsr_unet_upblock1_resnet0.py && python3 tools/export_tadsr_unet_upblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET0_AUDIT | PASS | official up_blocks.1.resnets.0 audit | python3 tools/audit_official_tadsr_unet_upblock1_resnet0.py && python3 tools/export_tadsr_unet_upblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET0_RESIDUAL_CONTRACT_AUDIT | PASS | official residual pop/concat contract audit after local up_blocks.0 | python3 tools/audit_official_tadsr_unet_upblock1_resnet0.py && python3 tools/export_tadsr_unet_upblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET0_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.1.resnets.0 exported | python3 tools/audit_official_tadsr_unet_upblock1_resnet0.py && python3 tools/export_tadsr_unet_upblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET0_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.1.resnets.0 exported locally | python3 tools/audit_official_tadsr_unet_upblock1_resnet0.py && python3 tools/export_tadsr_unet_upblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET0_RESIDUAL_CONCAT_ALIGNMENT | PASS | up_blocks.1.resnets.0 hidden/residual concat alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet0.py && python3 tools/export_tadsr_unet_upblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET0_NORM1_ALIGNMENT | PASS | up_blocks.1.resnets.0 norm1 alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet0.py && python3 tools/export_tadsr_unet_upblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET0_CONV1_ALIGNMENT | PASS | up_blocks.1.resnets.0 conv1 alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet0.py && python3 tools/export_tadsr_unet_upblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET0_TEMB_PROJ_ALIGNMENT | PASS | up_blocks.1.resnets.0 time_emb_proj alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet0.py && python3 tools/export_tadsr_unet_upblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET0_CONV2_ALIGNMENT | PASS | up_blocks.1.resnets.0 conv2 alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet0.py && python3 tools/export_tadsr_unet_upblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET0_ALIGNMENT | PASS | isolated up_blocks.1.resnets.0 after official residual concat alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet0.py && python3 tools/export_tadsr_unet_upblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET0_SYNTHETIC_ALIGNMENT | PASS | synthetic hidden/residual -> up_blocks.1.resnets.0 alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet0.py && python3 tools/export_tadsr_unet_upblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET0_BRIDGE_ALIGNMENT | PASS | UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet0.py && python3 tools/export_tadsr_unet_upblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET0_SHORTCUT_ALIGNMENT | PASS | up_blocks.1.resnets.0 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent | continue only if shortcut remains PASS or NOT_APPLICABLE |
| TADSR_UNET_UPBLOCK1_PRE_ATTENTION0_ALIGNMENT | PARTIAL | up_blocks.1 path is aligned through resnets.0 only | continue with up_blocks.1.attentions.0 |
| TADSR_UNET_UPBLOCK1_ATTENTION0_AUDIT | PASS | official up_blocks.1.attentions.0 topology/LoRA audit | python3 tools/audit_official_tadsr_unet_upblock1_attention0.py && python3 tools/export_tadsr_unet_upblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION0_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.1.attentions.0 exported | python3 tools/audit_official_tadsr_unet_upblock1_attention0.py && python3 tools/export_tadsr_unet_upblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION0_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.1.attentions.0 exported locally | python3 tools/audit_official_tadsr_unet_upblock1_attention0.py && python3 tools/export_tadsr_unet_upblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION0_NORM_ALIGNMENT | PASS | up_blocks.1.attentions.0 group norm alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention0.py && python3 tools/export_tadsr_unet_upblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION0_SEQUENCE_ALIGNMENT | PASS | up_blocks.1.attentions.0 NCHW to sequence reshape alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention0.py && python3 tools/export_tadsr_unet_upblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION0_PROJ_IN_ALIGNMENT | PASS | up_blocks.1.attentions.0 proj_in alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention0.py && python3 tools/export_tadsr_unet_upblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION0_ATTN1_QKV_ALIGNMENT | PASS | up_blocks.1.attentions.0 self-attention q/k/v alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention0.py && python3 tools/export_tadsr_unet_upblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION0_ATTN1_ALIGNMENT | PASS | up_blocks.1.attentions.0 self-attention output alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention0.py && python3 tools/export_tadsr_unet_upblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION0_AFTER_ATTN1_ALIGNMENT | PASS | up_blocks.1.attentions.0 residual after attn1 alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention0.py && python3 tools/export_tadsr_unet_upblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION0_ATTN2_QKV_ALIGNMENT | PASS | up_blocks.1.attentions.0 cross-attention q/k/v alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention0.py && python3 tools/export_tadsr_unet_upblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION0_ATTN2_ALIGNMENT | PASS | up_blocks.1.attentions.0 cross-attention output alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention0.py && python3 tools/export_tadsr_unet_upblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION0_AFTER_ATTN2_ALIGNMENT | PASS | up_blocks.1.attentions.0 residual after attn2 alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention0.py && python3 tools/export_tadsr_unet_upblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION0_FF_ALIGNMENT | PASS | up_blocks.1.attentions.0 feed-forward output alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention0.py && python3 tools/export_tadsr_unet_upblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION0_TRANSFORMER0_ALIGNMENT | PASS | up_blocks.1.attentions.0 transformer block output alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention0.py && python3 tools/export_tadsr_unet_upblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION0_ALIGNMENT | PASS | isolated up_blocks.1.attentions.0 output alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention0.py && python3 tools/export_tadsr_unet_upblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_RESNET0_ATTENTION0_ALIGNMENT | PASS | UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention0.py && python3 tools/export_tadsr_unet_upblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION0_BRIDGE_ALIGNMENT | PASS | up_blocks.1 path is aligned through resnets.0 -> attentions.0 only | continue with up_blocks.1.resnets.1 |
| TADSR_UNET_UPBLOCK1_PRE_RESNET1_ALIGNMENT | PASS | input to next unopened up_blocks.1.resnets.1 is now aligned | continue with up_blocks.1.resnets.1 |
| TADSR_UNET_UPBLOCK1_RESNET1_AUDIT | PASS | official up_blocks.1.resnets.1 audit | python3 tools/audit_official_tadsr_unet_upblock1_resnet1.py && python3 tools/export_tadsr_unet_upblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET1_RESIDUAL_CONTRACT_AUDIT | PASS | official residual pop/concat contract audit after up_blocks.1.attentions.0 | python3 tools/audit_official_tadsr_unet_upblock1_resnet1.py && python3 tools/export_tadsr_unet_upblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET1_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.1.resnets.1 exported | python3 tools/audit_official_tadsr_unet_upblock1_resnet1.py && python3 tools/export_tadsr_unet_upblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET1_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.1.resnets.1 exported locally | python3 tools/audit_official_tadsr_unet_upblock1_resnet1.py && python3 tools/export_tadsr_unet_upblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET1_RESIDUAL_CONCAT_ALIGNMENT | PASS | up_blocks.1.resnets.1 hidden/residual concat alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet1.py && python3 tools/export_tadsr_unet_upblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET1_NORM1_ALIGNMENT | PASS | up_blocks.1.resnets.1 norm1 alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet1.py && python3 tools/export_tadsr_unet_upblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET1_CONV1_ALIGNMENT | PASS | up_blocks.1.resnets.1 conv1 alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet1.py && python3 tools/export_tadsr_unet_upblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET1_TEMB_PROJ_ALIGNMENT | PASS | up_blocks.1.resnets.1 time_emb_proj alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet1.py && python3 tools/export_tadsr_unet_upblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET1_CONV2_ALIGNMENT | PASS | up_blocks.1.resnets.1 conv2 alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet1.py && python3 tools/export_tadsr_unet_upblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET1_ALIGNMENT | PASS | isolated up_blocks.1.resnets.1 after official residual concat alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet1.py && python3 tools/export_tadsr_unet_upblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET1_SYNTHETIC_ALIGNMENT | PASS | synthetic hidden/residual -> up_blocks.1.resnets.1 alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet1.py && python3 tools/export_tadsr_unet_upblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_RESNET0_ATTENTION0_RESNET1_ALIGNMENT | PASS | UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 -> up_blocks.1.resnets.1 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet1.py && python3 tools/export_tadsr_unet_upblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET1_SHORTCUT_ALIGNMENT | PASS | up_blocks.1.resnets.1 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent | continue only if shortcut remains PASS or NOT_APPLICABLE |
| TADSR_UNET_UPBLOCK1_RESNET1_BRIDGE_ALIGNMENT | PASS | up_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 only | continue with up_blocks.1.attentions.1 |
| TADSR_UNET_UPBLOCK1_PRE_ATTENTION1_ALIGNMENT | PASS | input to next unopened up_blocks.1.attentions.1 is now aligned | continue with up_blocks.1.attentions.1 |
| TADSR_UNET_UPBLOCK1_ATTENTION1_AUDIT | PASS | official up_blocks.1.attentions.1 topology/LoRA audit | python3 tools/audit_official_tadsr_unet_upblock1_attention1.py && python3 tools/export_tadsr_unet_upblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION1_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.1.attentions.1 exported | python3 tools/audit_official_tadsr_unet_upblock1_attention1.py && python3 tools/export_tadsr_unet_upblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION1_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.1.attentions.1 exported locally | python3 tools/audit_official_tadsr_unet_upblock1_attention1.py && python3 tools/export_tadsr_unet_upblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION1_NORM_ALIGNMENT | PASS | up_blocks.1.attentions.1 group norm alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention1.py && python3 tools/export_tadsr_unet_upblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION1_SEQUENCE_ALIGNMENT | PASS | up_blocks.1.attentions.1 NCHW to sequence reshape alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention1.py && python3 tools/export_tadsr_unet_upblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION1_PROJ_IN_ALIGNMENT | PASS | up_blocks.1.attentions.1 proj_in alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention1.py && python3 tools/export_tadsr_unet_upblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION1_ATTN1_QKV_ALIGNMENT | PASS | up_blocks.1.attentions.1 self-attention q/k/v alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention1.py && python3 tools/export_tadsr_unet_upblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION1_ATTN1_ALIGNMENT | PASS | up_blocks.1.attentions.1 self-attention output alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention1.py && python3 tools/export_tadsr_unet_upblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION1_AFTER_ATTN1_ALIGNMENT | PASS | up_blocks.1.attentions.1 residual after attn1 alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention1.py && python3 tools/export_tadsr_unet_upblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION1_ATTN2_QKV_ALIGNMENT | PASS | up_blocks.1.attentions.1 cross-attention q/k/v alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention1.py && python3 tools/export_tadsr_unet_upblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION1_ATTN2_ALIGNMENT | PASS | up_blocks.1.attentions.1 cross-attention output alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention1.py && python3 tools/export_tadsr_unet_upblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION1_AFTER_ATTN2_ALIGNMENT | PASS | up_blocks.1.attentions.1 residual after attn2 alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention1.py && python3 tools/export_tadsr_unet_upblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION1_FF_ALIGNMENT | PASS | up_blocks.1.attentions.1 feed-forward output alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention1.py && python3 tools/export_tadsr_unet_upblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION1_TRANSFORMER0_ALIGNMENT | PASS | up_blocks.1.attentions.1 transformer block output alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention1.py && python3 tools/export_tadsr_unet_upblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION1_ALIGNMENT | PASS | isolated up_blocks.1.attentions.1 output alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention1.py && python3 tools/export_tadsr_unet_upblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT | PASS | UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 -> up_blocks.1.resnets.1 -> up_blocks.1.attentions.1 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention1.py && python3 tools/export_tadsr_unet_upblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION1_BRIDGE_ALIGNMENT | PASS | up_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 only | continue with up_blocks.1.resnets.2 |
| TADSR_UNET_UPBLOCK1_PRE_RESNET2_ALIGNMENT | PASS | input to next unopened up_blocks.1.resnets.2 is now aligned | continue with up_blocks.1.resnets.2 |
| TADSR_UNET_UPBLOCK1_RESNET2_AUDIT | PASS | official up_blocks.1.resnets.2 audit after up_blocks.1.attentions.1 | python3 tools/audit_official_tadsr_unet_upblock1_resnet2.py && python3 tools/export_tadsr_unet_upblock1_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET2_RESIDUAL_CONTRACT_AUDIT | PASS | official residual pop/concat contract audit after up_blocks.1.attentions.1 | python3 tools/audit_official_tadsr_unet_upblock1_resnet2.py && python3 tools/export_tadsr_unet_upblock1_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET2_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.1.resnets.2 exported | python3 tools/audit_official_tadsr_unet_upblock1_resnet2.py && python3 tools/export_tadsr_unet_upblock1_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET2_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.1.resnets.2 exported locally | python3 tools/audit_official_tadsr_unet_upblock1_resnet2.py && python3 tools/export_tadsr_unet_upblock1_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET2_RESIDUAL_CONCAT_ALIGNMENT | PASS | up_blocks.1.resnets.2 hidden/residual concat alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet2.py && python3 tools/export_tadsr_unet_upblock1_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET2_NORM1_ALIGNMENT | PASS | up_blocks.1.resnets.2 norm1 alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet2.py && python3 tools/export_tadsr_unet_upblock1_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET2_CONV1_ALIGNMENT | PASS | up_blocks.1.resnets.2 conv1 alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet2.py && python3 tools/export_tadsr_unet_upblock1_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET2_TEMB_PROJ_ALIGNMENT | PASS | up_blocks.1.resnets.2 time_emb_proj alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet2.py && python3 tools/export_tadsr_unet_upblock1_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET2_CONV2_ALIGNMENT | PASS | up_blocks.1.resnets.2 conv2 alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet2.py && python3 tools/export_tadsr_unet_upblock1_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET2_ALIGNMENT | PASS | isolated up_blocks.1.resnets.2 after attention1 output and official residual concat alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet2.py && python3 tools/export_tadsr_unet_upblock1_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET2_SYNTHETIC_ALIGNMENT | PASS | synthetic hidden/residual -> up_blocks.1.resnets.2 alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet2.py && python3 tools/export_tadsr_unet_upblock1_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ALIGNMENT | PASS | UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 -> up_blocks.1.resnets.1 -> up_blocks.1.attentions.1 -> up_blocks.1.resnets.2 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock1_resnet2.py && python3 tools/export_tadsr_unet_upblock1_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESNET2_SHORTCUT_ALIGNMENT | PASS | up_blocks.1.resnets.2 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent | continue only if shortcut remains PASS or NOT_APPLICABLE |
| TADSR_UNET_UPBLOCK1_RESNET2_BRIDGE_ALIGNMENT | PASS | up_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 only | continue with up_blocks.1.attentions.2 |
| TADSR_UNET_UPBLOCK1_PRE_ATTENTION2_ALIGNMENT | PASS | input to next unopened up_blocks.1.attentions.2 is now aligned | continue with up_blocks.1.attentions.2 |
| TADSR_UNET_UPBLOCK1_ATTENTION2_AUDIT | PASS | official up_blocks.1.attentions.2 topology/LoRA audit | python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION2_RESIDUAL_CONTRACT_AUDIT | PASS | official no-residual-consumption contract audit for up_blocks.1.attentions.2 | python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION2_LORA_AUDIT | PASS | LoRA/effective static weight audit for up_blocks.1.attentions.2 | python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION2_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.1.attentions.2 exported | python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION2_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.1.attentions.2 exported locally | python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION2_RESIDUAL_CONTRACT_ORACLE | PASS | oracle records attention2 consumes no accumulated residuals | python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION2_NORM_ALIGNMENT | PASS | up_blocks.1.attentions.2 group norm alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION2_SEQUENCE_ALIGNMENT | PASS | up_blocks.1.attentions.2 NCHW to sequence reshape alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION2_PROJ_IN_ALIGNMENT | PASS | up_blocks.1.attentions.2 proj_in alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION2_ATTN1_QKV_ALIGNMENT | PASS | up_blocks.1.attentions.2 self-attention q/k/v alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION2_ATTN1_ALIGNMENT | PASS | up_blocks.1.attentions.2 self-attention output alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION2_AFTER_ATTN1_ALIGNMENT | PASS | up_blocks.1.attentions.2 residual after attn1 alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION2_ATTN2_QKV_ALIGNMENT | PASS | up_blocks.1.attentions.2 cross-attention q/k/v alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION2_ATTN2_ALIGNMENT | PASS | up_blocks.1.attentions.2 cross-attention output alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION2_AFTER_ATTN2_ALIGNMENT | PASS | up_blocks.1.attentions.2 residual after attn2 alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION2_FF_ALIGNMENT | PASS | up_blocks.1.attentions.2 feed-forward output alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION2_TRANSFORMER0_ALIGNMENT | PASS | up_blocks.1.attentions.2 transformer block output alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION2_ALIGNMENT | PASS | isolated up_blocks.1.attentions.2 output alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ATTENTION2_ALIGNMENT | BLOCKED | UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 -> up_blocks.1.resnets.1 -> up_blocks.1.attentions.1 -> up_blocks.1.resnets.2 -> up_blocks.1.attentions.2 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_ATTENTION2_BRIDGE_ALIGNMENT | PARTIAL | up_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 only | continue with up_blocks.1.upsamplers.0 |
| TADSR_UNET_UPBLOCK1_PRE_UPSAMPLER_ALIGNMENT | PARTIAL | input to next unopened up_blocks.1.upsamplers.0 is now aligned | continue with up_blocks.1.upsamplers.0 |
| TADSR_UNET_UPBLOCK1_UPSAMPLER_AUDIT | PASS | official up_blocks.1.upsamplers.0 topology/operation audit | python3 tools/audit_official_tadsr_unet_upblock1_upsampler.py && python3 tools/export_tadsr_unet_upblock1_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_UPSAMPLER_LORA_AUDIT | PASS | LoRA/effective static weight audit for up_blocks.1.upsamplers.0 | python3 tools/audit_official_tadsr_unet_upblock1_upsampler.py && python3 tools/export_tadsr_unet_upblock1_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_UPSAMPLER_RESIDUAL_CONTRACT_AUDIT | PASS | official upsampler consumes no accumulated residuals | python3 tools/audit_official_tadsr_unet_upblock1_upsampler.py && python3 tools/export_tadsr_unet_upblock1_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_UPSAMPLER_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.1.upsamplers.0 exported | python3 tools/audit_official_tadsr_unet_upblock1_upsampler.py && python3 tools/export_tadsr_unet_upblock1_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_UPSAMPLER_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.1.upsamplers.0 exported locally | python3 tools/audit_official_tadsr_unet_upblock1_upsampler.py && python3 tools/export_tadsr_unet_upblock1_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_UPSAMPLER_RESIDUAL_CONTRACT_ORACLE | PASS | oracle records upsampler consumes no accumulated residuals | python3 tools/audit_official_tadsr_unet_upblock1_upsampler.py && python3 tools/export_tadsr_unet_upblock1_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_UPSAMPLER_INTERPOLATION_ALIGNMENT | PASS | up_blocks.1.upsamplers.0 nearest interpolation alignment | python3 tools/audit_official_tadsr_unet_upblock1_upsampler.py && python3 tools/export_tadsr_unet_upblock1_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_UPSAMPLER_CONV_ALIGNMENT | PASS | up_blocks.1.upsamplers.0 exported effective conv alignment | python3 tools/audit_official_tadsr_unet_upblock1_upsampler.py && python3 tools/export_tadsr_unet_upblock1_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_UPSAMPLER_ALIGNMENT | PASS | isolated up_blocks.1.upsamplers.0 output alignment | python3 tools/audit_official_tadsr_unet_upblock1_upsampler.py && python3 tools/export_tadsr_unet_upblock1_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_UPSAMPLER_SYNTHETIC_ALIGNMENT | PASS | synthetic hidden_states -> up_blocks.1.upsamplers.0 alignment | python3 tools/audit_official_tadsr_unet_upblock1_upsampler.py && python3 tools/export_tadsr_unet_upblock1_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_THROUGH_UPSAMPLER_ALIGNMENT | PASS | UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1 through upsamplers.0 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock1_upsampler.py && python3 tools/export_tadsr_unet_upblock1_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_UPSAMPLER_BRIDGE_ALIGNMENT | PARTIAL | up_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 -> upsamplers.0 only; full aggregate is intentionally deferred | complete full local up_blocks.1 aggregate verification in the next stage |
| TADSR_UNET_UPBLOCK1_LOCAL_FORWARD_AUDIT | PASS | official local up_blocks.1 forward equals manual resnet/attention/upsampler chain | python3 tools/audit_official_tadsr_unet_upblock1_local.py && python3 tools/export_tadsr_unet_upblock1_local_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_RESIDUAL_CONTRACT_AUDIT | PASS | official local up_blocks.1 residual tuple contract audited | python3 tools/audit_official_tadsr_unet_upblock1_local.py && python3 tools/export_tadsr_unet_upblock1_local_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_LOCAL_ORACLE_TENSORS | PASS | PyTorch oracle tensors for full local up_blocks.1 aggregate exported | python3 tools/audit_official_tadsr_unet_upblock1_local.py && python3 tools/export_tadsr_unet_upblock1_local_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_OUTPUT_HIDDEN_ALIGNMENT | PASS | Jittor full local up_blocks.1 hidden output alignment | python3 tools/audit_official_tadsr_unet_upblock1_local.py && python3 tools/export_tadsr_unet_upblock1_local_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK1_OUTPUT_STATES_ALIGNMENT | NOT_APPLICABLE | official local up_blocks.1 output_states are not returned in this config, or aligned if present | continue only if PASS or NOT_APPLICABLE |
| TADSR_UNET_UPBLOCK1_ALIGNMENT | PARTIAL | full local up_blocks.1 aggregate is aligned through output hidden; execution still stops before up_blocks.2 | continue with up_blocks.2.resnets.0 only after this remains PASS |
| TADSR_UNET_UPBLOCK2_AUDIT | PASS | official up_blocks.2 topology audited only through resnets.0 | python3 tools/audit_official_tadsr_unet_upblock2_resnet0.py && python3 tools/export_tadsr_unet_upblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET0_AUDIT | PASS | official up_blocks.2.resnets.0 config audited | python3 tools/audit_official_tadsr_unet_upblock2_resnet0.py && python3 tools/export_tadsr_unet_upblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET0_RESIDUAL_CONTRACT_AUDIT | PASS | official residual consumption for up_blocks.2.resnets.0 audited | python3 tools/audit_official_tadsr_unet_upblock2_resnet0.py && python3 tools/export_tadsr_unet_upblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET0_LORA_AUDIT | PASS | LoRA/effective static weight audit for up_blocks.2.resnets.0 | python3 tools/audit_official_tadsr_unet_upblock2_resnet0.py && python3 tools/export_tadsr_unet_upblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET0_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.2.resnets.0 exported | python3 tools/audit_official_tadsr_unet_upblock2_resnet0.py && python3 tools/export_tadsr_unet_upblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET0_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.2.resnets.0 exported locally | python3 tools/audit_official_tadsr_unet_upblock2_resnet0.py && python3 tools/export_tadsr_unet_upblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET0_RESIDUAL_CONTRACT_ORACLE | PASS | oracle records the exact residual consumed by up_blocks.2.resnets.0 | python3 tools/audit_official_tadsr_unet_upblock2_resnet0.py && python3 tools/export_tadsr_unet_upblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET0_RESIDUAL_CONCAT_ALIGNMENT | PASS | up_blocks.2.resnets.0 residual concat alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet0.py && python3 tools/export_tadsr_unet_upblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET0_NORM1_ALIGNMENT | PASS | up_blocks.2.resnets.0 norm1 alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet0.py && python3 tools/export_tadsr_unet_upblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET0_CONV1_ALIGNMENT | PASS | up_blocks.2.resnets.0 conv1 alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet0.py && python3 tools/export_tadsr_unet_upblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET0_TEMB_PROJ_ALIGNMENT | PASS | up_blocks.2.resnets.0 time embedding projection alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet0.py && python3 tools/export_tadsr_unet_upblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET0_CONV2_ALIGNMENT | PASS | up_blocks.2.resnets.0 conv2 alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet0.py && python3 tools/export_tadsr_unet_upblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET0_SHORTCUT_ALIGNMENT | PASS | up_blocks.2.resnets.0 shortcut alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet0.py && python3 tools/export_tadsr_unet_upblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET0_ALIGNMENT | PASS | isolated entry hidden/residual -> up_blocks.2.resnets.0 output alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet0.py && python3 tools/export_tadsr_unet_upblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET0_SYNTHETIC_ALIGNMENT | PASS | synthetic hidden/residual -> up_blocks.2.resnets.0 alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet0.py && python3 tools/export_tadsr_unet_upblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_RESNET0_ALIGNMENT | PASS | UNet entry -> down_blocks -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2.resnets.0 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet0.py && python3 tools/export_tadsr_unet_upblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET0_BRIDGE_ALIGNMENT | PARTIAL | up_blocks.2 is aligned only through resnets.0 and stops before attentions.0 | continue with up_blocks.2.attentions.0; keep full up_blocks.2/full UNet/full inference incomplete |
| TADSR_UNET_UPBLOCK2_PRE_ATTENTION0_ALIGNMENT | PARTIAL | execution reaches the input boundary before up_blocks.2.attentions.0 without entering attention0 | next target is up_blocks.2.attentions.0 |
| TADSR_UNET_UPBLOCK2_ATTENTION0_AUDIT | PASS | official up_blocks.2.attentions.0 topology/LoRA audit | python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION0_RESIDUAL_CONTRACT_AUDIT | PASS | official no-residual-consumption contract audit for up_blocks.2.attentions.0 | python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION0_LORA_AUDIT | PASS | LoRA/effective static weight audit for up_blocks.2.attentions.0 | python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION0_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.2.attentions.0 exported | python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION0_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.2.attentions.0 exported locally | python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION0_RESIDUAL_CONTRACT_ORACLE | PASS | oracle records attention0 consumes no accumulated residuals | python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION0_NORM_ALIGNMENT | PASS | up_blocks.2.attentions.0 group norm alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION0_SEQUENCE_ALIGNMENT | PASS | up_blocks.2.attentions.0 NCHW to sequence reshape alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION0_PROJ_IN_ALIGNMENT | PASS | up_blocks.2.attentions.0 proj_in alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION0_ATTN1_QKV_ALIGNMENT | PASS | up_blocks.2.attentions.0 self-attention q/k/v alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION0_ATTN1_ALIGNMENT | PASS | up_blocks.2.attentions.0 self-attention output alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION0_AFTER_ATTN1_ALIGNMENT | PASS | up_blocks.2.attentions.0 residual after attn1 alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION0_ATTN2_QKV_ALIGNMENT | PASS | up_blocks.2.attentions.0 cross-attention q/k/v alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION0_ATTN2_ALIGNMENT | PASS | up_blocks.2.attentions.0 cross-attention output alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION0_AFTER_ATTN2_ALIGNMENT | PASS | up_blocks.2.attentions.0 residual after attn2 alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION0_FF_ALIGNMENT | PASS | up_blocks.2.attentions.0 feed-forward output alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION0_TRANSFORMER0_ALIGNMENT | PASS | up_blocks.2.attentions.0 transformer block output alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION0_ALIGNMENT | PASS | isolated up_blocks.2.attentions.0 output alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_RESNET0_ATTENTION0_ALIGNMENT | PASS | UNet entry -> down_blocks -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2.resnets.0 -> up_blocks.2.attentions.0 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION0_BRIDGE_ALIGNMENT | PARTIAL | up_blocks.2 is aligned through resnets.0 -> attentions.0 only and stops before resnets.1 | continue with up_blocks.2.resnets.1; keep full up_blocks.2/full UNet/full inference incomplete |
| TADSR_UNET_UPBLOCK2_PRE_RESNET1_ALIGNMENT | PARTIAL | execution reaches the input boundary before up_blocks.2.resnets.1 without entering resnet1 | next target is up_blocks.2.resnets.1 |
| TADSR_UNET_UPBLOCK2_RESNET1_AUDIT | PASS | official up_blocks.2.resnets.1 config audited | python3 tools/audit_official_tadsr_unet_upblock2_resnet1.py && python3 tools/export_tadsr_unet_upblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET1_RESIDUAL_CONTRACT_AUDIT | PASS | official residual consumption for up_blocks.2.resnets.1 audited | python3 tools/audit_official_tadsr_unet_upblock2_resnet1.py && python3 tools/export_tadsr_unet_upblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET1_LORA_AUDIT | PASS | LoRA/effective static weight audit for up_blocks.2.resnets.1 | python3 tools/audit_official_tadsr_unet_upblock2_resnet1.py && python3 tools/export_tadsr_unet_upblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET1_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.2.resnets.1 exported | python3 tools/audit_official_tadsr_unet_upblock2_resnet1.py && python3 tools/export_tadsr_unet_upblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET1_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.2.resnets.1 exported locally | python3 tools/audit_official_tadsr_unet_upblock2_resnet1.py && python3 tools/export_tadsr_unet_upblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET1_RESIDUAL_CONTRACT_ORACLE | PASS | oracle records the exact residual consumed by up_blocks.2.resnets.1 | python3 tools/audit_official_tadsr_unet_upblock2_resnet1.py && python3 tools/export_tadsr_unet_upblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET1_RESIDUAL_CONCAT_ALIGNMENT | PASS | up_blocks.2.resnets.1 residual concat alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet1.py && python3 tools/export_tadsr_unet_upblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET1_NORM1_ALIGNMENT | PASS | up_blocks.2.resnets.1 norm1 alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet1.py && python3 tools/export_tadsr_unet_upblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET1_CONV1_ALIGNMENT | PASS | up_blocks.2.resnets.1 conv1 alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet1.py && python3 tools/export_tadsr_unet_upblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET1_TEMB_PROJ_ALIGNMENT | PASS | up_blocks.2.resnets.1 time embedding projection alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet1.py && python3 tools/export_tadsr_unet_upblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET1_CONV2_ALIGNMENT | PASS | up_blocks.2.resnets.1 conv2 alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet1.py && python3 tools/export_tadsr_unet_upblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET1_SHORTCUT_ALIGNMENT | PASS | up_blocks.2.resnets.1 shortcut alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet1.py && python3 tools/export_tadsr_unet_upblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET1_ALIGNMENT | PASS | isolated attention0 hidden/residual -> up_blocks.2.resnets.1 output alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet1.py && python3 tools/export_tadsr_unet_upblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET1_SYNTHETIC_ALIGNMENT | PASS | synthetic hidden/residual -> up_blocks.2.resnets.1 alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet1.py && python3 tools/export_tadsr_unet_upblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_RESNET0_ATTENTION0_RESNET1_ALIGNMENT | PASS | UNet entry -> down_blocks -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2.resnets.0 -> up_blocks.2.attentions.0 -> up_blocks.2.resnets.1 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet1.py && python3 tools/export_tadsr_unet_upblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET1_BRIDGE_ALIGNMENT | PARTIAL | up_blocks.2 is aligned through resnets.0 -> attentions.0 -> resnets.1 only and stops before attentions.1 | continue with up_blocks.2.attentions.1; keep full up_blocks.2/full UNet/full inference incomplete |
| TADSR_UNET_UPBLOCK2_PRE_ATTENTION1_ALIGNMENT | PARTIAL | execution reaches the input boundary before up_blocks.2.attentions.1 without entering attention1 | next target is up_blocks.2.attentions.1 |
| TADSR_UNET_UPBLOCK2_ATTENTION1_AUDIT | PASS | official up_blocks.2.attentions.1 topology/LoRA audit | python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION1_RESIDUAL_CONTRACT_AUDIT | PASS | official no-residual-consumption contract audit for up_blocks.2.attentions.1 | python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION1_LORA_AUDIT | PASS | LoRA/effective static weight audit for up_blocks.2.attentions.1 | python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION1_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.2.attentions.1 exported | python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION1_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.2.attentions.1 exported locally | python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION1_RESIDUAL_CONTRACT_ORACLE | PASS | oracle records attention1 consumes no accumulated residuals | python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION1_NORM_ALIGNMENT | PASS | up_blocks.2.attentions.1 group norm alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION1_SEQUENCE_ALIGNMENT | PASS | up_blocks.2.attentions.1 NCHW to sequence reshape alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION1_PROJ_IN_ALIGNMENT | PASS | up_blocks.2.attentions.1 proj_in alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION1_ATTN1_QKV_ALIGNMENT | PASS | up_blocks.2.attentions.1 self-attention q/k/v alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION1_ATTN1_ALIGNMENT | PASS | up_blocks.2.attentions.1 self-attention output alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION1_AFTER_ATTN1_ALIGNMENT | PASS | up_blocks.2.attentions.1 residual after attn1 alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION1_ATTN2_QKV_ALIGNMENT | PASS | up_blocks.2.attentions.1 cross-attention q/k/v alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION1_ATTN2_ALIGNMENT | PASS | up_blocks.2.attentions.1 cross-attention output alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION1_AFTER_ATTN2_ALIGNMENT | PASS | up_blocks.2.attentions.1 residual after attn2 alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION1_FF_ALIGNMENT | PASS | up_blocks.2.attentions.1 feed-forward output alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION1_TRANSFORMER0_ALIGNMENT | PASS | up_blocks.2.attentions.1 transformer block output alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION1_ALIGNMENT | PASS | isolated up_blocks.2.attentions.1 output alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT | PASS | UNet entry -> down_blocks -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2.resnets.0 -> up_blocks.2.attentions.0 -> up_blocks.2.resnets.1 -> up_blocks.2.attentions.1 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION1_BRIDGE_ALIGNMENT | PARTIAL | up_blocks.2 is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 only and stops before resnets.2 | continue with up_blocks.2.resnets.2; keep full up_blocks.2/full UNet/full inference incomplete |
| TADSR_UNET_UPBLOCK2_PRE_RESNET2_ALIGNMENT | PARTIAL | execution reaches the input boundary before up_blocks.2.resnets.2 without entering resnet2 | next target is up_blocks.2.resnets.2 |
| TADSR_UNET_UPBLOCK2_RESNET2_AUDIT | PASS | official up_blocks.2.resnets.2 config audited | python3 tools/audit_official_tadsr_unet_upblock2_resnet2.py && python3 tools/export_tadsr_unet_upblock2_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET2_RESIDUAL_CONTRACT_AUDIT | PASS | official residual consumption for up_blocks.2.resnets.2 audited | python3 tools/audit_official_tadsr_unet_upblock2_resnet2.py && python3 tools/export_tadsr_unet_upblock2_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET2_LORA_AUDIT | PASS | LoRA/effective static weight audit for up_blocks.2.resnets.2 | python3 tools/audit_official_tadsr_unet_upblock2_resnet2.py && python3 tools/export_tadsr_unet_upblock2_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET2_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.2.resnets.2 exported | python3 tools/audit_official_tadsr_unet_upblock2_resnet2.py && python3 tools/export_tadsr_unet_upblock2_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET2_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.2.resnets.2 exported locally | python3 tools/audit_official_tadsr_unet_upblock2_resnet2.py && python3 tools/export_tadsr_unet_upblock2_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET2_RESIDUAL_CONTRACT_ORACLE | PASS | oracle records the exact residual consumed by up_blocks.2.resnets.2 | python3 tools/audit_official_tadsr_unet_upblock2_resnet2.py && python3 tools/export_tadsr_unet_upblock2_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET2_RESIDUAL_CONCAT_ALIGNMENT | PASS | up_blocks.2.resnets.2 residual concat alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet2.py && python3 tools/export_tadsr_unet_upblock2_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET2_NORM1_ALIGNMENT | PASS | up_blocks.2.resnets.2 norm1 alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet2.py && python3 tools/export_tadsr_unet_upblock2_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET2_CONV1_ALIGNMENT | PASS | up_blocks.2.resnets.2 conv1 alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet2.py && python3 tools/export_tadsr_unet_upblock2_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET2_TEMB_PROJ_ALIGNMENT | PASS | up_blocks.2.resnets.2 time embedding projection alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet2.py && python3 tools/export_tadsr_unet_upblock2_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET2_CONV2_ALIGNMENT | PASS | up_blocks.2.resnets.2 conv2 alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet2.py && python3 tools/export_tadsr_unet_upblock2_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET2_SHORTCUT_ALIGNMENT | PASS | up_blocks.2.resnets.2 shortcut alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet2.py && python3 tools/export_tadsr_unet_upblock2_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET2_ALIGNMENT | PASS | isolated attention1 hidden/residual -> up_blocks.2.resnets.2 output alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet2.py && python3 tools/export_tadsr_unet_upblock2_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET2_SYNTHETIC_ALIGNMENT | PASS | synthetic hidden/residual -> up_blocks.2.resnets.2 alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet2.py && python3 tools/export_tadsr_unet_upblock2_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ALIGNMENT | BLOCKED | UNet entry -> down_blocks -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2.resnets.0 -> up_blocks.2.attentions.0 -> up_blocks.2.resnets.1 -> up_blocks.2.attentions.1 -> up_blocks.2.resnets.2 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock2_resnet2.py && python3 tools/export_tadsr_unet_upblock2_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESNET2_BRIDGE_ALIGNMENT | PARTIAL | up_blocks.2 is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 only and stops before attentions.2 | continue with up_blocks.2.attentions.2; keep full up_blocks.2/full UNet/full inference incomplete |
| TADSR_UNET_UPBLOCK2_PRE_ATTENTION2_ALIGNMENT | PARTIAL | execution reaches the input boundary before up_blocks.2.attentions.2 without entering attention2 | next target is up_blocks.2.attentions.2 |
| TADSR_UNET_UPBLOCK2_ATTENTION2_AUDIT | PASS | official up_blocks.2.attentions.2 topology/LoRA audit | python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION2_RESIDUAL_CONTRACT_AUDIT | PASS | official no-residual-consumption contract audit for up_blocks.2.attentions.2 | python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION2_LORA_AUDIT | PASS | LoRA/effective static weight audit for up_blocks.2.attentions.2 | python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION2_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.2.attentions.2 exported | python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION2_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.2.attentions.2 exported locally | python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION2_RESIDUAL_CONTRACT_ORACLE | PASS | oracle records attention2 consumes no accumulated residuals | python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION2_NORM_ALIGNMENT | PASS | up_blocks.2.attentions.2 group norm alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION2_SEQUENCE_ALIGNMENT | PASS | up_blocks.2.attentions.2 NCHW to sequence reshape alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION2_PROJ_IN_ALIGNMENT | PASS | up_blocks.2.attentions.2 proj_in alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION2_ATTN1_QKV_ALIGNMENT | PASS | up_blocks.2.attentions.2 self-attention q/k/v alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION2_ATTN1_ALIGNMENT | PASS | up_blocks.2.attentions.2 self-attention output alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION2_AFTER_ATTN1_ALIGNMENT | PASS | up_blocks.2.attentions.2 residual after attn1 alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION2_ATTN2_QKV_ALIGNMENT | PASS | up_blocks.2.attentions.2 cross-attention q/k/v alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION2_ATTN2_ALIGNMENT | PASS | up_blocks.2.attentions.2 cross-attention output alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION2_AFTER_ATTN2_ALIGNMENT | PASS | up_blocks.2.attentions.2 residual after attn2 alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION2_FF_ALIGNMENT | PASS | up_blocks.2.attentions.2 feed-forward output alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION2_TRANSFORMER0_ALIGNMENT | PASS | up_blocks.2.attentions.2 transformer block output alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION2_ALIGNMENT | PASS | isolated up_blocks.2.attentions.2 output alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ATTENTION2_ALIGNMENT | BLOCKED | UNet entry -> down_blocks -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2.resnets.0 -> up_blocks.2.attentions.0 -> up_blocks.2.resnets.1 -> up_blocks.2.attentions.1 -> up_blocks.2.resnets.2 -> up_blocks.2.attentions.2 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_ATTENTION2_BRIDGE_ALIGNMENT | PARTIAL | up_blocks.2 is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 only and stops before upsamplers.0 | continue with up_blocks.2.upsamplers.0; keep full up_blocks.2/full UNet/full inference incomplete |
| TADSR_UNET_UPBLOCK2_PRE_UPSAMPLER_ALIGNMENT | PARTIAL | execution reaches the input boundary before up_blocks.2.upsamplers.0 without entering upsampler | next target is up_blocks.2.upsamplers.0 |
| TADSR_UNET_UPBLOCK2_UPSAMPLER_AUDIT | PASS | official up_blocks.2.upsamplers.0 topology/operation audit | python3 tools/audit_official_tadsr_unet_upblock2_upsampler.py && python3 tools/export_tadsr_unet_upblock2_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_UPSAMPLER_LORA_AUDIT | PASS | LoRA/effective static weight audit for up_blocks.2.upsamplers.0 | python3 tools/audit_official_tadsr_unet_upblock2_upsampler.py && python3 tools/export_tadsr_unet_upblock2_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_UPSAMPLER_RESIDUAL_CONTRACT_AUDIT | PASS | official upsampler consumes no accumulated residuals | python3 tools/audit_official_tadsr_unet_upblock2_upsampler.py && python3 tools/export_tadsr_unet_upblock2_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_UPSAMPLER_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.2.upsamplers.0 exported | python3 tools/audit_official_tadsr_unet_upblock2_upsampler.py && python3 tools/export_tadsr_unet_upblock2_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_UPSAMPLER_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.2.upsamplers.0 exported locally | python3 tools/audit_official_tadsr_unet_upblock2_upsampler.py && python3 tools/export_tadsr_unet_upblock2_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_UPSAMPLER_RESIDUAL_CONTRACT_ORACLE | PASS | oracle records upsampler consumes no accumulated residuals | python3 tools/audit_official_tadsr_unet_upblock2_upsampler.py && python3 tools/export_tadsr_unet_upblock2_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_UPSAMPLER_INTERPOLATION_ALIGNMENT | PASS | up_blocks.2.upsamplers.0 nearest interpolation alignment | python3 tools/audit_official_tadsr_unet_upblock2_upsampler.py && python3 tools/export_tadsr_unet_upblock2_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_UPSAMPLER_CONV_ALIGNMENT | PASS | up_blocks.2.upsamplers.0 exported effective conv alignment | python3 tools/audit_official_tadsr_unet_upblock2_upsampler.py && python3 tools/export_tadsr_unet_upblock2_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_UPSAMPLER_ALIGNMENT | PASS | isolated up_blocks.2.upsamplers.0 output alignment | python3 tools/audit_official_tadsr_unet_upblock2_upsampler.py && python3 tools/export_tadsr_unet_upblock2_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_UPSAMPLER_SYNTHETIC_ALIGNMENT | PASS | synthetic hidden_states -> up_blocks.2.upsamplers.0 alignment | python3 tools/audit_official_tadsr_unet_upblock2_upsampler.py && python3 tools/export_tadsr_unet_upblock2_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_THROUGH_UPSAMPLER_ALIGNMENT | PASS | UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> up_blocks.2 through upsamplers.0 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock2_upsampler.py && python3 tools/export_tadsr_unet_upblock2_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_UPSAMPLER_BRIDGE_ALIGNMENT | PARTIAL | up_blocks.2 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 -> upsamplers.0 only; full aggregate is checked separately | complete full local up_blocks.2 aggregate verification in the next stage |
| TADSR_UNET_UPBLOCK2_LOCAL_FORWARD_AUDIT | PASS | official full local up_blocks.2 forward output matches manual resnet/attention/upsampler chain | python3 tools/audit_official_tadsr_unet_upblock2_local.py && python3 tools/export_tadsr_unet_upblock2_local_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_RESIDUAL_CONTRACT_AUDIT | PASS | official residual consumption contract for full local up_blocks.2 audited | python3 tools/audit_official_tadsr_unet_upblock2_local.py && python3 tools/export_tadsr_unet_upblock2_local_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_LOCAL_ORACLE_TENSORS | PASS | PyTorch oracle tensors for full local up_blocks.2 aggregate exported | python3 tools/audit_official_tadsr_unet_upblock2_local.py && python3 tools/export_tadsr_unet_upblock2_local_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK2_OUTPUT_HIDDEN_ALIGNMENT | PASS | full local up_blocks.2 hidden-state output alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_upblock2_local_alignment.py |
| TADSR_UNET_UPBLOCK2_OUTPUT_STATES_ALIGNMENT | NOT_APPLICABLE | up_blocks.2 output-state contract alignment; official CrossAttnUpBlock2D returns hidden states only in this path | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_upblock2_local_alignment.py |
| TADSR_UNET_UPBLOCK2_ALIGNMENT | PARTIAL | full local up_blocks.2 aggregate is aligned through output hidden; execution still stops before up_blocks.3 | continue with up_blocks.3.resnets.0 only after this remains PASS |
| TADSR_UNET_UPBLOCK3_AUDIT | PASS | official up_blocks.3 topology audit | python3 tools/audit_official_tadsr_unet_upblock3_resnet0.py && python3 tools/export_tadsr_unet_upblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET0_AUDIT | PASS | official up_blocks.3.resnets.0 module audit | python3 tools/audit_official_tadsr_unet_upblock3_resnet0.py && python3 tools/export_tadsr_unet_upblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET0_RESIDUAL_CONTRACT_AUDIT | PASS | official residual consumption contract for up_blocks.3.resnets.0 | python3 tools/audit_official_tadsr_unet_upblock3_resnet0.py && python3 tools/export_tadsr_unet_upblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET0_LORA_AUDIT | PASS | LoRA/effective static weight audit for up_blocks.3.resnets.0 | python3 tools/audit_official_tadsr_unet_upblock3_resnet0.py && python3 tools/export_tadsr_unet_upblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET0_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.3.resnets.0 exported | python3 tools/audit_official_tadsr_unet_upblock3_resnet0.py && python3 tools/export_tadsr_unet_upblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET0_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.3.resnets.0 exported locally | python3 tools/audit_official_tadsr_unet_upblock3_resnet0.py && python3 tools/export_tadsr_unet_upblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET0_RESIDUAL_CONTRACT_ORACLE | PASS | oracle records up_blocks.3.resnets.0 residual pop/concat contract | python3 tools/audit_official_tadsr_unet_upblock3_resnet0.py && python3 tools/export_tadsr_unet_upblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET0_RESIDUAL_CONCAT_ALIGNMENT | PASS | up_blocks.3.resnets.0 residual concat alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet0.py && python3 tools/export_tadsr_unet_upblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET0_NORM1_ALIGNMENT | PASS | up_blocks.3.resnets.0 norm1 alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet0.py && python3 tools/export_tadsr_unet_upblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET0_CONV1_ALIGNMENT | PASS | up_blocks.3.resnets.0 conv1 alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet0.py && python3 tools/export_tadsr_unet_upblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET0_TEMB_PROJ_ALIGNMENT | PASS | up_blocks.3.resnets.0 time embedding projection alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet0.py && python3 tools/export_tadsr_unet_upblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET0_CONV2_ALIGNMENT | PASS | up_blocks.3.resnets.0 conv2 alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet0.py && python3 tools/export_tadsr_unet_upblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET0_SHORTCUT_ALIGNMENT | PASS | up_blocks.3.resnets.0 shortcut alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet0.py && python3 tools/export_tadsr_unet_upblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET0_ALIGNMENT | PASS | isolated up_blocks.3.resnets.0 output alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet0.py && python3 tools/export_tadsr_unet_upblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET0_SYNTHETIC_ALIGNMENT | PASS | synthetic hidden/residual/temb -> up_blocks.3.resnets.0 alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet0.py && python3 tools/export_tadsr_unet_upblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_UPBLOCK3_RESNET0_ALIGNMENT | PASS | UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet0.py && python3 tools/export_tadsr_unet_upblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET0_BRIDGE_ALIGNMENT | PARTIAL | up_blocks.3 path is aligned through resnets.0 only and deliberately stops before the next up_blocks.3 module | continue with the next official up_blocks.3 module |
| TADSR_UNET_UPBLOCK3_PRE_ATTENTION0_ALIGNMENT | PARTIAL | execution reaches the input boundary before up_blocks.3.attentions.0 without entering attention0 | next target is up_blocks.3.attentions.0 |
| TADSR_UNET_UPBLOCK3_ATTENTION0_AUDIT | PASS | official up_blocks.3.attentions.0 top-level audit | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION0_TRANSFORMER_AUDIT | PASS | official transformer block audit for up_blocks.3.attentions.0 | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION0_LORA_AUDIT | PASS | LoRA/effective static weight audit for up_blocks.3.attentions.0 | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION0_RESIDUAL_CONTRACT_AUDIT | PASS | official residual contract: attention0 consumes no residual | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION0_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.3.attentions.0 exported | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION0_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.3.attentions.0 exported locally | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION0_RESIDUAL_CONTRACT_ORACLE | PASS | oracle records unchanged residual tuple across up_blocks.3.attentions.0 | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION0_NORM_ALIGNMENT | PASS | up_blocks.3.attentions.0 top norm alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION0_PROJ_IN_ALIGNMENT | PASS | up_blocks.3.attentions.0 proj_in alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION0_SEQUENCE_ALIGNMENT | PASS | up_blocks.3.attentions.0 NCHW-to-sequence contract alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION0_ATTN1_QKV_ALIGNMENT | PASS | up_blocks.3.attentions.0 self-attention Q/K/V alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION0_ATTN1_ALIGNMENT | PASS | up_blocks.3.attentions.0 self-attention output alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION0_AFTER_ATTN1_ALIGNMENT | PASS | up_blocks.3.attentions.0 residual after self-attention alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION0_ATTN2_QKV_ALIGNMENT | PASS | up_blocks.3.attentions.0 cross-attention Q/K/V alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION0_ATTN2_ALIGNMENT | PASS | up_blocks.3.attentions.0 cross-attention output alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION0_AFTER_ATTN2_ALIGNMENT | PASS | up_blocks.3.attentions.0 residual after cross-attention alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION0_FF_ALIGNMENT | PASS | up_blocks.3.attentions.0 feed-forward alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION0_TRANSFORMER0_ALIGNMENT | PASS | up_blocks.3.attentions.0 transformer block output alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION0_ALIGNMENT | PASS | isolated up_blocks.3.attentions.0 output alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_UPBLOCK3_RESNET0_ATTENTION0_ALIGNMENT | PASS | UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 -> up_blocks.3.attentions.0 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION0_BRIDGE_ALIGNMENT | PARTIAL | up_blocks.3 path is aligned through attentions.0 only and deliberately stops before the next up_blocks.3 module | continue with the next official up_blocks.3 module |
| TADSR_UNET_UPBLOCK3_PRE_RESNET1_ALIGNMENT | PARTIAL | execution reaches the input boundary before up_blocks.3.resnets.1 without entering resnet1 | next target is up_blocks.3.resnets.1 |
| TADSR_UNET_UPBLOCK3_RESNET1_AUDIT | PASS | official up_blocks.3.resnets.1 module audit | python3 tools/audit_official_tadsr_unet_upblock3_resnet1.py && python3 tools/export_tadsr_unet_upblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET1_RESIDUAL_CONTRACT_AUDIT | PASS | official residual consumption contract for up_blocks.3.resnets.1 | python3 tools/audit_official_tadsr_unet_upblock3_resnet1.py && python3 tools/export_tadsr_unet_upblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET1_LORA_AUDIT | PASS | LoRA/effective static weight audit for up_blocks.3.resnets.1 | python3 tools/audit_official_tadsr_unet_upblock3_resnet1.py && python3 tools/export_tadsr_unet_upblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET1_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.3.resnets.1 exported | python3 tools/audit_official_tadsr_unet_upblock3_resnet1.py && python3 tools/export_tadsr_unet_upblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET1_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.3.resnets.1 exported locally | python3 tools/audit_official_tadsr_unet_upblock3_resnet1.py && python3 tools/export_tadsr_unet_upblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET1_RESIDUAL_CONTRACT_ORACLE | PASS | oracle records up_blocks.3.resnets.1 residual pop/concat contract | python3 tools/audit_official_tadsr_unet_upblock3_resnet1.py && python3 tools/export_tadsr_unet_upblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET1_RESIDUAL_CONCAT_ALIGNMENT | PASS | up_blocks.3.resnets.1 residual concat alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet1.py && python3 tools/export_tadsr_unet_upblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET1_NORM1_ALIGNMENT | PASS | up_blocks.3.resnets.1 norm1 alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet1.py && python3 tools/export_tadsr_unet_upblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET1_CONV1_ALIGNMENT | PASS | up_blocks.3.resnets.1 conv1 alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet1.py && python3 tools/export_tadsr_unet_upblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET1_TEMB_PROJ_ALIGNMENT | PASS | up_blocks.3.resnets.1 time embedding projection alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet1.py && python3 tools/export_tadsr_unet_upblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET1_CONV2_ALIGNMENT | PASS | up_blocks.3.resnets.1 conv2 alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet1.py && python3 tools/export_tadsr_unet_upblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET1_SHORTCUT_ALIGNMENT | PASS | up_blocks.3.resnets.1 shortcut alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet1.py && python3 tools/export_tadsr_unet_upblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET1_ALIGNMENT | PASS | isolated up_blocks.3.resnets.1 output alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet1.py && python3 tools/export_tadsr_unet_upblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET1_SYNTHETIC_ALIGNMENT | PASS | synthetic hidden/residual/temb -> up_blocks.3.resnets.1 alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet1.py && python3 tools/export_tadsr_unet_upblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_UPBLOCK3_RESNET0_ATTENTION0_RESNET1_ALIGNMENT | PASS | UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 -> up_blocks.3.attentions.0 -> up_blocks.3.resnets.1 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet1.py && python3 tools/export_tadsr_unet_upblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET1_BRIDGE_ALIGNMENT | PARTIAL | up_blocks.3 path is aligned through resnets.0 -> attentions.0 -> resnets.1 only and deliberately stops before the next up_blocks.3 module | continue with the next official up_blocks.3 module |
| TADSR_UNET_UPBLOCK3_PRE_ATTENTION1_ALIGNMENT | PARTIAL | execution reaches the input boundary before up_blocks.3.attentions.1 without entering attention1 | next target is up_blocks.3.attentions.1 |
| TADSR_UNET_UPBLOCK3_ATTENTION1_AUDIT | PASS | official up_blocks.3.attentions.1 top-level audit | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION1_TRANSFORMER_AUDIT | PASS | official transformer block audit for up_blocks.3.attentions.1 | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION1_LORA_AUDIT | PASS | LoRA/effective static weight audit for up_blocks.3.attentions.1 | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION1_RESIDUAL_CONTRACT_AUDIT | PASS | official residual contract: attention1 consumes no residual | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION1_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.3.attentions.1 exported | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION1_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.3.attentions.1 exported locally | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION1_RESIDUAL_CONTRACT_ORACLE | PASS | oracle records unchanged residual tuple across up_blocks.3.attentions.1 | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION1_NORM_ALIGNMENT | PASS | up_blocks.3.attentions.1 top norm alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION1_PROJ_IN_ALIGNMENT | PASS | up_blocks.3.attentions.1 proj_in alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION1_SEQUENCE_ALIGNMENT | PASS | up_blocks.3.attentions.1 NCHW-to-sequence contract alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION1_ATTN1_QKV_ALIGNMENT | PASS | up_blocks.3.attentions.1 self-attention Q/K/V alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION1_ATTN1_ALIGNMENT | PASS | up_blocks.3.attentions.1 self-attention output alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION1_AFTER_ATTN1_ALIGNMENT | PASS | up_blocks.3.attentions.1 residual after self-attention alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION1_ATTN2_QKV_ALIGNMENT | PASS | up_blocks.3.attentions.1 cross-attention Q/K/V alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION1_ATTN2_ALIGNMENT | PASS | up_blocks.3.attentions.1 cross-attention output alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION1_AFTER_ATTN2_ALIGNMENT | PASS | up_blocks.3.attentions.1 residual after cross-attention alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION1_FF_ALIGNMENT | PASS | up_blocks.3.attentions.1 feed-forward alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION1_TRANSFORMER0_ALIGNMENT | PASS | up_blocks.3.attentions.1 transformer block output alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION1_ALIGNMENT | PASS | isolated up_blocks.3.attentions.1 output alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_UPBLOCK3_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT | BLOCKED | UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 -> up_blocks.3.attentions.0 -> up_blocks.3.resnets.1 -> up_blocks.3.attentions.1 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION1_BRIDGE_ALIGNMENT | PARTIAL | up_blocks.3 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 only and deliberately stops before the next up_blocks.3 module | continue with the next official up_blocks.3 module |
| TADSR_UNET_UPBLOCK3_PRE_RESNET2_ALIGNMENT | PARTIAL | execution reaches the input boundary before up_blocks.3.resnets.2 without entering resnet2 | next target is up_blocks.3.resnets.2 |
| TADSR_UNET_UPBLOCK3_RESNET2_AUDIT | PASS | official up_blocks.3.resnets.2 module audit | python3 tools/audit_official_tadsr_unet_upblock3_resnet2.py && python3 tools/export_tadsr_unet_upblock3_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET2_RESIDUAL_CONTRACT_AUDIT | PASS | official residual consumption contract for up_blocks.3.resnets.2 | python3 tools/audit_official_tadsr_unet_upblock3_resnet2.py && python3 tools/export_tadsr_unet_upblock3_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET2_LORA_AUDIT | PASS | LoRA/effective static weight audit for up_blocks.3.resnets.2 | python3 tools/audit_official_tadsr_unet_upblock3_resnet2.py && python3 tools/export_tadsr_unet_upblock3_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET2_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.3.resnets.2 exported | python3 tools/audit_official_tadsr_unet_upblock3_resnet2.py && python3 tools/export_tadsr_unet_upblock3_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET2_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.3.resnets.2 exported locally | python3 tools/audit_official_tadsr_unet_upblock3_resnet2.py && python3 tools/export_tadsr_unet_upblock3_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET2_RESIDUAL_CONTRACT_ORACLE | PASS | oracle records up_blocks.3.resnets.2 residual pop/concat contract | python3 tools/audit_official_tadsr_unet_upblock3_resnet2.py && python3 tools/export_tadsr_unet_upblock3_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET2_RESIDUAL_CONCAT_ALIGNMENT | PASS | up_blocks.3.resnets.2 residual concat alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet2.py && python3 tools/export_tadsr_unet_upblock3_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET2_NORM1_ALIGNMENT | PASS | up_blocks.3.resnets.2 norm1 alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet2.py && python3 tools/export_tadsr_unet_upblock3_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET2_CONV1_ALIGNMENT | PASS | up_blocks.3.resnets.2 conv1 alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet2.py && python3 tools/export_tadsr_unet_upblock3_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET2_TEMB_PROJ_ALIGNMENT | PASS | up_blocks.3.resnets.2 time embedding projection alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet2.py && python3 tools/export_tadsr_unet_upblock3_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET2_CONV2_ALIGNMENT | PASS | up_blocks.3.resnets.2 conv2 alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet2.py && python3 tools/export_tadsr_unet_upblock3_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET2_SHORTCUT_ALIGNMENT | PASS | up_blocks.3.resnets.2 shortcut alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet2.py && python3 tools/export_tadsr_unet_upblock3_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET2_ALIGNMENT | PASS | isolated up_blocks.3.resnets.2 output alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet2.py && python3 tools/export_tadsr_unet_upblock3_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET2_SYNTHETIC_ALIGNMENT | PASS | synthetic hidden/residual/temb -> up_blocks.3.resnets.2 alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet2.py && python3 tools/export_tadsr_unet_upblock3_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_UPBLOCK3_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ALIGNMENT | BLOCKED | UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 -> up_blocks.3.attentions.0 -> up_blocks.3.resnets.1 -> up_blocks.3.attentions.1 -> up_blocks.3.resnets.2 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock3_resnet2.py && python3 tools/export_tadsr_unet_upblock3_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_RESNET2_BRIDGE_ALIGNMENT | PARTIAL | up_blocks.3 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 only and deliberately stops before the next up_blocks.3 module | continue with the next official up_blocks.3 module |
| TADSR_UNET_UPBLOCK3_PRE_ATTENTION2_ALIGNMENT | PARTIAL | execution reaches the input boundary before up_blocks.3.attentions.2 without entering attention2 | next target is up_blocks.3.attentions.2 |
| TADSR_UNET_UPBLOCK3_ATTENTION2_AUDIT | PASS | official up_blocks.3.attentions.2 module audit | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION2_TRANSFORMER_AUDIT | PASS | official Transformer2DModel/BasicTransformerBlock audit for up_blocks.3.attentions.2 | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION2_LORA_AUDIT | PASS | LoRA/effective static weight audit for up_blocks.3.attentions.2 | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION2_RESIDUAL_CONTRACT_AUDIT | PASS | official residual non-consumption contract for up_blocks.3.attentions.2 | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION2_ORACLE_TENSORS | PASS | PyTorch oracle tensors for up_blocks.3.attentions.2 exported | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION2_EFFECTIVE_WEIGHTS | PASS | effective static weights for up_blocks.3.attentions.2 exported locally | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION2_RESIDUAL_CONTRACT_ORACLE | PASS | oracle records up_blocks.3.attentions.2 residual non-consumption contract | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION2_NORM_ALIGNMENT | PASS | attention2 top-level GroupNorm alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION2_PROJ_IN_ALIGNMENT | PASS | attention2 proj_in alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION2_SEQUENCE_ALIGNMENT | PASS | attention2 NCHW-to-sequence alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION2_ATTN1_QKV_ALIGNMENT | PASS | attention2 transformer0 self-attention QKV alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION2_ATTN1_ALIGNMENT | PASS | attention2 transformer0 self-attention output alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION2_AFTER_ATTN1_ALIGNMENT | PASS | attention2 transformer0 after-attn1 residual alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION2_ATTN2_QKV_ALIGNMENT | PASS | attention2 transformer0 cross-attention QKV alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION2_ATTN2_ALIGNMENT | PASS | attention2 transformer0 cross-attention output alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION2_AFTER_ATTN2_ALIGNMENT | PASS | attention2 transformer0 after-attn2 residual alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION2_FF_ALIGNMENT | PASS | attention2 feed-forward output alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION2_TRANSFORMER0_ALIGNMENT | PASS | attention2 full transformer block alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION2_ALIGNMENT | PASS | isolated up_blocks.3.attentions.2 output alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_UPBLOCK3_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ATTENTION2_ALIGNMENT | BLOCKED | UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 bridge alignment | python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_UPBLOCK3_ATTENTION2_BRIDGE_ALIGNMENT | PARTIAL | up_blocks.3 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 and deliberately stops before output tail/full local aggregate | next target is the full local up_blocks.3 aggregate / output-tail-boundary verification |
| TADSR_UNET_UPBLOCK3_PRE_LOCAL_AGGREGATE_ALIGNMENT | PARTIAL | execution reaches the boundary before output tail / full local up_blocks.3 aggregate without entering full UNet forward | next target is full local up_blocks.3 aggregate verification only |
| TADSR_UNET_UPBLOCK3_LOCAL_FORWARD_AUDIT | PASS | official local up_blocks.3 forward agrees with the audited manual resnet/attention chain | python3 tools/audit_official_tadsr_unet_upblock3_local.py && python3 tools/export_tadsr_unet_upblock3_local_oracle.py && USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_upblock3_local_alignment.py |
| TADSR_UNET_UPBLOCK3_RESIDUAL_CONTRACT_AUDIT | PASS | official up_blocks.3 local residual tuple consumption contract audited | python3 tools/audit_official_tadsr_unet_upblock3_local.py && python3 tools/export_tadsr_unet_upblock3_local_oracle.py && USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_upblock3_local_alignment.py |
| TADSR_UNET_UPBLOCK3_LOCAL_ORACLE_TENSORS | PASS | PyTorch oracle tensors for full local up_blocks.3 aggregate exported | python3 tools/audit_official_tadsr_unet_upblock3_local.py && python3 tools/export_tadsr_unet_upblock3_local_oracle.py && USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_upblock3_local_alignment.py |
| TADSR_UNET_UPBLOCK3_RESIDUAL_CONTRACT_ORACLE | PASS | oracle records up_blocks.3 residual consumption and zero residuals after attention2 | python3 tools/audit_official_tadsr_unet_upblock3_local.py && python3 tools/export_tadsr_unet_upblock3_local_oracle.py && USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_upblock3_local_alignment.py |
| TADSR_UNET_OUTPUT_TAIL_BOUNDARY_AUDIT | PASS | official output-tail boundary after up_blocks.3 audited without executing the tail | python3 tools/audit_official_tadsr_unet_upblock3_local.py && python3 tools/export_tadsr_unet_upblock3_local_oracle.py && USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_upblock3_local_alignment.py |
| TADSR_UNET_UPBLOCK3_OUTPUT_HIDDEN_ALIGNMENT | PASS | Jittor full local up_blocks.3 hidden output matches the PyTorch local oracle | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_upblock3_local_alignment.py |
| TADSR_UNET_UPBLOCK3_OUTPUT_STATES_ALIGNMENT | NOT_APPLICABLE | official up_blocks.3 output_states contract; NOT_APPLICABLE if it returns a hidden tensor only | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_upblock3_local_alignment.py |
| TADSR_UNET_OUTPUT_TAIL_BOUNDARY_ALIGNMENT | PASS | Jittor bridge stops at the audited output-tail boundary without running conv_norm_out/conv_act/conv_out | next target is output tail audit/export/port |
| TADSR_UNET_UPBLOCK3_ALIGNMENT | PARTIAL | up_blocks.3 leaves plus full local aggregate hidden output are aligned and the output-tail boundary is audited | continue with output tail audit/export/port before full UNet forward |
| TADSR_UNET_OUTPUT_TAIL_AUDIT | PASS | official output tail conv_norm_out -> conv_act -> conv_out audit | python3 tools/audit_official_tadsr_unet_output_tail.py && python3 tools/export_tadsr_unet_output_tail_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_OUTPUT_TAIL_TOPOLOGY_AUDIT | PASS | official output tail topology/config audit | python3 tools/audit_official_tadsr_unet_output_tail.py && python3 tools/export_tadsr_unet_output_tail_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_OUTPUT_TAIL_LORA_AUDIT | PASS | output tail conv_out LoRA/effective static weight audit | python3 tools/audit_official_tadsr_unet_output_tail.py && python3 tools/export_tadsr_unet_output_tail_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_OUTPUT_TAIL_LOCAL_EXECUTION_AUDIT | PASS | official local output tail execution without official unet.forward | python3 tools/audit_official_tadsr_unet_output_tail.py && python3 tools/export_tadsr_unet_output_tail_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_OUTPUT_TAIL_ORACLE_TENSORS | PASS | PyTorch oracle tensors for output tail exported | python3 tools/audit_official_tadsr_unet_output_tail.py && python3 tools/export_tadsr_unet_output_tail_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_OUTPUT_TAIL_EFFECTIVE_WEIGHTS | PASS | conv_norm_out raw affine parameters and conv_out LoRA-merged effective weights exported | python3 tools/audit_official_tadsr_unet_output_tail.py && python3 tools/export_tadsr_unet_output_tail_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh |
| TADSR_UNET_OUTPUT_TAIL_NORM_ALIGNMENT | PASS | conv_norm_out GroupNorm alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_output_tail_alignment.py |
| TADSR_UNET_OUTPUT_TAIL_ACT_ALIGNMENT | PASS | conv_act SiLU alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_output_tail_alignment.py |
| TADSR_UNET_OUTPUT_TAIL_CONV_OUT_ALIGNMENT | PASS | conv_out LoRA-merged effective convolution alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_output_tail_alignment.py |
| TADSR_UNET_OUTPUT_TAIL_ALIGNMENT | PASS | isolated output tail alignment from the PyTorch up_blocks.3 output tensor | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_output_tail_alignment.py |
| TADSR_UNET_OUTPUT_TAIL_SYNTHETIC_ALIGNMENT | PASS | synthetic hidden tensor -> output tail alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_output_tail_synthetic_alignment.py |
| TADSR_UNET_ENTRY_TO_OUTPUT_TAIL_ALIGNMENT | PASS | entry -> all down_blocks -> full local mid_block -> full local up_blocks.0/1/2/3 -> output tail bridge alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_entry_downblocks_midblock_upblocks_output_tail_alignment.py |
| TADSR_UNET_MANUAL_BLOCKS_TO_TAIL_ALIGNMENT | PASS | manual block composition through output tail is aligned without official UNet.forward | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_entry_downblocks_midblock_upblocks_output_tail_alignment.py |
| TADSR_UNET_MANUAL_FULL_WRAPPER_AUDIT | PASS | official manual wrapper contract audit without official UNet.forward | python3 tools/audit_official_tadsr_unet_manual_wrapper.py && python3 tools/export_tadsr_unet_manual_wrapper_oracle.py |
| TADSR_UNET_MANUAL_FULL_WRAPPER_CONTRACT_AUDIT | PASS | manual wrapper input/output contract audit | python3 tools/audit_official_tadsr_unet_manual_wrapper.py && python3 tools/export_tadsr_unet_manual_wrapper_oracle.py |
| TADSR_UNET_MANUAL_FULL_CHAIN_TO_TAIL_AUDIT | PASS | official module chain audit through output tail without full forward | python3 tools/audit_official_tadsr_unet_manual_wrapper.py && python3 tools/export_tadsr_unet_manual_wrapper_oracle.py |
| TADSR_UNET_MANUAL_FULL_WRAPPER_ORACLE_TENSORS | PASS | PyTorch oracle tensors for alignment-only manual full-chain wrapper exported | python3 tools/audit_official_tadsr_unet_manual_wrapper.py && python3 tools/export_tadsr_unet_manual_wrapper_oracle.py |
| TADSR_UNET_MANUAL_WRAPPER_ENTRY_ALIGNMENT | PASS | manual wrapper entry path alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_manual_wrapper_alignment.py |
| TADSR_UNET_MANUAL_WRAPPER_DOWNBLOCKS_ALIGNMENT | PASS | manual wrapper down_blocks alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_manual_wrapper_alignment.py |
| TADSR_UNET_MANUAL_WRAPPER_MIDBLOCK_ALIGNMENT | PASS | manual wrapper mid_block alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_manual_wrapper_alignment.py |
| TADSR_UNET_MANUAL_WRAPPER_UPBLOCKS_ALIGNMENT | PASS | manual wrapper up_blocks.0/1/2/3 alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_manual_wrapper_alignment.py |
| TADSR_UNET_MANUAL_WRAPPER_OUTPUT_TAIL_ALIGNMENT | PASS | manual wrapper output-tail alignment | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_manual_wrapper_alignment.py |
| TADSR_UNET_MANUAL_FULL_WRAPPER_ALIGNMENT | PASS | alignment-only manual UNet chain wrapper output matches the PyTorch manual oracle | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_manual_wrapper_alignment.py |
| TADSR_UNET_MANUAL_FULL_CHAIN_ALIGNMENT | PASS | manual full chain from UNet inputs through output tail is aligned; this is not official full forward | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_manual_wrapper_alignment.py |
| TADSR_UNET_OFFICIAL_FULL_FORWARD_AUDIT | PASS | official PyTorch UNet.forward contract audit | python3 tools/audit_official_tadsr_unet_full_forward.py && python3 tools/export_tadsr_unet_full_forward_oracle.py |
| TADSR_UNET_FULL_FORWARD_CONTRACT_AUDIT | PASS | official full forward input/output and return contract audit | python3 tools/audit_official_tadsr_unet_full_forward.py && python3 tools/export_tadsr_unet_full_forward_oracle.py |
| TADSR_UNET_MANUAL_VS_OFFICIAL_FULL_FORWARD_AUDIT | PASS | PyTorch manual wrapper output matches official full forward output | python3 tools/audit_official_tadsr_unet_full_forward.py && python3 tools/export_tadsr_unet_full_forward_oracle.py |
| TADSR_UNET_OFFICIAL_FULL_FORWARD_ORACLE_TENSORS | PASS | official full forward oracle tensors exported | python3 tools/audit_official_tadsr_unet_full_forward.py && python3 tools/export_tadsr_unet_full_forward_oracle.py |
| TADSR_UNET_MANUAL_VS_OFFICIAL_FULL_FORWARD_ALIGNMENT | PASS | manual wrapper oracle matches official full forward oracle | python3 tools/audit_official_tadsr_unet_full_forward.py && python3 tools/export_tadsr_unet_full_forward_oracle.py |
| TADSR_UNET_JITTOR_VS_OFFICIAL_FULL_FORWARD_ALIGNMENT | PASS | Jittor alignment-only full forward output matches official PyTorch UNet.forward oracle | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_full_forward_alignment.py |
| TADSR_UNET_JITTOR_VS_MANUAL_WRAPPER_ALIGNMENT | PASS | Jittor alignment-only full forward output matches PyTorch manual wrapper oracle | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_full_forward_alignment.py |
| TADSR_UNET_FULL_FORWARD_RETURN_CONTRACT_ALIGNMENT | PASS | alignment-only return_dict/tensor contract matches the test contract | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_full_forward_alignment.py |
| TADSR_UNET_UPBLOCKS_ALIGNMENT | NOT_COMPLETE | UNet up_blocks.0/1/2/3 local aggregate path is aligned; output tail is checked separately and full UNet forward remains unopened | continue with full UNet forward-wrapper audit/export after output tail stays PASS |
| TADSR_UNET_CROSS_ATTENTION_ALIGNMENT | NOT_COMPLETE | all local UNet cross-attention modules through up_blocks.3 are aligned; global full-forward integration remains incomplete until full forward wrapper is compared | continue with full-forward migration |
| TADSR_UNET_FULL_FORWARD_ALIGNMENT | PASS | UNet full forward numerical alignment against official PyTorch UNet.forward; not full TADSR inference | continue pipeline boundary planning only after keeping inference guard intact |
| TADSR_UNET_LORA_RUNTIME_INTEGRATION | PARTIAL | LoRA-bearing modules are exported as static effective weights; generic runtime LoRA is not implemented | implement runtime LoRA only if a later audit proves dynamic adapter behavior is required |
| TADSR_LORA_POLICY_AUDIT | PASS | official TADSR LoRA/PEFT source usage, active module inventory, and inference-time adapter policy audited | python3 tools/audit_official_tadsr_lora_policy.py |
| TADSR_STATIC_EFFECTIVE_LORA_POLICY_AUDIT | PASS | static effective weights are the selected alignment policy when official inference uses fixed active adapters | docs/lora_policy_decision.md |
| TADSR_RUNTIME_DYNAMIC_LORA_REQUIREMENT_AUDIT | PASS | generic runtime LoRA is required only if official inference dynamically changes adapters or scale | python3 tools/audit_official_tadsr_lora_policy.py |
| TADSR_LORA_MODULE_INVENTORY_AUDIT | PASS | active LoRA A/B module inventory from the official TADSR checkpoint and source policy audit | python3 tools/audit_official_tadsr_lora_policy.py |
| TIME_VAE_LORA_EFFECTIVE_WEIGHTS_AUDIT | PASS | official active TimeVAE LoRA pair inventory is exported as static effective weights | python3 tools/export_tadsr_timevae_lora_effective_weights.py |
| TIME_VAE_LORA_EFFECTIVE_WEIGHTS_EXPORT | PASS | converted_timevae_lora_effective_weights.npz exists as a local git-ignored static effective-weight artifact | python3 tools/export_tadsr_timevae_lora_effective_weights.py |
| TIME_VAE_LORA_EFFECTIVE_WEIGHT_MANUAL_VERIFY | PASS | official active LoRA module forward is compared with manual effective-weight forward for each TimeVAE pair | python3 tools/export_tadsr_timevae_lora_effective_weights.py |
| TIME_VAE_LORA_EFFECTIVE_ARTIFACT_COVERAGE | PASS | all active TimeVAE LoRA pairs have static effective-weight artifact metadata and verification evidence | python3 tools/audit_jittor_tadsr_effective_lora_coverage.py |
| TIME_VAE_ACTIVE_LORA_MODULE_COVERAGE | PASS | TimeVAE active LoRA module coverage in the project-wide static effective-weight policy audit | python3 tools/audit_jittor_tadsr_effective_lora_coverage.py |
| TADSR_STATIC_EFFECTIVE_LORA_COVERAGE_AUDIT | PASS | Jittor static effective-weight artifacts are checked against official active LoRA modules | python3 tools/audit_jittor_tadsr_effective_lora_coverage.py |
| TADSR_ACTIVE_LORA_MODULE_COVERAGE | PASS | active official LoRA modules covered by exported effective weights or explicitly reported as partial/missing | python3 tools/audit_jittor_tadsr_effective_lora_coverage.py |
| TADSR_EFFECTIVE_WEIGHT_ARTIFACT_COVERAGE | PASS | effective-weight artifacts exist and large NPZ files remain local artifacts rather than committed payloads | python3 tools/audit_jittor_tadsr_effective_lora_coverage.py |
| TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION | NOT_IMPLEMENTED_BY_DESIGN | generic runtime LoRA adapter loading/switching/scale is deferred and not claimed complete | dedicated runtime LoRA stage only if required by a future official behavior audit |
| TADSR_LORA_POLICY_CONTRACT_TEST | PASS | metadata-only contract test for LoRA policy/coverage evidence and full-inference guard | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_lora_policy_coverage.py |
| TIME_VAE_LORA_EFFECTIVE_WEIGHT_COVERAGE_TEST | PASS | metadata-only contract test for TimeVAE LoRA effective artifact coverage and full-inference guard | USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_lora_effective_weight_coverage.py |
| TIME_VAE_ENCODER_TO_QUANT_ALIGNMENT | PASS | TimeAware VAE encoder-side path to quant_conv aligned; deterministic decoder stack through tail aligned separately | continue full UNet audit/export/port |
| TIME_VAE_FULL_ALIGNMENT | NOT_COMPLETE | TimeVAE full-boundary alignment is PASS, but full TimeVAE alignment remains NOT_COMPLETE because audit metadata says the boundary is not sufficient for full TADSR pipeline: VAEHook tiling, internal stochastic sampling policy, runtime LoRA, and full inference remain unopened. | audit VAEHook tiled path / runtime LoRA before full inference |
| JITTOR_FULL_INFERENCE | NOT_COMPLETE | full VAE/UNet/LoRA inference is intentionally NotImplemented | continue block-level migration |
| JITTOR_MIGRATION_REPORT | PASS | migration evidence report exists | python3 scripts/make_jittor_migration_report.py |
| JITTOR_FULL_PORT | PARTIAL | skeleton/check CLI only; full VAE/UNet not complete | continue module port |
| FINAL_README | PASS | README has Jittor migration evidence sections | update README |
| TADSR_FINAL_EVIDENCE_MANIFEST | PASS | final evidence manifest indexes committed audit/report files | python3 scripts/collect_final_evidence_manifest.py |
| TADSR_PRODUCTION_CLI_DESIGN_AUDIT | PASS | production CLI design memo exists and keeps full inference guarded | update docs/production_cli_design_audit.md |
| TADSR_FINAL_SUBMISSION_CHECKLIST | PASS | final submission checklist exists with status, commands, evidence and limitations | update docs/final_submission_checklist.md |
| TADSR_REPORTING_READINESS_AUDIT | PASS | README/docs/report consistently document completed boundaries and honest gaps | update README/docs/report wording |
| TADSR_LARGE_ARTIFACT_POLICY_AUDIT | PASS | large tensor artifacts are ignored; metadata/report files are committed | check .gitignore and staged files before commit |
| TADSR_FINAL_PACKAGING_READINESS | PASS | final evidence package is ready for presentation/repository handoff while full inference remains guarded | prepare final presentation/video/repository handoff |
| TADSR_FINAL_PRESENTATION_PACKAGE | PASS | final PPT outline and presentation handoff docs are ready | python3 scripts/validate_final_presentation_package.py |
| TADSR_VIDEO_SCRIPT_READY | PASS | final video script is ready and avoids full-inference/image-generation claims | update docs/04_video_script.md |
| TADSR_DEMO_RUNBOOK_READY | PASS | demo runbook lists reproducible audit/alignment commands and expected markers | update docs/final_demo_runbook.md |
| TADSR_REPOSITORY_HANDOFF_READY | PASS | repository handoff guide maps code, evidence, large-file policy and limitations | update docs/repository_handoff_guide.md |
| TADSR_FINAL_HANDOFF_READINESS | PASS | repository is ready for final presentation/video recording and submission handoff | record final presentation/video; future technical work should be controlled production CLI validation |
| TADSR_FINAL_PPT_READY | PASS | final PPTX presentation exists and is non-empty | generate deliverables/TADSR-Jittor_final_presentation.pptx |
| TADSR_FINAL_PDF_READY | PASS | final PDF presentation export exists and is non-empty | generate deliverables/TADSR-Jittor_final_presentation.pdf |
| TADSR_FINAL_VIDEO_RECORDING_PACKAGE_READY | PASS | video recording guide exists with safe demo commands and honest limitations | update deliverables/TADSR-Jittor_video_recording_guide.md |
| TADSR_FINAL_SUBMISSION_README_READY | PASS | submission README exists and indexes final deliverables/evidence | update deliverables/TADSR-Jittor_submission_readme.md |
| TADSR_FINAL_DELIVERABLE_SIZE_AUDIT | PASS | final deliverables stay below the 100 MB submission budget | remove large generated artifacts from deliverables if needed |
| TADSR_FINAL_DELIVERABLES_INCLUDE_SMOKE_TRAINING | PASS | final PPT/video/submission README include small-data PyTorch-vs-Jittor smoke-training evidence | update final deliverables with smoke-training logs, curves and alignment summaries |
| TADSR_FINAL_PPT_INCLUDES_SMOKE_TRAINING | PASS | final PPT Markdown/PPTX source includes the small-data smoke-training evidence slide | update docs/03_ppt_outline.md and deliverables/TADSR-Jittor_final_presentation.md |
| TADSR_VIDEO_SCRIPT_INCLUDES_SMOKE_TRAINING | PASS | final video script/recording guide includes the smoke-training segment and demo command | update docs/04_video_script.md and deliverables/TADSR-Jittor_video_recording_guide.md |
| TADSR_SUBMISSION_README_INCLUDES_SMOKE_TRAINING | PASS | submission README indexes smoke-training logs, loss curves, prediction visualizations and multi-seed evidence | update deliverables/TADSR-Jittor_submission_readme.md |
| TADSR_FINAL_DELIVERABLES_READY | PASS | PPT/PDF/video guide/submission README are ready and free of misleading full-inference claims | python3 scripts/validate_final_deliverables.py |
| TADSR_FINAL_SUBMISSION_READY | PASS | final deliverables are ready for submission while full inference remains explicitly guarded | record final video and submit the repository package |
| TADSR_FINAL_SUBMISSION_CONTENT_VALIDATION | PASS | PPT/video/submission wording contains required final markers and avoids misleading claims | python3 scripts/validate_final_submission_content.py |
| TADSR_FINAL_PPT_CONTENT_GUARDRAIL | PASS | PPT Markdown and outline preserve honest full-inference limitations | update docs/03_ppt_outline.md and deliverables/TADSR-Jittor_final_presentation.md |
| TADSR_FINAL_VIDEO_SCRIPT_GUARDRAIL | PASS | video script and recording guide avoid full-inference/image-generation/runtime-LoRA over-claims | update docs/04_video_script.md and deliverables/TADSR-Jittor_video_recording_guide.md |
| TADSR_FINAL_SUBMISSION_README_GUARDRAIL | PASS | submission README indexes evidence and keeps guarded-scope wording | update deliverables/TADSR-Jittor_submission_readme.md |
| TADSR_REPOSITORY_HANDOFF_VALIDATION | PASS | repository handoff files, git readability and large artifact policy are validated | python3 scripts/validate_repository_handoff.py |
| TADSR_GITHUB_SUBMISSION_GUIDE_READY | PASS | GitHub handoff guide exists with remote/push commands and no automatic push policy | update docs/github_submission_handoff.md |
| TADSR_TRACKED_LARGE_ARTIFACT_AUDIT | PASS | no new, staged or out-of-policy .npy/.npz files are present; historical oracle tensors are separately reported | check git ls-files and git status for .npy/.npz |
| TADSR_VIDEO_RECORDING_PREFLIGHT_CHECKLIST | PASS | recording checklist exists with windows, commands, limits and timing | update docs/video_recording_preflight_checklist.md |
| TADSR_RELEASE_CANDIDATE_QA_READINESS | PASS | release-candidate QA package is ready for human GitHub handoff and video recording | human can record final video, upload/push repository to GitHub, and submit deliverables |
| TADSR_SMOKE_TRAINING_DATA_PREP | PASS | deterministic output-tail conv_out feature-target pairs metadata exists; raw .npy tensors stay ignored | python3 tools/export_tadsr_smoke_training_data.py --num-samples 32 --train-samples 24 --val-samples 8 --seed 1234 |
| TADSR_SMOKE_TRAINING_TRAIN_VAL_SPLIT | PASS | smoke training data has a deterministic 24/8 train-validation split over 32 output-tail samples | python3 tools/export_tadsr_smoke_training_data.py --num-samples 32 --train-samples 24 --val-samples 8 --seed 1234 |
| TADSR_PYTORCH_SMOKE_TRAINING | PASS | PyTorch reference output-tail conv_out smoke training ran with real optimizer steps | /mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python tools/train_smoke_pytorch_output_tail.py |
| TADSR_PYTORCH_SMOKE_TRAINING_LOSS_DECREASE | PASS | PyTorch smoke-training loss decreased without NaN/Inf | inspect experiments/smoke_training/output_tail/pytorch/loss.csv |
| TADSR_JITTOR_SMOKE_TRAINING | PASS | Jittor output-tail conv_out smoke training ran with real optimizer steps | USE_CUDA=0 nvcc_path="" python3 scripts/train_smoke_jittor_output_tail.py |
| TADSR_JITTOR_SMOKE_TRAINING_LOSS_DECREASE | PASS | Jittor smoke-training loss decreased without NaN/Inf | inspect experiments/smoke_training/output_tail/jittor/loss.csv |
| TADSR_SMOKE_TRAINING_LOSS_LOG | PASS | PyTorch and Jittor smoke-training loss.csv files exist | run PyTorch/Jittor smoke training scripts |
| TADSR_SMOKE_TRAINING_VALIDATION_LOSS_LOG | PASS | PyTorch and Jittor validation_loss.csv files exist for small-data validation evidence | run PyTorch/Jittor smoke training scripts with --eval-interval 10 |
| TADSR_SMOKE_TRAINING_LOSS_CURVE | PASS | loss_curve.png generated from both smoke-training logs | python3 scripts/plot_smoke_training_curves.py |
| TADSR_SMOKE_TRAINING_TRAIN_VAL_LOSS_CURVE | PASS | train/validation loss curve visualizes both PyTorch and Jittor smoke training | python3 scripts/visualize_smoke_training_alignment.py --data-dir experiments/smoke_training/output_tail |
| TADSR_SMOKE_TRAINING_LOSS_GAP_CURVE | PASS | absolute and relative PyTorch/Jittor loss gap curves are generated | python3 scripts/visualize_smoke_training_alignment.py --data-dir experiments/smoke_training/output_tail |
| TADSR_SMOKE_TRAINING_PERFORMANCE_LOG | PASS | performance_log.csv exists for both PyTorch and Jittor smoke training | run PyTorch/Jittor smoke training scripts |
| TADSR_SMOKE_TRAINING_PERFORMANCE_VISUALIZATION | PASS | step-time and samples/sec visualizations exist for PyTorch/Jittor smoke training | python3 scripts/visualize_smoke_training_alignment.py --data-dir experiments/smoke_training/output_tail |
| TADSR_SMOKE_TRAINING_PYTORCH_JITTOR_LOSS_ALIGNMENT | PASS | PyTorch and Jittor output-tail smoke losses both decrease with aligned trend | python3 scripts/plot_smoke_training_curves.py |
| TADSR_SMOKE_TRAINING_PREDICTION_ALIGNMENT | PASS | PyTorch and Jittor final validation prediction tensors align for the output-tail smoke task | python3 scripts/analyze_smoke_training_alignment.py --data-dir experiments/smoke_training/output_tail |
| TADSR_SMOKE_TRAINING_VALIDATION_ALIGNMENT | PASS | validation loss curves and prediction summaries align between PyTorch and Jittor | python3 scripts/analyze_smoke_training_alignment.py --data-dir experiments/smoke_training/output_tail |
| TADSR_SMOKE_TRAINING_TENSOR_VISUALIZATION | PASS | prediction/target/error tensor heatmaps are generated as diagnostics, not restored images | python3 scripts/visualize_smoke_training_alignment.py --data-dir experiments/smoke_training/output_tail |
| TADSR_SMOKE_TRAINING_MULTI_SEED_STABILITY | PASS | multi-seed small-data output-tail training stability summary exists | python3 scripts/run_smoke_training_multiseed.py --seeds 1234 2025 42 --steps 200 |
| TADSR_SMOKE_TRAINING_ARTIFACTS_TEST | PASS | smoke-training artifacts are present and do not claim full training/inference/image/video generation | python3 tests_jittor_alignment/test_smoke_training_artifacts.py |
| TADSR_SMALL_DATA_TRAINING_READINESS | PASS | small-data training pipeline evidence is ready; this is output-tail smoke training, not full TADSR training | update final presentation/video materials with smoke-training evidence |
| TADSR_GITHUB_HEAD_ARTIFACT_AUDIT | PASS | tracked tensor artifacts are counted and classified without deleting evidence files | python3 scripts/audit_github_release_readiness.py |
| TADSR_GIT_HISTORY_LARGE_BLOB_AUDIT | PASS | Git history blobs are scanned for >100MB hard-limit and >50MB warning risks | python3 scripts/audit_github_release_readiness.py |
| TADSR_GITHUB_DELIVERABLE_SIZE_AUDIT | PASS | final deliverables remain below the 100MB submission budget | python3 scripts/audit_github_release_readiness.py |
| TADSR_GITHUB_WORKTREE_LARGE_ARTIFACT_AUDIT | PASS | worktree/staged tensor artifact risks are reported without adding new tensor files | python3 scripts/audit_github_release_readiness.py |
| TADSR_EVIDENCE_DEPENDENCY_AUDIT | PASS | final evidence manifest depends on metadata/reports rather than requiring raw oracle tensors | python3 scripts/audit_github_release_readiness.py |
| TADSR_GITHUB_RELEASE_SLIMMING_DECISION | PASS | release slimming decision memo exists and records direct-push vs cleanup options | update docs/github_release_slimming_decision.md |
| TADSR_GITHUB_RELEASE_READINESS_AUDIT | PASS | GitHub release readiness audit reports no hard-limit risk or explains remaining size risk | python3 scripts/audit_github_release_readiness.py |
| TADSR_GITHUB_RELEASE_AFTER_PHASE5B_READY | PASS | Phase 5-B summary and diagnostic plan are present in GitHub release readiness evidence | python3 scripts/audit_github_release_readiness.py |
| TADSR_JITTOR_MIGRATION_FEASIBILITY_VALIDATION | PASS | matrix-style migration feasibility validation exists and passes | python3 scripts/validate_jittor_migration_feasibility.py |
| TADSR_MODULE_COVERAGE_MATRIX | PASS | UNet / TimeVAE / LoRA / scheduler / training / deliverables coverage matrix is ready | python3 scripts/validate_jittor_migration_feasibility.py |
| TADSR_WEIGHT_LOADING_COVERAGE_MATRIX | PASS | weight loading and static effective LoRA coverage matrix is ready | python3 scripts/validate_jittor_migration_feasibility.py |
| TADSR_LORA_EFFECTIVE_WEIGHT_COVERAGE_MATRIX | PASS | all active static effective LoRA modules are covered by evidence | python3 scripts/validate_jittor_migration_feasibility.py |
| TADSR_NUMERICAL_ALIGNMENT_COVERAGE_MATRIX | PASS | numerical alignment evidence matrix is ready | python3 scripts/validate_jittor_migration_feasibility.py |
| TADSR_INTEGRATION_PATH_COVERAGE_MATRIX | PASS | UNet + TimeVAE + LoRA + scheduler + minimal integration paths are summarized | python3 scripts/validate_jittor_migration_feasibility.py |
| TADSR_TRAINING_PATH_FEASIBILITY_MATRIX | PASS | small-data PyTorch-vs-Jittor training feasibility matrix is ready | python3 scripts/validate_jittor_migration_feasibility.py |
| TADSR_BOUNDARY_LEVEL_REPRODUCTION | PASS | boundary-level Jittor migration evidence is sufficient for the final report | python3 scripts/validate_jittor_migration_feasibility.py |
| TADSR_SMALL_DATA_TRAINING_ALIGNMENT | PASS | small-data PyTorch-vs-Jittor loss/prediction/multi-seed alignment is recorded | python3 scripts/validate_jittor_migration_feasibility.py |
| TADSR_FULL_INFERENCE_GUARD_VALIDATION | PASS | full inference CLI still raises NotImplementedError and remains guarded | python3 scripts/validate_jittor_migration_feasibility.py |
| TADSR_FULL_INFERENCE_GAP_ANALYSIS | PASS | full inference gap is analyzed and documented while JITTOR_FULL_INFERENCE stays NOT_COMPLETE | python3 scripts/validate_jittor_migration_feasibility.py |
| TADSR_TIMEVAE_FULL_ALIGNMENT_GAP_ANALYSIS | PASS | TimeVAE full-alignment gap is analyzed while TIME_VAE_FULL_ALIGNMENT stays NOT_COMPLETE | python3 scripts/validate_jittor_migration_feasibility.py |
| TADSR_LORA_RUNTIME_GAP_ANALYSIS | PASS | static effective LoRA vs dynamic runtime LoRA boundary is documented | python3 scripts/validate_jittor_migration_feasibility.py |
| TADSR_RUNTIME_LORA_LAYER_FORMULA_ALIGNMENT | PASS | fixed-adapter/fixed-scale LoRA formula equivalence is checked without claiming runtime adapter switching | python3 scripts/validate_lora_layer_formula_alignment.py |
| TADSR_SUBMISSION_FACING_STATUS_SUMMARY | PASS | teacher-facing final status summary exists | python3 scripts/validate_jittor_migration_feasibility.py |
| TADSR_ENVIRONMENT_BLOCKERS_EXPLAINED | PASS | resource and environment blockers are explicitly explained in the teacher-facing summary | python3 scripts/validate_jittor_migration_feasibility.py |
| TADSR_GAP_ANALYSIS_READINESS | PASS | all known gaps are analyzed, scoped, and guarded against false completion claims | python3 scripts/validate_jittor_migration_feasibility.py |
| TADSR_PRODUCTION_COMPLETION_READINESS | PASS | production completion branch/readiness checks are generated without opening full inference | python3 scripts/validate_production_completion_readiness.py |
| TADSR_PRODUCTION_COMPLETION_BRANCH_READY | PASS | codex/tadsr-production-completion branch and submission-freeze ref are available | git branch --show-current |
| TADSR_PRODUCTION_COMPLETION_BASELINE_PRESERVED | PASS | baseline submission markers and NotImplemented full-inference guard are preserved | python3 scripts/validate_production_completion_readiness.py |
| TADSR_TIMEVAE_FULL_PRODUCTION_PATH_AUDIT | PASS | TimeVAE production path audit report exists; PARTIAL means live official runtime remains a blocker | python3 tools/audit_timevae_full_production_path.py --metadata-only 1 |
| TADSR_OFFICIAL_RUNTIME_LORA_BEHAVIOR_AUDIT | PASS | official runtime LoRA behavior audit report exists; PARTIAL means existing reports were used | python3 tools/audit_official_runtime_lora_behavior.py --metadata-only 1 |
| TADSR_DYNAMIC_RUNTIME_LORA_NOT_REQUIRED_FOR_OFFICIAL_INFERENCE | PASS | existing fixed-adapter evidence indicates dynamic LoRA is not required for the audited fixed inference path | python3 tools/audit_official_runtime_lora_behavior.py --metadata-only 1 |
| TADSR_FULL_INFERENCE_CONTROLLED_VALIDATION_PLAN | PASS | controlled full-inference validation plan exists; stages were not executed in this phase | inspect docs/production_completion/full_inference_controlled_validation_plan.md |
| TADSR_PRODUCTION_COMPLETION_BLOCKER_REPORT | PASS | blocker report documents official-env and future-stage limits without failing current submission | inspect experiments/production_completion/blockers/production_completion_blockers.md |
| TADSR_PRODUCTION_COMPLETION_PHASE1_VALIDATION | PASS_WITH_BLOCKERS | Phase 1 validation summarizes readiness/audits/plan/blockers without upgrading NOT_COMPLETE markers | python3 scripts/validate_production_completion_phase1.py |
| TADSR_PRODUCTION_OFFICIAL_ENV_RESOLUTION | PASS | Phase 2 official repo/weights/python availability is resolved without loading models | python3 scripts/resolve_production_official_env.py |
| TADSR_TIMEVAE_PRODUCTION_ORACLE_METADATA | PASS | TimeVAE production oracle metadata report exists or is explicitly blocked by official environment | python3 tools/export_timevae_production_metadata_oracle.py --metadata-only 1 |
| TADSR_TIMEVAE_METADATA_PARTIAL_DIAGNOSIS | PASS | TimeVAE metadata PARTIAL reason, missing fields and next repair actions are reported | python scripts/diagnose_timevae_metadata_partial.py |
| TADSR_OFFICIAL_DIFFUSERS_OVERLAY_READY | PASS | isolated diffusers overlay readiness is recorded without modifying the official strict venv | python3 scripts/prepare_official_runtime_dependency_overlay.py --execute 1 |
| TADSR_OFFICIAL_RUNTIME_DEPENDENCY_REPAIR_READINESS | PASS | official dependency repair readiness is recorded from overlay dry-run or execution | python3 scripts/prepare_official_runtime_dependency_overlay.py --execute 1 |
| TADSR_OFFICIAL_RUNTIME_DEPENDENCY_OVERLAY_ACTIVE | PASS | official dependency diagnosis records whether PYTHONPATH overlay was active for imports | python3 scripts/diagnose_official_runtime_dependencies.py --pythonpath-overlay <overlay> |
| TADSR_OFFICIAL_RUNTIME_DEPENDENCY_DIAGNOSIS | PASS | official Python runtime dependency diagnosis is recorded without modifying the environment | python3 scripts/diagnose_official_runtime_dependencies.py |
| TADSR_TIMEVAE_METADATA_REPAIR_ATTEMPT | PASS | TimeVAE production metadata repair attempt is recorded without running full inference | python3 tools/export_timevae_production_metadata_oracle.py --metadata-only 1 |
| TADSR_TIMEVAE_METADATA_FIELDS_COMPLETE | PASS | TimeVAE required metadata field completeness is evaluated explicitly | python3 tests_jittor_alignment/test_timevae_production_alignment_preflight.py |
| TADSR_TIMEVAE_METADATA_READY_FOR_ONE_STEP | PASS | TimeVAE metadata readiness for controlled one-step tensor alignment is evaluated separately | python3 scripts/validate_full_inference_metadata_contract.py |
| TADSR_TIMEVAE_LIVE_METADATA_COMPLETION | PASS | controlled TimeVAE live encode/decode metadata completion is validated from JSON only | python3 scripts/validate_timevae_live_metadata_completion.py |
| TADSR_TIMEVAE_LIVE_ENCODE_METADATA | PASS | controlled live TimeVAE encode/posterior/latent metadata fields are complete | python3 scripts/validate_timevae_live_metadata_completion.py |
| TADSR_TIMEVAE_LIVE_DECODE_METADATA | PASS | controlled live TimeVAE decode/clamp metadata fields are complete | python3 scripts/validate_timevae_live_metadata_completion.py |
| TADSR_TIMEVAE_LIVE_SAFETY_FLAGS | PASS | controlled live TimeVAE metadata export proves no full inference, scheduler loop, UNet call, image/video generation or raw tensor commit | python3 scripts/validate_timevae_live_metadata_completion.py |
| TADSR_TIMEVAE_PRODUCTION_ALIGNMENT_PREFLIGHT | PASS | Jittor TimeVAE production alignment preflight checks metadata contracts without upgrading full alignment | python3 tests_jittor_alignment/test_timevae_production_alignment_preflight.py |
| TADSR_FULL_INFERENCE_METADATA_DRY_RUN_CONTRACT | PARTIAL | full-inference stage contract is documented without running denoising loop or image generation | python3 scripts/validate_full_inference_metadata_contract.py |
| TADSR_PRODUCTION_COMPLETION_PHASE2_VALIDATION | PASS_WITH_BLOCKERS | Phase 2 validation aggregates official env resolution, TimeVAE metadata, LoRA audit and metadata contract | python3 scripts/validate_production_completion_phase2.py |
| TADSR_PRODUCTION_COMPLETION_PHASE3_VALIDATION | PASS | Phase 3 validation aggregates live official-env readiness, TimeVAE metadata export, LoRA live audit and the full-inference metadata contract | python3 scripts/validate_production_completion_phase3.py |
| TADSR_TIMEVAE_LIVE_METADATA_EXPORT | PASS | live TimeVAE metadata export is PASS only when official runtime metadata oracle is actually available | python3 scripts/validate_production_completion_phase3.py |
| TADSR_LORA_LIVE_RUNTIME_AUDIT | PASS | live LoRA runtime audit is PASS only when official runtime evidence is available; PARTIAL means existing reports were used | python3 scripts/validate_production_completion_phase3.py |
| TADSR_RUNTIME_LORA_DECISION_FINALIZED_FOR_FIXED_INFERENCE | PASS | fixed-adapter official inference path does not require dynamic runtime LoRA; dynamic runtime LoRA remains future work | inspect docs/lora_runtime_gap_analysis.md |
| TADSR_SMOKE_TRAINING_SUBMISSION_SUMMARY | PASS | small-data PyTorch-vs-Jittor smoke-training evidence is consolidated for submission | python scripts/summarize_smoke_training_evidence.py |
| TADSR_FULL_INFERENCE_CONTRACT_READY_FOR_ONE_STEP | PASS | one-step tensor alignment readiness is separate from full inference execution and remains blocked until Phase 3 prerequisites pass | python3 scripts/validate_production_completion_phase3.py |
| TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT_PLAN | PASS | controlled one-step tensor alignment plan is generated only after the live metadata and readiness gates pass | inspect docs/production_completion/one_step_tensor_alignment_plan.md and experiments/production_completion/full_inference/one_step_tensor_alignment_plan.md |
| TADSR_PHASE3B_LINUX_MANUAL_RUNBOOK_READY | PASS | Linux Phase 3-B manual execution runbook exists and contains safe instructions | inspect docs/production_completion/linux_phase3b_manual_runbook.md |
| TADSR_PHASE3B_LINUX_LIVE_AUDIT_SCRIPT_READY | PASS | Linux one-click Phase 3-B live audit script exists and remains metadata/audit-only | bash -n scripts/run_phase3b_live_audit_linux.sh |
| TADSR_PHASE3B_RESULT_PACKAGER_READY | PASS | Phase 3-B result packager exists and only packages JSON/Markdown/log/txt evidence | python3 scripts/package_phase3b_live_results.py --repo-root . --output experiments/production_completion/phase3b_live_results_package.zip |
| TADSR_PHASE3B_RESULT_IMPORT_VALIDATOR_READY | PASS | Windows/local Phase 3-B result import validator exists and rejects raw tensors | python scripts/import_phase3b_live_results.py --package phase3b_live_results_package.zip --repo-root . |
| TADSR_PHASE3B_MANUAL_EXECUTION_KIT_READY | PASS | manual Linux execution kit is ready while live audit remains blocked by authentication / official runtime gates | manual login: ssh -p 10022 sj@10.195.21.2 |
| TADSR_PHASE3B_FINAL_MANUAL_HANDOFF_READY | PASS | manual handoff validation confirms Linux runbook, env template, live-audit script, packager and importer are ready | python scripts/validate_phase3b_manual_handoff.py |
| TADSR_PHASE3B_LIVE_RESULTS_IMPORT_VALIDATION | PASS | live result import validation is PASS only after a Linux package is safely imported | python scripts/import_phase3b_live_results.py --package phase3b_live_results_package.zip --repo-root . |
| TADSR_PHASE3B_LIVE_RESULTS_NO_RAW_TENSORS | PASS | live result import package contains no .npy/.npz/local_tensors/weights/checkpoints | inspect experiments/production_completion/phase3_import_validation.md |
| TADSR_PHASE3D_LIVE_RESULTS_PACKAGE_FOUND | PASS | Phase 3-D found the Linux live-audit result package in an allowed local candidate path | inspect experiments/production_completion/phase3d_import_gate_status.md |
| TADSR_PHASE3D_LIVE_RESULTS_PACKAGE_SECURITY | PASS | Phase 3-D package security passes only after safe dry-run / no-raw-tensor validation | python scripts/import_phase3b_live_results.py --package phase3b_live_results_package.zip --repo-root . --dry-run 1 |
| TADSR_PHASE3D_IMPORT_GATE | PASS | Phase 3-D import gate passes only after live results are safely imported | inspect experiments/production_completion/phase3d_import_gate_status.md |
| TADSR_PHASE3D_ONE_STEP_GATE_DECISION | PASS | one-step tensor alignment gate decision is separate from full inference and remains blocked until prerequisites pass | inspect experiments/production_completion/phase3_import_decision.md |
| TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT_PROTOCOL_READY | PASS | controlled one-step protocol is ready only after import gate and one-step prerequisites pass | inspect experiments/production_completion/full_inference/one_step_tensor_alignment_protocol.md |
| TADSR_FULL_INFERENCE_ONE_STEP_GATE_BLOCKER_SUMMARY | PASS | blocker summary explains why controlled one-step tensor alignment is not yet allowed | inspect experiments/production_completion/full_inference/one_step_gate_blocker_summary.md |
| TADSR_ONE_STEP_OFFICIAL_PATH_AUDIT | PASS | official controlled one-step tensor path is audited without running the full denoising loop or image/video generation | python3 tools/audit_official_tadsr_one_step_tensor_path.py |
| TADSR_ONE_STEP_OFFICIAL_ORACLE_TENSORS | PASS | official PyTorch one-step oracle tensor metadata is exported; raw tensors remain local_tensors-only and ignored | python3 tools/export_tadsr_one_step_tensor_oracle.py |
| TADSR_ONE_STEP_JITTOR_TENSOR_RUN | PASS | Jittor executes the corresponding controlled one-step tensor path when official local tensors are available | python3 scripts/run_jittor_one_step_tensor_alignment.py |
| TADSR_ONE_STEP_TENSOR_STAGE_ALIGNMENT | PASS | per-stage shape/range/error/cosine alignment passes only for controlled one-step tensor stages | python3 scripts/run_jittor_one_step_tensor_alignment.py |
| TADSR_ONE_STEP_TENSOR_ALIGNMENT_VALIDATION | PASS | metadata-only validator checks one-step artifacts, safety flags, and raw tensor policy | python3 scripts/validate_one_step_tensor_alignment.py |
| TADSR_ONE_STEP_NO_IMAGE_VIDEO_GUARD | PASS | one-step phase did not generate restored images or videos | python3 scripts/validate_one_step_tensor_alignment.py |
| TADSR_ONE_STEP_RAW_TENSOR_POLICY | PASS | one-step raw .npy/.npz tensors are not tracked or staged | python3 scripts/validate_one_step_tensor_alignment.py |
| TADSR_ONE_STEP_FULL_INFERENCE_GUARD_PRESERVED | PASS | production full-inference CLI remains guarded by NotImplementedError | python3 scripts/validate_one_step_tensor_alignment.py |
| TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT | PASS | controlled one-step tensor alignment status; this is not full TADSR inference completion | python3 scripts/validate_one_step_tensor_alignment.py |
| TADSR_OFFICIAL_ACTUAL_INFERENCE_PATH_AUDIT | PASS | official TADSR actual inference path is audited without executing production full inference | python3 tools/audit_official_tadsr_actual_inference_path.py |
| TADSR_OFFICIAL_INFERENCE_MULTISTEP_REQUIREMENT_AUDIT | PASS | single-step vs multi-step requirement is determined from the official actual path | python3 tools/audit_official_tadsr_actual_inference_path.py |
| TADSR_OFFICIAL_ACTUAL_PATH_SINGLE_STEP_CONTRACT | PASS | official actual inference path is documented as a single-step/get_x0_from_res contract when applicable | python3 tools/audit_official_tadsr_actual_inference_path.py |
| TADSR_POSTPROCESS_CONTRACT_AUDIT | PASS | postprocess/output tensor-to-image contract is documented without image/video generation | python3 tools/audit_official_tadsr_postprocess_contract.py |
| TADSR_OUTPUT_IMAGE_SAVE_POLICY_AUDIT | PASS | official image save policy is audited but not executed | python3 tools/audit_official_tadsr_postprocess_contract.py |
| TADSR_POSTPROCESS_NOT_EXECUTED_GUARD | PASS | postprocess image save remains documented_not_executed in this project | python3 tools/audit_official_tadsr_postprocess_contract.py |
| TADSR_TINY_MULTISTEP_ALIGNMENT_APPLICABILITY | NOT_REQUIRED_FOR_OFFICIAL_ACTUAL_INFERENCE | decision marker records whether tiny multi-step is required by the official actual path | python3 scripts/decide_multistep_alignment_applicability.py |
| TADSR_TINY_MULTISTEP_ALIGNMENT_DECISION | PASS | multi-step applicability decision is generated without running multi-step alignment | python3 scripts/decide_multistep_alignment_applicability.py |
| TADSR_EXPERIMENTAL_CLI_METADATA_ONLY_PLAN | PASS | future experimental metadata-only CLI plan is documented without implementation | inspect docs/production_completion/experimental_cli_metadata_only_plan.md |
| TADSR_PHASE5B_SUBMISSION_FREEZE_SUMMARY | PASS | Phase 5-B final submission freeze summary exists and preserves honest scope | inspect docs/final_phase5b_submission_freeze_summary.md |
| TADSR_DIAGNOSTIC_IMAGE_SMOKE_PLAN | PASS | future diagnostic image-smoke plan is documented without execution | inspect docs/production_completion/diagnostic_image_smoke_plan.md |
| TADSR_DIAGNOSTIC_IMAGE_SMOKE_EXECUTED | PASS | diagnostic image-smoke execution status is recorded honestly | python3 scripts/validate_diagnostic_image_smoke.py |
| TADSR_DIAGNOSTIC_IMAGE_SMOKE_READY | PASS | diagnostic image-smoke readiness is recorded as PASS or honest PARTIAL/BLOCKED | python3 scripts/generate_diagnostic_image_smoke.py |
| TADSR_DIAGNOSTIC_IMAGE_SMOKE_ALIGNMENT | PASS | diagnostic image-smoke alignment is PASS only when local tensors and PNG evidence exist | python3 scripts/generate_diagnostic_image_smoke.py |
| TADSR_DIAGNOSTIC_IMAGE_SMOKE_NOT_FULL_INFERENCE | PASS | diagnostic image-smoke explicitly does not run full inference | python3 scripts/validate_diagnostic_image_smoke.py |
| TADSR_DIAGNOSTIC_IMAGE_SMOKE_NO_RAW_TENSORS | PASS | diagnostic image-smoke does not stage raw .npy/.npz tensors | python3 scripts/validate_diagnostic_image_smoke.py |
| TADSR_DIAGNOSTIC_IMAGE_SMOKE_ARTIFACTS_READY | PASS | diagnostic image-smoke artifacts are available or honestly partial | python3 scripts/validate_diagnostic_image_smoke.py |
| TADSR_DIAGNOSTIC_IMAGE_SMOKE_VALIDATION | PASS | diagnostic image-smoke validator checks artifacts and guardrails | python3 scripts/validate_diagnostic_image_smoke.py |
| TADSR_DIAGNOSTIC_IMAGE_SMOKE_FALSE_CLAIM_GUARD | PASS | diagnostic image-smoke wording avoids false full-inference/image-generation claims | python3 scripts/validate_diagnostic_image_smoke.py |
| TADSR_DIAGNOSTIC_IMAGE_SMOKE_ARTIFACT_POLICY | PASS | diagnostic image-smoke artifact policy rejects staged raw tensors | python3 scripts/validate_diagnostic_image_smoke.py |
| TADSR_ONE_STEP_DIAGNOSTIC_CLI_READY | PASS | metadata-only one-step diagnostic CLI is ready | python3 scripts/validate_one_step_diagnostic_cli.py |
| TADSR_ONE_STEP_DIAGNOSTIC_CLI_NOT_FULL_INFERENCE | PASS | one-step diagnostic CLI keeps full inference guarded | python3 scripts/validate_one_step_diagnostic_cli.py |
| TADSR_TIMEVAE_FULL_ALIGNMENT_CLOSURE_SUMMARY | PASS | TimeVAE closure summary exists without upgrading full alignment | python3 scripts/validate_timevae_closure.py |
| TADSR_TIMEVAE_FULL_ALIGNMENT_REMAINS_SCOPED | PASS | TimeVAE full alignment remains scoped and NOT_COMPLETE | python3 scripts/validate_timevae_closure.py |
| TADSR_RUNTIME_LORA_FINAL_DECISION_PROOF | PASS | runtime LoRA final decision proof is documented | python3 scripts/validate_runtime_lora_decision.py |
| TADSR_DYNAMIC_RUNTIME_LORA_NOT_REQUIRED_FOR_AUDITED_PATH | PASS | dynamic runtime LoRA is not required for the audited fixed path | python3 scripts/validate_runtime_lora_decision.py |
| TADSR_OFFICIAL_ACTUAL_SINGLE_STEP_PATH_DOCUMENTED | PASS | official actual path is documented as single-step/get_x0_from_res | python3 scripts/validate_production_completion_phase3.py |
| TADSR_MULTISTEP_NOT_REQUIRED_FOR_OFFICIAL_ACTUAL_PATH | PASS | tiny multi-step is documented as not required for the official actual path | python3 scripts/validate_production_completion_phase3.py |
| TADSR_CONTROLLED_ONE_STEP_EVIDENCE_DOCUMENTED | PASS | controlled one-step tensor evidence and key metrics are documented | python3 scripts/validate_production_completion_phase3.py |
| TADSR_FINAL_SUBMISSION_AFTER_PHASE5B_READY | PASS | final submission materials are ready after Phase 5-B without opening full inference | python3 scripts/validate_production_completion_phase3.py |
| TADSR_FINAL_SUBMISSION_WITH_DIAGNOSTIC_SMOKE_READY | PASS | final submission materials include Phase 6-A diagnostic smoke/CLI/closure proof status | python3 scripts/validate_production_completion_phase3.py |
| TADSR_COURSE_REQUIREMENT_COMPLIANCE | PASS | course requirement matrix covers the submitted evidence | python3 scripts/validate_course_requirement_compliance.py |
| TADSR_COURSE_REQUIREMENT_EVIDENCE_MATRIX | PASS | course requirement evidence matrix is generated | python3 scripts/validate_course_requirement_compliance.py |
| TADSR_COURSE_TRAINING_REQUIREMENT_EVIDENCE | PASS | training logs/loss curves/performance evidence are indexed for course requirements | python3 scripts/validate_course_requirement_compliance.py |
| TADSR_COURSE_VISUALIZATION_REQUIREMENT_EVIDENCE | PASS | visualization evidence is indexed for course requirements | python3 scripts/validate_course_requirement_compliance.py |
| TADSR_COURSE_GITHUB_REQUIREMENT_EVIDENCE | PASS | GitHub submission readiness evidence is indexed for course requirements | python3 scripts/validate_course_requirement_compliance.py |
| TADSR_COURSE_PPT_VIDEO_REQUIREMENT_EVIDENCE | PASS | PPT/PDF/video guide evidence is indexed for course requirements | python3 scripts/validate_course_requirement_compliance.py |
| TADSR_FINAL_EVIDENCE_INDEX | PASS | teacher-readable evidence index is generated | python3 scripts/build_final_evidence_index.py |
| TADSR_FINAL_EVIDENCE_INDEX_TEACHER_READABLE | PASS | final evidence index is readable by course reviewers | python3 scripts/build_final_evidence_index.py |
| TADSR_TRAINING_ALIGNMENT_EVIDENCE_VALIDATION | PASS | small-data training evidence completeness is audited | python3 scripts/validate_training_alignment_evidence.py |
| TADSR_TRAINING_LOSS_CURVE_EVIDENCE | PASS | training and validation loss curves are present | python3 scripts/validate_training_alignment_evidence.py |
| TADSR_TRAINING_PERFORMANCE_LOG_EVIDENCE | PASS | training performance logs are present | python3 scripts/validate_training_alignment_evidence.py |
| TADSR_TRAINING_OUTPUT_ALIGNMENT_EVIDENCE | PASS | training prediction/output alignment visualizations are present | python3 scripts/validate_training_alignment_evidence.py |
| TADSR_TRAINING_GRAD_PARAM_UPDATE_EVIDENCE | PARTIAL | parameter-update evidence exists; gradient norm may be optional if not logged originally | python3 scripts/validate_training_alignment_evidence.py |
| TADSR_TRAINING_EVIDENCE_TEACHER_READY | PASS | training evidence is ready for teacher review without claiming full TADSR training | python3 scripts/validate_training_alignment_evidence.py |
| TADSR_FINAL_CLAIMS_CONSISTENCY | PASS | final materials preserve scope consistency | python3 scripts/validate_final_claims_consistency.py |
| TADSR_FINAL_FALSE_CLAIM_GUARD | PASS | final materials avoid misleading full-inference/image-generation/runtime-LoRA claims | python3 scripts/validate_final_claims_consistency.py |
| TADSR_FINAL_SCOPE_CONSISTENCY | PASS | required NOT_COMPLETE / NOT_IMPLEMENTED statuses remain visible | python3 scripts/validate_final_claims_consistency.py |
| TADSR_DEFENSE_RISK_RESPONSE_PACK | PASS | Chinese defense risk-response pack covers high-risk reviewer questions | python3 scripts/validate_defense_risk_response_pack.py |
| TADSR_DEFENSE_SHORT_ANSWERS_READY | PASS | short Chinese defense answers are ready for live Q&A | python3 scripts/validate_defense_risk_response_pack.py |
| TADSR_DEFENSE_LONG_ANSWERS_READY | PASS | long Chinese defense answers are ready for detailed reviewer follow-up | python3 scripts/validate_defense_risk_response_pack.py |
| TADSR_DEFENSE_FALSE_CLAIM_GUARD | PASS | defense materials avoid unguarded full-inference/image-generation/runtime-LoRA overclaims | python3 scripts/validate_defense_risk_response_pack.py |
| TADSR_DEFENSE_EVIDENCE_LOOKUP_READY | PASS | defense evidence lookup table maps claims to files and demo commands | python3 scripts/validate_defense_risk_response_pack.py |
| TADSR_FINAL_RELEASE_CANDIDATE_SIGNOFF | PASS | final release candidate signoff report is present | docs/final_release_candidate_signoff.md |
| TADSR_FINAL_RELEASE_CANDIDATE_TECHNICAL_EVIDENCE | PASS | final signoff summarizes core technical evidence | docs/final_release_candidate_signoff.md |
| TADSR_FINAL_RELEASE_CANDIDATE_SCOPE_GUARD | PASS | final signoff preserves NOT_COMPLETE / NOT_IMPLEMENTED scope guards | docs/final_release_candidate_signoff.md |
| TADSR_FINAL_LINKS_AND_PATHS_VALIDATION | PASS | final critical links and paths are validated | python3 scripts/validate_final_links_and_paths.py |
| TADSR_FINAL_MARKDOWN_LINKS_VALIDATION | PASS | final Markdown links are scanned; noncritical warnings are recorded | python3 scripts/validate_final_links_and_paths.py |
| TADSR_FINAL_PLACEHOLDER_SCAN | PASS | final materials are scanned for obvious placeholders | python3 scripts/validate_final_links_and_paths.py |
| TADSR_FINAL_CHINESE_MATERIALS_VALIDATION | PASS | final reviewer-facing materials are primarily Chinese | python3 scripts/validate_final_chinese_materials.py |
| TADSR_FINAL_MOJIBAKE_SCAN | PASS | final Chinese materials are scanned for obvious mojibake | python3 scripts/validate_final_chinese_materials.py |
| TADSR_FINAL_CHINESE_READABILITY_READY | PASS | final Chinese materials are ready for human review | python3 scripts/validate_final_chinese_materials.py |
| TADSR_FINAL_COMMAND_INDEX | PASS | final demo command index is present | docs/final_command_index.md |
| TADSR_FINAL_DEMO_COMMANDS_READY | PASS | teacher-facing demo commands are grouped and explained | docs/final_command_index.md |
| TADSR_FINAL_GITHUB_UPLOAD_CHECKLIST | PASS | GitHub / LMS upload checklist is present | docs/final_github_upload_checklist.md |
| TADSR_FINAL_UPLOAD_PACKAGE_READY | PASS | upload package safety checklist is ready | docs/final_github_upload_checklist.md |
| TADSR_CLEAN_PUBLIC_RELEASE_PACKAGE_BUILD | PASS | clean public release tree is built for manual GitHub upload | python3 scripts/build_clean_public_release_package.py |
| TADSR_CLEAN_PUBLIC_RELEASE_NO_RAW_TENSORS | PASS | clean public release tree excludes raw tensors and large private artifacts | python3 scripts/build_clean_public_release_package.py |
| TADSR_CLEAN_PUBLIC_RELEASE_SIZE_AUDIT | PASS | clean public release tree is below the course upload size budget | python3 scripts/build_clean_public_release_package.py |
| TADSR_CLEAN_PUBLIC_RELEASE_PACKAGE_VALIDATION | PASS | clean public release tree validates as GitHub-ready | python3 scripts/validate_clean_public_release_package.py |
| TADSR_CLEAN_PUBLIC_RELEASE_GITHUB_READY | PASS | clean public release tree can be manually uploaded to GitHub | python3 scripts/validate_clean_public_release_package.py |
| TADSR_CLEAN_PUBLIC_RELEASE_FALSE_CLAIM_GUARD | PASS | clean public release tree avoids misleading completion claims | python3 scripts/validate_clean_public_release_package.py |
| TADSR_FINAL_GITHUB_URL_STATUS | PARTIAL_GITHUB_URL_PENDING | GitHub URL status is recorded; pending is acceptable before manual repo creation | python3 scripts/set_final_github_url.py |
| TADSR_FINAL_GITHUB_URL_UPDATE_SCRIPT_READY | PASS | GitHub URL update script is ready and does not push or add remotes | python3 scripts/set_final_github_url.py |
| TADSR_FINAL_HUMAN_SUBMISSION_INSTRUCTIONS | PASS | final human submission instructions are present | docs/final_human_submission_instructions.md |
| TADSR_FINAL_VIDEO_REHEARSAL_CHECKLIST | PASS | final video rehearsal checklist is present | docs/final_video_rehearsal_checklist.md |
| TADSR_FINAL_VIDEO_RECORDING_READY | PASS | video recording flow is ready for human execution | docs/final_video_rehearsal_checklist.md |
| TADSR_FINAL_SUBMISSION_FREEZE_TAG_DOC | PASS | final freeze/tag documentation is present without auto-tagging or pushing | docs/final_submission_freeze_tag.md |
| TADSR_FINAL_HUMAN_SUBMISSION_LOCK_REPORT | PASS | final human submission lock report is present | docs/final_human_submission_lock_report.md |
| TADSR_FINAL_HUMAN_SUBMISSION_LOCK_READY | PASS | final human submission lock is ready; only manual submission actions remain | docs/final_human_submission_lock_report.md |
| TADSR_FINAL_VIDEO_SUBMISSION_CHECK | PASS | final video submission check is present | docs/final_video_submission_check.md |
| TADSR_FINAL_VIDEO_RECORDING_READY | PASS | video recording flow is ready for human execution | docs/final_video_submission_check.md |
| TADSR_FULL_ENGINEERING_COMPLETION_ROADMAP | PASS | future full-engineering completion roadmap is documented without execution | docs/full_engineering_completion_roadmap.md |
| TADSR_CONTROLLED_EXTENSION_PLAN_READY | PASS | controlled extension plan is ready if future work is approved | docs/full_engineering_completion_roadmap.md |
| TADSR_TEACHER_REVIEW_GUIDE_READY | PASS | teacher review guide exists and indexes fast review commands/files | update docs/teacher_review_guide.md |
| TADSR_FINAL_DEFENSE_QA_READY | PASS | final defense Q&A exists for high-risk reviewer questions | update docs/final_defense_QA.md |
| TADSR_FINAL_PERFECTION_READINESS | PASS | course compliance, evidence index, training audit, claims guard, defense guide and roadmap are ready while guarded limitations remain honest | run Final Perfection validators and keep full inference guarded |
| TADSR_FINAL_READY_FOR_HUMAN_SUBMISSION | PASS | release candidate signoff, links/path QA, Chinese readability QA, command index and GitHub upload checklist are ready | run final release-candidate QA validators and keep full inference guarded |
| TADSR_FINAL_PUBLIC_RELEASE_AND_SUBMISSION_READY | PASS | clean public release tree, URL setter, human submission instructions, video rehearsal and freeze docs are ready | create GitHub repo manually, run set_final_github_url.py with the real URL, record video, then submit |
| TADSR_FINAL_MANUAL_ACTIONS_ONLY_REMAINING | PASS | all technical and material checks are complete; remaining work is GitHub creation, URL writeback, video recording and course submission only | create GitHub repo manually, upload clean public release, write back URL, record video and submit |

## NEXT_ACTION

Manual actions only: create the GitHub repo, upload the clean public release, run set_final_github_url.py with the real URL, record the video, and submit PPT/PDF/video/GitHub URL. Do not expand technical scope further.
