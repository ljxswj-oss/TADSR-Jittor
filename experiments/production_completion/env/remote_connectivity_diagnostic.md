# Remote SSH connectivity diagnostic

`TADSR_REMOTE_SSH_CONNECTIVITY: AUTH_AVAILABLE_FOR_PHASE3B`

## Result

- Remote host: `10.195.21.2`
- Expected user: `sj`
- Correct SSH port: `10022`
- Reference port `22`: not reachable for SSH (`TcpTestSucceeded=False`)
- Corrected port `10022`: reachable (`TcpTestSucceeded=True`)
- SSH protocol handshake: reached the remote OpenSSH service
- SSH authentication: available for this Phase 3-B run through a temporary in-memory Paramiko session after the user provided updated credentials

## Evidence

The corrected command path is:

```powershell
Test-NetConnection -ComputerName 10.195.21.2 -Port 10022
ssh -p 10022 -o ConnectTimeout=8 -o BatchMode=yes sj@10.195.21.2 "hostname && whoami && pwd && date"
```

The TCP check succeeded on port `10022`. A later temporary Paramiko session authenticated as `sj` and executed the Phase 3-B live audit. The password was not written to any repository file, script, result package, or git commit.

The Linux result package was downloaded to Windows and must still pass the local import gate before any one-step decision can change.

## Safety decision

- Do not treat SSH access or package import as full inference completion.
- Do not enter one-step tensor alignment unless the Phase 3-D one-step gate becomes `PASS`.
- Do not run full denoising loop or production full inference.
- Keep `JITTOR_FULL_INFERENCE: NOT_COMPLETE`.
- Keep `JITTOR_FULL_PORT: PARTIAL`.
- Keep `TIME_VAE_FULL_ALIGNMENT: NOT_COMPLETE`.
- Keep `TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION: NOT_IMPLEMENTED_BY_DESIGN`.
