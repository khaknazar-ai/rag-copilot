# Limitations

- This is an informational demo, not official consular advice. Its bundled source manifest is not an automated source refresh system, and the source corpus is not included.
- Router evaluation used 154 training examples and a 36-question clean holdout. The results are small, domain-specific offline evidence, not production-scale quality proof.
- Generated-answer quality has not been evaluated systematically for this public candidate. Local CPU inference can have high first-request latency.
- The API has no authentication or rate limiting. Raw-question logging is disabled by default; enabling it stores questions locally. Do not submit personal or sensitive data.
- No model weights or evaluation datasets are included. Users must supply current source documents and an external model/runtime they are authorized to use.
- A clean end-to-end startup and runtime check was not completed for this candidate.
