# Security Policy

This repository is a hackathon ranking system. It should not contain secrets, API keys, private tokens, or private candidate data beyond the official competition dataset and generated artifacts needed for reproduction.

## Supported Version

The supported version is the current repository state used for the Redrob Hackathon submission.

## Reporting A Problem

If you find a security-sensitive issue, contact the primary maintainer listed in `submission_metadata.yaml` instead of opening a public issue.

Examples:

- accidental secrets or credentials
- private data committed unintentionally
- unsafe file handling
- reproducibility issue that could expose local paths or system details

## Runtime Security Expectations

The official ranking path:

- does not call external APIs
- does not require network access
- does not require GPU access
- reads only the input candidate file and precomputed artifacts
- writes only the requested CSV and debug trace

The offline preprocessing path may download or load third-party models when artifacts need to be rebuilt. That is separate from the final evaluated ranking path.
