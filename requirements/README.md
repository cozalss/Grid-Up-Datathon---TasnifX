# Dependency policy

The base.in, dev.in, and security.in files describe intent. Local setup and
every CI/release job install the same cross-platform, hash-backed graph with
`uv sync --locked`; CI also proves `uv lock --check`. The minor-specific
constraints remain compatibility/audit exports for legacy consumers, not the
CI resolver. Each Python 3.10-3.13 matrix job audits its marker-resolved lock
export. The local uv bootstrap is itself exact-versioned and hash-enforced.

The Kaggle offline wheels have a stricter, artifact-level contract in
security/wheel-manifest.json: exact normalized package name, exact version,
exact filename, target platform, and a SHA-256 calculated from the actual
download. An unknown digest is written as unverified; it must never be
replaced with a guessed value, and publication/installation must fail.

Update procedure:

1. Let Dependabot open the weekly pip/action update PR, or change intent manually.
2. Run `uv lock --upgrade-package NAME` and review the complete `uv.lock` diff.
3. Update compatibility constraints and download wheels for the declared target.
4. Calculate SHA-256 from those bytes and update the wheel manifest.
5. Run the supply-chain contracts, locked pip-audit, and installed-environment SBOM.

Global workstation packages are deliberately outside this contract. CI builds
a clean project environment and reports only dependencies resolved for this
repository.
