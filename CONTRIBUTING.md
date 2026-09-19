# Contributing

Contributions must preserve the no-fabrication standard.

1. Do not submit source imaging or annotations unless redistribution rights are documented in `datasets/registry.json`.
2. Add source/evidence identifiers to every biological record and retain original author/source identifiers on imports.
3. Put synthetic fixtures only under `fixtures/synthetic/`, using the `SYN-` namespace.
4. Run `vigilia verify` and the test suite before proposing a release-related change.
5. Never upgrade a prediction to `MANUALLY_VERIFIED` without an attributable review record.
