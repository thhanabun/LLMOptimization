# Deprecation And Versioning Policy

The package is still `0.x`, but the public contract is split by module:

- `llm_memlab.production`: stable surface. Breaking changes require a minor version bump and a documented migration note.
- `llm_memlab.experimental`: opt-in surface. APIs can change while kernels are being certified, but changes should remain documented.
- Internal modules may change without compatibility guarantees.

Deprecation policy for stable APIs:

1. Add the replacement API and keep the old symbol working.
2. Document the migration in release notes or docs.
3. Emit a warning only when it will not pollute tight benchmark loops.
4. Remove the old API no earlier than the next minor version.

Schema policy:

- Benchmark DB, quality report, memory profile, and kernel certification outputs include schema version strings.
- CI and dashboards should reject unknown major schema versions.
