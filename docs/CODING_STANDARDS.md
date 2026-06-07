# Neuronix Coding Standards

## Language

- Use C++20.
- Prefer standard library facilities before custom utilities.
- Do not introduce external math or machine learning libraries.

## Naming

- Namespace: `neuronix`.
- Types: `PascalCase`.
- Functions and variables: `snake_case`.
- Constants: descriptive `snake_case` unless they are preprocessor macros.
- Macros: `NEURONIX_UPPER_SNAKE_CASE`.

## Headers And Sources

- Public headers live in `include/neuronix/...`.
- Private implementation lives in `src/...`.
- Keep headers minimal and include what they use.
- Do not expose implementation details through public headers unless they are part of the intended API.

## Error Handling

- Prefer explicit validation at API boundaries.
- Use exceptions for invalid user-facing operations when recovery is possible at the caller level.
- Use assertions only for internal invariants that indicate programmer error.

## Testing

- Every public operation must have unit tests.
- Matrix operations in Phase 2 must be fully tested before higher-level systems depend on them.
- Tests should cover success cases, dimension errors, boundary conditions, and numerical behavior.

## Performance

- Avoid unnecessary allocations in numeric hot paths.
- Prefer clear, correct code first, then benchmark before optimizing.
- Keep benchmark code separate from unit tests.
