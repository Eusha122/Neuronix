# Phase 1: Production Foundation

Phase 1 initializes Neuronix as a clean, buildable C++20 project.

## Included

- CMake project configuration
- `neuronix` library target
- Public headers under `include/neuronix`
- Private implementation under `src`
- Catch2 smoke test
- GitHub Actions build workflow
- Example executable
- Benchmark scaffold
- Documentation for architecture, coding standards, and roadmap

## Excluded

- Matrix implementation
- Tensor implementation
- Layer, activation, loss, optimizer, or model APIs
- Serialization file formats
- Logging behavior
- Training and inference
- Image loading
- GPU acceleration

## Acceptance Criteria

- Configure succeeds.
- Build succeeds.
- Tests run through CTest.
- The foundation does not depend on machine learning frameworks or external math libraries.
- The repository is ready for Phase 2 Matrix work.
