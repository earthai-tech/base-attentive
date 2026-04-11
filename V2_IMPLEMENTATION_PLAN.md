# BaseAttentive V2 Implementation Plan

## Why V2

The current package already exposes backend detection and backend metadata, but
the actual `BaseAttentive` model assembly is still largely tied to the current
TensorFlow-oriented component stack.

That means:

- backend selection exists today
- backend execution parity does not
- `torch` and `jax` are still not safe targets for the full `BaseAttentive`
  path

V2 will solve this by introducing a backend-neutral model specification layer
and a backend-aware resolution layer while keeping the current implementation
alive until the new stack is ready.

## V2 Goals

- Support a true multi-backend `BaseAttentive` architecture.
- Keep TensorFlow, Torch, and JAX as first-class runtime targets.
- Separate logical model design from backend-specific implementation details.
- Allow partial rollout of new components without breaking the current package.
- Preserve backward compatibility during the migration period.

## V2 Non-Goals

- Replace the current `components/` package immediately.
- Rewrite everything in one PR.
- Break existing TensorFlow users while the new implementation is incomplete.
- Promise parity for every advanced component on day one.

## Guiding Principles

- Keep the current implementation as the legacy engine.
- Build V2 side-by-side until model-level parity is proven.
- Prefer backend-neutral `keras.ops` and Keras layers before adding backend
  overrides.
- Use backend-specific implementations only when a component truly needs them.
- Serialize logical configuration, not backend-specific classes.
- Fail early with clear capability errors when a backend path is not ready.

## Naming Policy During Migration

- During development on the `v2` branch, the experimental class may keep the
  `BaseAttentiveV2` suffix to avoid ambiguity with the legacy implementation.
- Once the V2 implementation is complete and becomes the main implementation,
  the public class name should return to `BaseAttentive`.
- The `V2` suffix is therefore a migration aid, not the intended final public
  API name.

## Proposed Package Shape

The final names can still be adjusted, but this is the recommended structure.
`resolver/` is preferred over `dispatcher/` because it better reflects the role
of resolving logical specs into backend-specific implementations.

```text
src/base_attentive/
  config/
    schema.py
    defaults.py
    normalize.py
    validate.py

  registry/
    component_registry.py
    model_registry.py
    capabilities.py

  resolver/
    backend_context.py
    component_resolver.py
    model_resolver.py

  implementations/
    generic/
    tensorflow/
    torch/
    jax/

  experimental/
    base_attentive_v2.py
```

## V2 Architecture

V2 should be built around five layers.

### 1. Config Layer

This layer defines backend-neutral model specifications.

Examples:

- `BaseAttentiveSpec`
- `EncoderSpec`
- `AttentionSpec`
- `DecoderSpec`
- `HeadSpec`

Responsibilities:

- normalize user configuration
- validate configuration
- fill defaults
- produce a stable logical model description

### 2. Backend Context Layer

This layer exposes runtime information and capabilities for the active backend.

Examples:

- backend name
- `keras`
- `keras.layers`
- `keras.ops`
- capability flags

Capability examples:

- `supports_memory_attention`
- `supports_scan`
- `supports_jit`
- `supports_dynamic_shape_asserts`

### 3. Registry Layer

This layer maps logical components to available implementations.

Examples:

- logical key: `attention.cross`
- backend key: `tensorflow`, `torch`, `jax`, `generic`
- value: builder/factory plus metadata

Responsibilities:

- register component implementations
- advertise backend support
- support safe fallback to generic implementations
- provide visibility into missing implementations

### 4. Resolver Layer

This layer chooses the correct implementation for each logical component.

Resolution order:

1. backend-specific implementation
2. generic implementation
3. explicit failure with a capability error

Responsibilities:

- resolve component builders
- assemble model graphs from specs
- avoid direct import coupling between the public model and a specific backend

### 5. Implementation Layer

This layer contains the actual component implementations.

The main split should be:

- `implementations/generic/` for backend-neutral Keras code
- `implementations/tensorflow/` for TensorFlow-specific overrides
- `implementations/torch/` for Torch-specific overrides
- `implementations/jax/` for JAX-specific overrides

## Model Strategy

The current `BaseAttentive` remains the legacy implementation.

V2 should start with:

- `experimental/base_attentive_v2.py`

This model should:

- accept the same high-level user configuration where practical
- convert that input to a backend-neutral spec
- use the resolver to assemble the model from registered components
- remain clearly marked as experimental until parity is proven

## Migration Strategy

### Phase 0: Freeze Legacy Behavior

- keep the current `components/` and `core/base_attentive.py`
- treat them as the legacy engine
- avoid risky rewrites inside the legacy path

### Phase 1: Build the V2 Skeleton

- add `config/`
- add `registry/`
- add `resolver/`
- add `experimental/base_attentive_v2.py`
- add tests for registry and resolver behavior

Deliverable:

- a minimal TensorFlow-backed V2 path using the new architecture

### Phase 2: Port Easy Components First

Start with low-risk components:

- activations
- positional encoding
- masks
- simple heads
- normalization helpers

Goal:

- establish registry and resolver conventions
- prove backend-neutral implementations where possible

### Phase 3: Port Medium-Complexity Blocks

- cross attention
- hierarchical attention
- decoder blocks
- encoder assembly helpers

Goal:

- move the main encoder-decoder flow onto the V2 architecture

### Phase 4: Port Hard Components Last

- memory-augmented attention
- dynamic time window
- runtime shape/assert logic that still assumes TensorFlow graph semantics

Goal:

- close feature parity for advanced model variants

### Phase 5: Validate Parity and Migrate

- run shared contract tests across TensorFlow, Torch, and JAX
- compare output shapes and serialization behavior
- compare behavior on representative configs
- flip the public alias only after parity gates pass

## Testing Strategy

Testing must move from backend detection only to backend execution contracts.

We should add:

- config normalization tests
- registry resolution tests
- backend capability tests
- component contract tests
- model smoke tests on TensorFlow, Torch, and JAX
- serialization and reload tests for V2

Minimum shared model contract:

- instantiate model
- run one forward pass
- verify output shape
- verify deterministic config round-trip

## Compatibility Rules

- legacy `BaseAttentive` remains importable during the migration period
- V2 stays experimental until parity is confirmed
- V2 config should serialize logical architecture choices, not backend class
  names
- unsupported backend-component combinations must fail with explicit guidance

## First V2 PR Scope

The first V2 implementation PR should stay intentionally small.

Include:

- `config/` scaffold
- `registry/` scaffold
- `resolver/` scaffold
- `experimental/base_attentive_v2.py`
- one minimal point-forecast model path on TensorFlow
- contract tests for registry and resolver

Do not include:

- a full component rewrite
- a direct replacement of the current public `BaseAttentive`
- full Torch or JAX parity in the first PR

## Success Criteria Before Public Migration

- TensorFlow V2 path is stable
- at least one end-to-end Torch path works
- at least one end-to-end JAX path works
- shared component contracts pass across supported backends
- serialization behavior is documented and tested
- legacy and V2 coexist cleanly

## Immediate Next Step

Create a dedicated `v2` branch and begin the V2 skeleton with the package
scaffolding for:

- `config/`
- `registry/`
- `resolver/`
- `experimental/`

The first coding step after this plan should be to create the V2 skeleton and
land a minimal TensorFlow-backed `BaseAttentiveV2` through the new resolver
path.
