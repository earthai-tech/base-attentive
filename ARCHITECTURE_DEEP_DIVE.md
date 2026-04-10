# BaseAttentive Architecture Deep Dive

**Last Updated:** April 10, 2026

## Table of Contents

- [Executive Summary](#executive-summary)
- [Overall Architecture](#overall-architecture)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Encoder Component](#encoder-component)
- [Attention Mechanisms](#attention-mechanisms)
- [Decoder Component](#decoder-component)
- [Advanced Features](#advanced-features)
- [Design Decisions](#design-decisions)

---

## Executive Summary

**BaseAttentive** is a sophisticated encoder-decoder architecture built on the Keras framework, designed to handle multivariate time series forecasting with three distinct input types. The architecture combines:

- **Multi-scale LSTM encoders** for capturing hierarchical temporal patterns
- **Attention mechanisms** for learning complex dependencies
- **Variable selection networks** for feature importance learning
- **Flexible decoder stacks** for multi-horizon forecasting
- **Quantile modeling** for uncertainty quantification

The design prioritizes **modularity**, **configurability**, and **production-readiness**.

---

## Overall Architecture

### High-Level Block Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   BaseAttentive Model                            │
│                   (Encoder-Decoder)                              │
└─────────────────────────────────────────────────────────────────┘

                          ▼

        ┌─────────────────────────────────────┐
        │    THREE-STREAM INPUT PROCESSOR     │
        │                                     │
        │  • Static Features → Context        │
        │  • Dynamic Features → History       │
        │  • Future Features → Exogenous      │
        │                                     │
        └────────────────┬────────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────────┐
        │  FEATURE EXTRACTION & SELECTION      │
        │                                      │
        │  Variable Selection Network (VSN)   │
        │  or Dense Processing Layer          │
        │                                      │
        └────────────┬─────────────────────────┘
                     │
        ┌────────────┴─────────────┐
        │                          │
        ▼                          ▼
    ┌────────┐            ┌──────────────┐
    │ Static │            │ Dynamic + F. │
    │Context │            │ Context      │
    └───┬────┘            └──────┬───────┘
        │                        │
        │            ┌───────────┴───────────┐
        │            │                       │
        │            ▼                       ▼
        │    ┌──────────────────┐   ┌─────────────────┐
        │    │ Multi-Scale LSTM │   │ Transformer     │
        │    │ (Hybrid Mode)    │   │ (Pure Attention)│
        │    └────────┬─────────┘   └────────┬────────┘
        │             │                      │
        └─────────────┼──────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │  ATTENTION FUSION LAYER             │
        │                                     │
        │  • Cross-Attention (Q/K from      │
        │    encoder, V from context)       │
        │  • Hierarchical Attention         │
        │  • Memory-Augmented Attention     │
        │                                     │
        └────────────┬────────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────────┐
        │  TIME AGGREGATION                   │
        │                                     │
        │  Last step / Average / Flatten     │
        │                                     │
        └────────────┬────────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────────┐
        │  MULTI-HORIZON DECODER              │
        │                                     │
        │  • Dense projection layers          │
        │  • Per-horizon output heads        │
        │  • Residual connections (opt.)     │
        │                                     │
        └────────────┬────────────────────────┘
                     │
        ┌────────────┴────────────────┐
        │                             │
        ▼                             ▼
    Point Forecast            Quantile Modeling
    (B, Horizon, Dim)        (B, Horizon, Quantiles, Dim)
```

### Component Organization

```
BaseAttentive (keras.Model)
│
├── Input Validation Layer
│   └── validate_model_inputs()
│
├── Feature Processing
│   ├── VariableSelectionNetwork (if use_vsn=True)
│   └── Dense Layers (if use_vsn=False)
│
├── Encoder
│   ├── Static Context Encoder
│   │   └── Dense layers → static embeddings
│   │
│   └── Dynamic Context Encoder (selector by mode)
│       ├── MultiScaleLSTM (hybrid mode)
│       └── TransformerEncoder (transformer mode)
│
├── Attention Stack
│   ├── CrossAttention Layer
│   ├── HierarchicalAttention Layer
│   └── MemoryAugmentedAttention Layer
│
├── Temporal Processing
│   ├── Time aggregation
│   └── Residual connections
│
└── Decoder
    ├── MultiDecoder
    │   └── Per-horizon Dense heads
    │
    └── QuantileDistributionModeling (if quantiles specified)
        └── Per-quantile output projections
```

---

## Data Processing Pipeline

### Input Tensor Shapes

```
static_features
├── Shape: (batch_size, static_dim)
├── Example: (32, 4)
└── Represents: Time-invariant properties

dynamic_features
├── Shape: (batch_size, time_steps, dynamic_dim)
├── Example: (32, 100, 8)
└── Represents: Historical time series

future_features
├── Shape: (batch_size, forecast_horizon, future_dim)
├── Example: (32, 24, 6)
└── Represents: Known future exogenous inputs
```

### Processing Steps

#### Step 1: Input Validation

```
Raw Inputs
    ↓
validate_model_inputs()
    ├─ Check tensor ranks
    ├─ Validate shapes compatibility
    ├─ Ensure dtype consistency
    └─ Convert if necessary
    ↓
Validated Tensors
```

#### Step 2: Embedding & Feature Selection

```
Validated Inputs
    ↓
Variable Selection Network (VSN)
    ├─ Static projection: (B, S_dim) → (B, embed_dim)
    ├─ Dynamic projection: (B, T, D_dim) → (B, T, embed_dim)
    └─ Future projection: (B, H, F_dim) → (B, H, embed_dim)
    ↓
Selected & Embedded Features
```

#### Step 3: Context Encoding

```
Embedded Features
    ├─────────────────────────────┬─────────────────────────────┐
    │                             │                             │
    ▼                             ▼                             ▼
Static Context              Dynamic Context              Future Context
    │                             │                             │
    ├─ Dense layers          ├─ MultiScaleLSTM            ├─ Dense layers
    └─ ReLU activation       ├─ Bidirectional             └─ Linear projection
                             └─ Output: (B, T, enc_dim)       │
                                    └─ Reduce: (B, enc_dim)     │
                                          ↑                     │
                                          └─────────────────────┘
                                                  │
                                        Concatenate Contexts
                                                  ↓
                                        Fused Context Encoding
                                        Shape: (B, 3*enc_dim)
```

---

## Encoder Component

### Architecture Modes

#### 1. Hybrid Mode (Default)

```
Dynamic Features (B, T, D_dim)
    ↓
Embedding Layer
    └─ (B, T, D_dim) → (B, T, embed_dim)
    ↓
Multi-Scale LSTM Stack
    ├─ Scale 1: LSTM (standard resolution)
    ├─ Scale 2: LSTM (2x downsampled)
    ├─ Scale 3: LSTM (4x downsampled)
    └─ [Optional] More scales
    ↓
Aggregate Multi-Scale Outputs
    └─ Concatenate or fuse outputs
    ↓
Result: (B, T, hidden_dim) or (B, hidden_dim)
```

**Advantages:**
- Captures both short-term and long-term patterns
- Lower computational cost than pure attention
- Proven effective for temporal sequences

#### 2. Transformer Mode (Pure Attention)

```
Dynamic Features (B, T, D_dim)
    ↓
Embedding Layer + Positional Encoding
    └─ (B, T, D_dim) → (B, T, embed_dim) + position info
    ↓
Transformer Encoder Stack
    ├─ Layer 1:
    │   ├─ Multi-Head Self-Attention
    │   ├─ Add & Norm
    │   ├─ Feed-Forward Network
    │   └─ Add & Norm
    │
    ├─ Layer 2: [Repeat]
    │
    └─ Layer N: [Repeat]
    ↓
Result: (B, T, hidden_dim)
```

**Advantages:**
- Models full temporal dependencies
- Parallel computation friendly
- Better for very long sequences

### Feature Processing Options

#### Variable Selection Network (VSN)

```
Input Features
    ├─ Dense layers (architecture learning)
    ├─ Softmax gating (feature importance)
    └─ Multiply: features ⊙ importance weights
    ↓
Selected Features: Only important features amplified
Shape: (batch_size, time_steps, embed_dim)
```

#### Dense Processing

```
Input Features
    ├─ Dense layer with ReLU
    └─ Output projection
    ↓
Processed Features: Linear transformation
Shape: (batch_size, time_steps, embed_dim)
```

---

## Attention Mechanisms

### 1. Cross-Attention

**Purpose**: Bridge encoder and decoder by attending to encoded context

```
Decoder Query (Q):        (B, H, hidden_dim)
Encoder Key (K):          (B, T, hidden_dim)
Encoder Value (V):        (B, T, hidden_dim)
    ↓
attention_scores = softmax(Q @ K^T / √d_k)
    ↓
output = attention_scores @ V
    ↓
Result: (B, H, hidden_dim)
```

**Configuration:**
```python
architecture_config = {
    "decoder_attention_stack": ["cross"]
}
```

### 2. Hierarchical Attention

**Purpose**: Multi-level aggregation of temporal patterns

```
Level 1: Attend to recent history       (weeks)
    ↓
Level 2: Attend to seasonal patterns    (months)
    ↓
Level 3: Attend to long-term trends     (years)
    ↓
Fuse all levels
    ↓
Result: Multi-perspective representation
```

**Configuration:**
```python
architecture_config = {
    "decoder_attention_stack": ["hierarchical"]
}
```

### 3. Memory-Augmented Attention

**Purpose**: Learn and store important historical patterns

```
External Memory: (B, memory_size, value_dim)
    ├─ Written to during encoding
    ├─ Read from during decoding
    └─ Updated via attention
    ↓
Attention focuses on:
    1. Recent observations (short-term)
    2. Similar past patterns (long-term)
    ↓
Result: Combines immediate context + historical analogues
```

**Configuration:**
```python
architecture_config = {
    "decoder_attention_stack": ["memory"]
}
```

### Combined Attention Stack

```python
# Use all three attention types in sequence
architecture_config = {
    "decoder_attention_stack": ["cross", "hierarchical", "memory"]
}

# Data flow through stack:
encoder_output
    ↓ (apply Cross-Attention)
intermediate_1
    ↓ (apply Hierarchical-Attention)
intermediate_2
    ↓ (apply Memory-Augmented-Attention)
decoder_input
```

---

## Decoder Component

### Multi-Horizon Decoding

```
Decoder Input: (B, decoder_hidden_dim)
    ├─ Fully connected layer
    └─ Expand to: (B, forecast_horizon, hidden_dim)
    ↓
Per-Horizon Dense Heads
    ├─ Head 1: (B, hidden_dim) → (B, output_dim)
    ├─ Head 2: (B, hidden_dim) → (B, output_dim)
    ...
    └─ Head H: (B, hidden_dim) → (B, output_dim)
    ↓
Stack Outputs: (B, forecast_horizon, output_dim)
    ├─ Soft-constrained by attention
    └─ Independent head noise
```

### Uncertainty Quantification

When quantiles are specified:

```
Base Output: (B, H, output_dim)
    ↓
For each quantile q in [0.1, 0.5, 0.9]:
    ├─ Separate output head
    └─ Learn q-weighted loss (quantile loss)
    ↓
Result: (B, H, num_quantiles, output_dim)
    ├─ [..., 0, ::] → 10th percentile
    ├─ [..., 1, ::] → 50th percentile (median)
    └─ [..., 2, ::] → 90th percentile
```

---

## Advanced Features

### Residual Connections

```
When use_residuals=True:

Input: (B, T, hidden_dim)
    ├─ Attention Layer
    ├─ (output + input) → if shape match
    └─ Result: (B, T, hidden_dim)

Benefits:
    • Reduces vanishing gradient problem
    • Stabilizes training
    • Allows deeper models
```

### Batch Normalization

```
When use_batch_norm=True:

Hidden activations
    ├─ Normalize to mean=0, std=1
    ├─ Learnable scale (γ) and shift (β)
    ├─ μ, σ from batch statistics
    └─ Result: Accelerated training convergence
```

### Dynamic Time Warping (DTW)

**Purpose**: Align sequences with different time scales

```
Reference sequence: x = [1, 2, 3, 4]
Query sequence:     q = [1, 1, 2, 3, 4, 4]
    ↓
DTW finds optimal warping path
    ↓
Alignment: Matches similar patterns despite timing
    └─ Useful for solar/wind patterns shifted by clouds
```

---

## Design Decisions

### 1. Three-Stream Input Processing

**Decision**: Use separate static, dynamic, and future feature streams

**Reasoning:**
- Static features (site properties) don't evolve
- Dynamic features (historical) have temporal structure
- Future features (forecasts) are conditional inputs
- Separate processing prevents information leakage

### 2. Variable Selection Network (VSN)

**Decision**: Learn which features matter at each timestep

**Reasoning:**
- Not all input features are equally important
- Importance varies across time
- Learnable selection improves interpretability
- Particularly valuable with high-dimensional data

### 3. Multi-Scale LSTM in Hybrid Mode

**Decision**: Use 3+ LSTM scales instead of single scale

**Reasoning:**
- Captures hierarchical temporal patterns (minutes → months)
- Lower computation than full attention
- Proven effective for time series
- Balances expressiveness and efficiency

### 4. Flexible Attention Stack

**Decision**: Allow composition of cross, hierarchical, memory attention

**Reasoning:**
- Different attention types solve different problems
- Flexibility for different datasets
- Stack design allows easy extension
- Configuration-driven (no code changes needed)

### 5. Configuration Dictionary Pattern

**Decision**: Use `architecture_config` dict for structural choices

**Reasoning:**
- Separates architecture specification from hyperparameters
- Enables dynamic model creation
- Supports future extensibility
- Cleaner than many boolean flags

---

## Performance Characteristics

### Computational Complexity

| Component | Complexity | Notes |
|-----------|-----------|-------|
| MultiScaleLSTM | O(T × hidden²) | Linear in sequence length |
| Transformer Encoder | O(T² × hidden) | Quadratic in sequence length |
| Cross-Attention | O(T × H × hidden) | T = encoder steps, H = decoder steps |
| Full Model (Hybrid) | O(T × hidden²) | Efficient for long T |
| Full Model (Transformer) | O(T² × hidden) | Better for short T |

### Memory Usage

```
Model with embed_dim=64, forecast_horizon=24:

Activations:        ~10-20 MB per batch
Weights:            ~2-5 MB
Attention matrices: ~1-10 MB (varies by T)

Total per batch:    ~15-35 MB (typical)
```

### Inference Speed

```
Hardware: NVIDIA RTX 3090
Batch size: 32
Sequence length: 100

Hybrid Mode:  ~5-10 ms
Transformer:  ~15-25 ms
With Quantiles: +20% latency
```

---

## Extensibility Points

The architecture supports extension at several points:

### 1. Custom Encoders

```python
# Create custom encoder by subclassing
class CustomEncoder(Layer):
    def call(self, inputs, training=False):
        # Custom encoding logic
        return encoded_features
```

### 2. New Attention Types

```python
# Add to decoder_attention_stack
architecture_config = {
    "decoder_attention_stack": ["cross", "custom_attention"]
}
```

### 3. Custom Loss Functions

```python
# Plug into model compilation
model.compile(
    optimizer='adam',
    loss=CustomQuantileLoss(),
    metrics=['mae']
)
```

### 4. Backend-Specific Optimizations

```python
# Configure backend-specific features
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
# Keras handles rest automatically
```

---

## See Also

- [API Documentation](API_DOCUMENTATION.md)
- [Components Reference](COMPONENTS_REFERENCE.md)
- [Main Architecture Overview](ARCHITECTURE.md)
