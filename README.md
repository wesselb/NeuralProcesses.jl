<img src="https://github.com/wesselb/NeuralProcesses.jl/raw/master/loop.gif" width="800px" />

# NeuralProcesses.jl

NeuralProcesses.jl is a framework for composing
[Neural Processes](https://arxiv.org/abs/1807.01622) built on top of
[Flux.jl](https://github.com/FluxML/Flux.jl).

**Important:**
NeuralProcesses.jl requires [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) at
a version newer than v1.2.0.
Currently, this means that `master` is required for
[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) and
[GPUArrays.jl](https://github.com/JuliaGPU/GPUArrays.jl): `]dev CUDA` and
`]dev GPUArrays`.

Contents:

- [Introduction](#introduction)
- [Manual](#manual)
- [State of the Package](#state-of-the-package)
- [Implementation Details](#implementation-details)

## Introduction

### Example: The Convolutional Conditional Neural Process

As an example, below is an implementation of the
[Convolutional Conditional Neural Process](https://openreview.net/forum?id=Skey4eBYPS):

```julia
convcnp = Model(
    FunctionalCoder(
        UniformDiscretisation1D(64f0, 1f0),
        Chain(
            set_conv(1, 2 / 64f0; density=true),
            Deterministic()
        )
    ),
    Chain(
        build_conv(
            1f0,  # Receptive field size
            10,   # Number of layers
            64,   # Number of channels
            points_per_unit =64f0,
            dimensionality  =1,
            num_in_channels =2,  # Account for density channel.
            num_out_channels=2
        ),
        set_conv(2, 2 / 64f0),
        HeterogeneousGaussian()
    )
)
```

We can then make predictions as follows:

```julia
means, lowers, uppers, samples = predict(
    convcnp,
    randn(10),  # Random context inputs
    randn(10),  # Random context outputs
    randn(10)   # Random target inputs
)
```

## Manual

### Models

### Coding

### Building Blocks

This section contains an overview of the building blocks that the package
provides.
For some building blocks, there is constructor function available that can be
used to more easily construct the block.

#### Glue

| Type | Constructor | Description |
| :- | :- | :- |
| `Chain` | | Put things in sequence. |
| `Parallel` | | Put things in parallel. |

####  Basic Blocks

| Type | Constructor | Description |
| :- | :- | :- |
| `BatchedMLP` | `batched_mlp` | Batched MLP. |
| `BatchedConv` | `build_conv` | Batched CNN. |
| `Splitter` | | Split off a given number of channels. |
| `LayerNorm` | | Layer normalisation. |
| `MeanPooling` | | Mean pooling. |
| `SumPooling` | | Sum pooling. |

#### Advanced Blocks

| Type | Constructor | Description |
| :- | :- | :- |
| `Attention` | `attention` | Attentive mechanism. |
| `SetConv` | `set_conv` | Set convolution. |

#### Likelihoods

| Type | Constructor | Description |
| :- | :- | :- |
| `Deterministic` | | Deterministic output. |
| `FixedGaussian` | | Gaussian likelihood with a fixed variance. |
| `AmortisedGaussian` | | Gaussian likelihood with a fixed variance that is calculated from split-off channels. |
| `HeterogeneousGaussian` | | Gaussian likelihood with input-dependent variance. |

#### Coders

| Type | Constructor | Description |
| :- | :- | :- |
| `Materialise` | | Materialise a sample. |
| `FunctionalCoder` | | Code into a function space: make the target inputs a discretisation. |
| `UniformDiscretisation1D` | | Discretise uniformly at a given density of points. |
| `InputsCoder` | | Code with the target inputs. |
| `MLPCoder` | | Rho-sum-phi coder.|


### Data Generators


### Training


## State of the Package


## Implementation Details

### Automatic Differentiation

[Tracker.jl](https://github.com/FluxML/Tracker.jl) is used to automatically
compute gradients.
A number of custom gradients are implemented in `src/util.jl`.

### Parameter Handling

The package uses [Functors.jl](https://github.com/FluxML/Functors.jl) to handle
parameters, like [Flux.jl](https://github.com/FluxML/Flux.jl), and adheres
to [Flux.jl](https://github.com/FluxML/Flux.jl)'s principles.
This means that _only_ `AbstractArray{<:Number}`s are parameters.
Nothing else will be trained, not even `Float32` or `Float64` scalars.

To not train an array, wrap it with `NeuralProcesses.Fixed(array)`,
and unwrap it with `NeuralProcesses.unwrap(fixed)` at runtime.

### GPU Acceleration

CUDA support for depthwise separable convolutions (`DepthwiseConv` from
[Flux.jl](https://github.com/FluxML/Flux.jl)) is implemented in `src/gpu.jl`.

Loop fusion can cause issues on the GPU, so oftentimes computations are
unrolled.
