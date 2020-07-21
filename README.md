<img src="https://github.com/wesselb/NeuralProcesses.jl/raw/master/loop.gif" width="800px" />

# NeuralProcesses.jl

NeuralProcesses.jl is a framework for building [Neural Processes](https://arxiv.org/abs/1807.01622) built on top of [Flux.jl](https://github.com/FluxML/Flux.jl).

**Important:**
NeuralProcesses.jl requires CUDA.jl at commit `8ce07c8` or later, which is
newer than v1.2.0.

## Example: The Convolutional Conditional Neural Process

As an example, below is an implementation of the [Convolutional Conditional Neural Process](https://openreview.net/forum?id=Skey4eBYPS):

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
