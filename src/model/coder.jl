export FunctionalCoder, InputsCoder, MLPCoder

"""
    struct FunctionalCoder

A coder that codes to a discretisation for a functional representation.

# Fields
- `disc::Discretisation`: Discretisation for the functional representation.
- `coder`: Coder.
"""
struct FunctionalCoder
    disc::Discretisation
    coder
end

@Flux.functor FunctionalCoder

code(c::FunctionalCoder, xz, z, x; kws...) =
    code(c.coder, xz, z, c.disc(xz, x; kws...); kws...)

function code_track(c::FunctionalCoder, xz, z, x, h; kws...)
    x_disc = c.disc(xz, x; kws...)
    return code_track(c.coder, xz, z, x_disc, vcat(h, [x_disc]); kws...)
end

recode(c::FunctionalCoder, xz, z, h; kws...) =
    recode(c.coder, xz, z, h[2:end]; kws...)

"""
    struct InputsCoder

Code with the target inputs.
"""
struct InputsCoder end

@Flux.functor InputsCoder

code(encoder::InputsCoder, xz, z, x::AA; kws...) = x, x

"""
    struct MLPCoder

Code with an MLP.

# Fields
- `mlp₁`: Pre-pooling MLP.
- `mlp₂`: Post-pooling MLP.
"""
struct MLPCoder
    mlp₁
    mlp₂
end

@Flux.functor MLPCoder

code(encoder::MLPCoder, xz::AA, z::AA, x::AA; kws...) =
    x, encoder.mlp₂(mean(encoder.mlp₁(cat(xz, z, dims=2)), dims=1))

function code(encoder::MLPCoder, xz::Nothing, z::Nothing, x::AA; kws...)
    batch_size = size(x, 3)
    r = zeros_gpu(Float32, 1, encoder.mlp₁.dim_out, batch_size)
    return x, encoder.mlp₂(r)
end
