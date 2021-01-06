export code, code_track, recode_stochastic, FunctionalCoder

"""
    code(c, xz, z, x; kws...)

Perform the coding operation specified by `c`.

# Arguments
- `c`: Coder.
- `xz`: Inputs of the functional representation.
- `z`: Outputs of the functional representation.
- `x`: Target inputs.

# Fields
- `kws...`: Further keywords to pass on.

# Returns
- `Tuple`: Tuple containing the inputs and outputs of a functional representation.
"""
code(c, xz, z, x; kws...) = xz, c(z)

# Implement `code` for `Poolings`s to discard the inputs.
code(c::Pooling, xz, z, x; kws...) = nothing, c(z)

function code(c::Chain, xz, z, x; kws...)
    for cᵢ in c
        xz, z = code(cᵢ, xz, z, x; kws...)
    end
    return xz, z
end

function code(p::Parallel, xz, z, x; kws...)
    xz, z = zip([code(pᵢ, xz, z, x; kws...) for pᵢ in p]...)
    return Parallel(xz...), Parallel(z...)
end

function code(p::Parallel{N}, xz, z::Parallel{N}, x; kws...) where N
    xz, z = zip([code(pᵢ, xz, zᵢ, x; kws...) for (pᵢ, zᵢ) in zip(p, z)]...)
    return Parallel(xz...), Parallel(z...)
end

function code(p::Parallel{N}, xz::Parallel{N}, z::Parallel{N}, x; kws...) where N
    xz, z = zip([code(pᵢ, xzᵢ, zᵢ, x; kws...) for (pᵢ, xzᵢ, zᵢ) in zip(p, xz, z)]...)
    return Parallel(xz...), Parallel(z...)
end

"""
    code_track(c, xz, z, x; kws...)

Perform the coding operation specified by `c` whilst keeping track of the sequence of target
inputs, called the history. This history can be used to perform the coding operation again
at that sequence of target inputs exactly.

# Arguments
- `c`: Coder.
- `xz`: Inputs of the functional representation.
- `z`: Outputs of the functional representation.
- `x`: Target inputs.

# Fields
- `kws...`: Further keywords to pass on.

# Returns
- `Tuple`: Tuple containing the inputs and outputs of a functional representation and
    the history of target inputs.
"""
code_track(c, xz, z, x; kws...) = code_track(c, xz, z, x, []; kws...)

function code_track(c, xz, z, x, h; kws...)
    xz, z = code(c, xz, z, x; kws...)
    return xz, z, vcat(h, [x])
end

function code_track(c::Chain, xz, z, x, h; kws...)
    for cᵢ in c
        xz, z, h = code_track(cᵢ, xz, z, x, h; kws...)
    end
    return xz, z, h
end

function code_track(p::Parallel, xz, z, x, h; kws...)
    xz, z, hists = zip([code_track(pᵢ, xz, z, x, []; kws...) for pᵢ in p]...)
    return Parallel(xz...), Parallel(z...), vcat(h, Parallel(hists...))
end

function code_track(p::Parallel{N}, xz, z::Parallel{N}, x, h; kws...) where N
    xz, z, hists = zip([
        code_track(pᵢ, xz, zᵢ, x, []; kws...) for (pᵢ, zᵢ) in zip(p, z)
    ]...)
    return Parallel(xz...), Parallel(z...), vcat(h, Parallel(hists...))
end

function code_track(p::Parallel{N}, xz::Parallel{N}, z::Parallel{N}, x, h; kws...) where N
    xz, z, hists = zip([
        code_track(pᵢ, xzᵢ, zᵢ, x, []; kws...) for (pᵢ, xzᵢ, zᵢ) in zip(p, xz, z)
    ]...)
    return Parallel(xz...), Parallel(z...), vcat(h, Parallel(hists...))
end

"""
    recode(c, xz, z, h; kws...)

Reperform the coding operation specified by `c` at a given sequence of target inputs `c`,
called the history.

# Arguments
- `c`: Coder.
- `xz`: Inputs of the functional representation.
- `z`: Outputs of the functional representation.
- `h`: Sequence of target inputs.

# Returns
- `Tuple`: Tuple containing the inputs and outputs of a functional representation and
    the remaining history of target inputs.
"""
function recode(c, xz, z, h; kws...)
    xz, z = code(c, xz, z, h[1]; kws...)
    return xz, z, h[2:end]
end

function recode(c::Chain, xz, z, h; kws...)
    for cᵢ in c
        xz, z, h = recode(cᵢ, xz, z, h; kws...)
    end
    return xz, z, h
end

function recode(p::Parallel, xz, z, h; kws...)
    xz, z, _ = zip([
        recode(pᵢ, xz, z, hᵢ; kws...) for (pᵢ, hᵢ) in zip(p, h[1])
    ]...)
    return Parallel(xz...), Parallel(z...), h[2:end]
end

function recode(p::Parallel{N}, xz, z::Parallel{N}, h; kws...) where N
    xz, z, _ = zip([
        recode(pᵢ, xz, zᵢ, hᵢ; kws...) for (pᵢ, zᵢ, hᵢ) in zip(p, z, h[1])
    ]...)
    return Parallel(xz...), Parallel(z...), h[2:end]
end

function recode(p::Parallel{N}, xz::Parallel{N}, z::Parallel{N}, h; kws...) where N
    xz, z, _ = zip([
        recode(pᵢ, xzᵢ, zᵢ, hᵢ; kws...) for (pᵢ, xzᵢ, zᵢ, hᵢ) in zip(p, xz, z, h[1])
    ]...)
    return Parallel(xz...), Parallel(z...), h[2:end]
end

"""
    recode_stochastic(
        coders::Parallel{N},
        codings::Parallel{N},
        xc,
        yc,
        h;
        kws...
    ) where N

In an existing aggregate coding `codings`, recode the codings that are not `Dirac`s for
a new context set.

# Arguments
- `coders::Parallel{N}`: Parallel of coders that produced the coding.
- `codings::Parallel{N}`: Parallel of codings.
- `xc`: Locations of new context set.
- `yc`: Observed values of new context set.
- `h`: History to replay.

# Fields
- `kws...`: Further keywords to pass on.

# Returns
- `Parallel`: Updated coding.
"""
recode_stochastic(
    coders::Parallel{N},
    codings::Parallel{N},
    xc,
    yc,
    h;
    kws...
) where N = Parallel([
    recode_stochastic(coder, coding, xc, yc, hᵢ; kws...)
    for (coder, coding, hᵢ) in zip(coders, codings, h[1])
]...)

# Do not recode `Dirac`s.

recode_stochastic(coder, coding::Dirac, xc, yc, h; kws...) = coding

# If the coding is aggregate, it can still contain `Dirac`s, so be careful.

recode_stochastic(coder, coding, xc, yc, h; kws...) =
    _choose(second(recode(coder, xc, yc, h; kws...)), coding)

_choose(new::Parallel{N}, old::Parallel{N}) where N =
    Parallel([_choose(newᵢ, oldᵢ) for (newᵢ, oldᵢ) in zip(new, old)]...)
_choose(new::Dirac, old::Dirac) = old
_choose(new::Gaussian, old::Gaussian) = new

"""
    struct Materialise

A coder that materialises a parallel of things.

# Fields
- `f_x`: Strategy for materialising the inputs. See `materialise`. Defaults to
    `repeat_merge` over the second dimension.
- `f_y`: Strategy for materialising the outputs. See `materialise`. Defaults to
    `repeat_cat` over the second dimension.
"""
struct Materialise
    f_x
    f_y
end

@Flux.functor Materialise

Materialise() = Materialise(
    xs -> repeat_merge(xs..., dims=2),
    xs -> repeat_cat(xs..., dims=2)
)

code(c::Materialise, xz, z, x; kws...) = materialise(xz, c.f_x), materialise(z, c.f_y)
