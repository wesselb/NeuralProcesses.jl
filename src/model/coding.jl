export code, recode_stochastic, FunctionalCoder

"""
    code(c, xz, z, x; kws...)

Perform the coding operation specified by `c`.

# Arguments
- `c`: Coder.
- `xz`: Input of the functional representation.
- `z`: Outputs of the functional representation.
- `x`: Target inputs.

# Returns
- `Tuple`: Tuple containing the inputs and outputs of a functional representation.
"""
code(c, xz, z, x; kws...) = xz, c(z)

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
    recode_stochastic(
        coders::Parallel{N},
        codings::Parallel{N},
        xc,
        yc,
        xt,
        xz::Parallel{N};
        kws...
    ) where N

In an existing aggregate coding `coding`, recode the codings that are not `Dirac`s for
a new context and target set.

# Arguments
- `coders::Parallel{N}`: Parallel of coders that produced the coding.
- `codings::Parallel{N}`: Parallel of codings.
- `xc`: Locations of context set.
- `yc`: Observed values of context set.
- `xt`: Locations of target set.
- `xz::Parallel{N}`: Location of the coding.

# Fields
- `kws...`: Further keywords to pass on.

# Returns
- `Parallel`: Updated coding.
"""
function recode_stochastic(
    coders::Parallel{N},
    codings::Parallel{N},
    xc,
    yc,
    xt,
    xz::Parallel{N};
    kws...
) where N
    xz, z = zip([
        recode_stochastic(coder, coding, xc, yc, xt, xzᵢ; kws...)
        for (xzᵢ, coder, coding) in zip(xz, coders, codings)
    ]...)
    return Parallel(xz...), Parallel(z...)
end

# Do not recode `Dirac`s.

recode_stochastic(coder, coding::Dirac, xc, yc, xt, xz; kws...) = xz, coding

# If the coding is aggregate, it can still contain `Dirac`s, so be careful.

recode_stochastic(coder, coding, xc, yc, xt, xz; kws...) =
    _choose(code(coder, xc, yc, xt; kws...), (xz, coding))

function _choose(
    new::Tuple{Parallel{N}, Parallel{N}},
    old::Tuple{Parallel{N}, Parallel{N}}
) where N
    xz, z = zip([_choose(newᵢ, oldᵢ) for (newᵢ, oldᵢ) in zip(zip(new...), zip(old...))]...)
    return Parallel(xz...), Parallel(z...)
end
_choose(new::Tuple{AA, Dirac}, old::Tuple{AA, Dirac}) = old
_choose(new::Tuple{AA, Normal}, old::Tuple{AA, Normal}) = new


"""
    struct Materialise

A coder that materialises a parallel of things.
"""
struct Materialise end

code(c::Materialise, xz, z, x; kws...) = first(flatten(xz)), materialise(z)

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

@Flux.treelike FunctionalCoder

code(c::FunctionalCoder, xz, z, x; kws...) =
    code(c.coder, xz, z, c.disc(xz, x; kws...); kws...)

# When a functional coder is recoded, the discretisation should not be recomputed, but taken
# from the existing encoding. Moreover, the inputs can be in parallel, e.g. in the case of
# multiple heads. In that case, simply assert that _any_ is valid and take the first.

recode_stochastic(c::FunctionalCoder, xc, yc, xt, xz; kws...) =
    code(c.coder, xc, yc, xz)
recode_stochastic(c::FunctionalCoder, xc, yc, xt, xz::Parallel; kws...) =
    code(c.coder, xc, yc, xt, first(flatten(xz)))
