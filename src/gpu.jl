using CUDA.CUBLAS
using CUDA.CUDNN

import CUDA: libcudnn, @checked, @argout
import CUDA.CUDNN:
    @check, @runtime_ccall, handle,
    cudnnStatus_t, cudnnConvolutionDescriptor_t,
    ConvDesc, TensorDesc, FilterDesc,
    CuArray, CuVector, CUDNNFloat, cdsize,
    cudnnConvolutionForward,
    cudnnGetConvolutionForwardWorkspaceSize, cudnnConvolutionFwdAlgo_t,
    cudnnConvolutionBackwardData,
    cudnnGetConvolutionBackwardDataWorkspaceSize, cudnnConvolutionBwdDataAlgo_t,
    cudnnConvolutionBackwardFilter,
    cudnnGetConvolutionBackwardFilterWorkspaceSize, cudnnConvolutionBwdFilterAlgo_t,
import NNlib: depthwiseconv!, ∇depthwiseconv_filter!, ∇depthwiseconv_data!

const CuOrVector = Union{CuVector, Vector}
const CuOrMatrix = Union{CuMatrix, Matrix}
const CuOrArray = Union{CuArray, Array}

# Implement conversion to dense, diagonal matrix.

diagonal(x::CuArray{T, 1}) where T<:Real = convert(CuArray, Diagonal(x))

# Implement GPU support for depthwise separable convolutions.

@checked function cudnnGetConvolutionGroupCount(convDesc, count)
    @runtime_ccall(
        (:cudnnGetConvolutionGroupCount, libcudnn()),
        cudnnStatus_t,
        (cudnnConvolutionDescriptor_t, Ptr{Cint}),
        convDesc,
        count
    )
end

@checked function cudnnSetConvolutionGroupCount(convDesc, count)
    @runtime_ccall(
        (:cudnnSetConvolutionGroupCount, libcudnn()),
        cudnnStatus_t,
        (cudnnConvolutionDescriptor_t, Cint),
        convDesc,
        count
    )
end

function ConvDesc(T, cdims::DepthwiseConvDims)
    cd = Ref{cudnnConvolutionDescriptor_t}()
    CUDNN.cudnnCreateConvolutionDescriptor(cd)
    N = NNlib.spatial_dims(cdims)
    CUDNN.cudnnSetConvolutionNdDescriptor(
        cd[],
        N,
        # Asymmetric padding is not supported.
        cdsize(NNlib.padding(cdims)[1:2:end], N),
        cdsize(NNlib.stride(cdims), N),
        cdsize(NNlib.dilation(cdims), N),
        NNlib.flipkernel(cdims),
        CUDNN.cudnnDataType(T)
    )
    # Set number of groups equal to number of channels to get a depthwise
    # separable convolution.
    cudnnSetConvolutionGroupCount(cd[], NNlib.channels_in(cdims))
    this = ConvDesc(cd[])
    CUDNN.finalizer(CUDNN.unsafe_free!, this)
    return this
end

function cudnnConvolutionForward(
    y::CuArray{T, N},
    x::CuArray{T, N},
    w::CuArray{T, N},
    cdims::DepthwiseConvDims;
    algo=0,
    alpha=1,
    beta=0
) where {T, N}
    @workspace size = @argout(
        cudnnGetConvolutionForwardWorkspaceSize(
            handle(),
            TensorDesc(x),
            FilterDesc(w),
            ConvDesc(T, cdims),
            TensorDesc(y),
            cudnnConvolutionFwdAlgo_t(algo),
            out(Ref{Csize_t}())
        )
    )[] workspace -> begin
        cudnnConvolutionForward(
            handle(),
            Ref(T(alpha)),
            TensorDesc(x),
            x,
            FilterDesc(w),
            w,
            ConvDesc(T,cdims),
            cudnnConvolutionFwdAlgo_t(algo),
            workspace,
            sizeof(workspace),
            Ref(T(beta)),
            TensorDesc(y),
            y
        )
    end
    return y
end

function cudnnConvolutionBackwardData(
    dx::CuArray{T, N},
    w::CuArray{T, N},
    dy::CuArray{T, N},
    cdims::DepthwiseConvDims;
    algo=0,
    alpha=1,
    beta=0
) where {T, N}
    @workspace size = @argout(
        cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle(),
            FilterDesc(w),
            TensorDesc(dy),
            ConvDesc(T, cdims),
            TensorDesc(dx),
            cudnnConvolutionBwdDataAlgo_t(algo),
            out(Ref{Csize_t}())
        )
    )[] workspace -> begin
        cudnnConvolutionBackwardData(
            handle(),
            Ref(T(alpha)),
            FilterDesc(w),
            w,
            TensorDesc(dy),
            dy,
            ConvDesc(T, cdims),
            cudnnConvolutionBwdDataAlgo_t(algo),
            workspace,
            sizeof(workspace),
            Ref(T(beta)),
            TensorDesc(dx),
            dx
        )
    end
    return dx
    cudnnConvolutionBackwardData(
        Ref(T(alpha)),
        FilterDesc(w), w,
        TensorDesc(dy), dy,
        ConvDesc(T, cdims),
        algo,
        workspace,
        workspace_size,
        Ref(T(beta)),
        TensorDesc(dx), dx
    )
    return dx
end

function cudnnConvolutionBackwardFilter(
    dw::CuArray{T, N},
    x::CuArray{T, N},
    dy::CuArray{T, N},
    cdims::DepthwiseConvDims;
    algo=0,
    alpha=1,
    beta=0
) where {T, N}
    @workspace size = @argout(
        cudnnGetConvolutionBackwardFilterWorkspaceSize(
            handle(),
            TensorDesc(x),
            TensorDesc(dy),
            ConvDesc(T, cdims),
            FilterDesc(dw),
            cudnnConvolutionBwdFilterAlgo_t(algo),
            out(Ref{Csize_t}())
        )
    )[] workspace -> begin
        cudnnConvolutionBackwardFilter(
            handle(),
            Ref(T(alpha)),
            TensorDesc(x),
            x,
            TensorDesc(dy),
            dy,
            ConvDesc(T, cdims),
            cudnnConvolutionBwdFilterAlgo_t(algo),
            workspace,
            sizeof(workspace),
            Ref(T(beta)),
            FilterDesc(dw),
            dw
        )
    end
    return dw
end

function depthwiseconv!(
    y::CuArray{T},
    x::CuArray{T},
    w::CuArray{T},
    cdims::DepthwiseConvDims;
    alpha=1,
    algo=0
) where T<:CUDNNFloat
    cudnnConvolutionForward(
        y,
        x,
        w,
        cdims;
        alpha=alpha,
        algo=algo
    )
    return y
end

function ∇depthwiseconv_filter!(
    dw::CuArray{T},
    x::CuArray{T},
    dy::CuArray{T},
    cdims::DepthwiseConvDims;
    alpha=1,
    algo=0
) where T<:CUDNNFloat
    cudnnConvolutionBackwardFilter(
        dw,
        x,
        dy,
        cdims;
        alpha=alpha,
        algo=algo
    )
    return dw
end

function ∇depthwiseconv_data!(
    dx::CuArray{T},
    dy::CuArray{T},
    w::CuArray{T},
    cdims::DepthwiseConvDims;
    alpha=1,
    algo=0
) where T<:CUDNNFloat
    cudnnConvolutionBackwardData(
        dx,
        w,
        dy,
        cdims;
        alpha=alpha,
        algo=algo
    )
end
