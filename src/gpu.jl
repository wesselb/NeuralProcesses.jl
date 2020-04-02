import CuArrays: @libcudnn
using CuArrays.CUDNN
import CuArrays.CUDNN:
    @check, cudnnStatus_t, cudnnConvolutionDescriptor_t,
    ConvDesc, TensorDesc, FilterDesc,
    CuArray, CuVector, CUDNNFloat, cdsize,
    cudnnConvolutionForward,
    cudnnGetConvolutionForwardWorkspaceSize,
    cudnnConvolutionBackwardData,
    cudnnGetConvolutionBackwardDataWorkspaceSize,
    cudnnConvolutionBackwardFilter,
    cudnnGetConvolutionBackwardFilterWorkspaceSize
import NNlib: depthwiseconv!, ∇depthwiseconv_filter!, ∇depthwiseconv_data!

function cudnnGetConvolutionGroupCount(convDesc, count)
    @check ccall(
		(:cudnnGetConvolutionGroupCount, @libcudnn),
        cudnnStatus_t,
        (cudnnConvolutionDescriptor_t, Ptr{Cint}),
        convDesc,
		count
	)
end

function cudnnSetConvolutionGroupCount(convDesc, count)
    @check ccall(
		(:cudnnSetConvolutionGroupCount, @libcudnn),
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
    CUDNN.finalizer(CUDNN.free, this)
    return this
end

function cudnnConvolutionForward(
    y::CuArray{T, N},
    x::CuArray{T, N},
    w::CuArray{T, N},
    cdims::DepthwiseConvDims;
    algo=0,
    workspace=CU_NULL,
    workspace_size=0,
    alpha=1,
    beta=0
) where {T, N}
    cudnnConvolutionForward(
        Ref(T(alpha)),
        TensorDesc(x), x,
        FilterDesc(w), w,
        ConvDesc(T,cdims),
        algo,
        workspace,
        workspace_size,
        Ref(T(beta)),
        TensorDesc(y), y
    )
    return y
end

function cudnnGetConvolutionForwardWorkspaceSize(
    y::CuArray{T, N},
    x::CuArray{T, N},
    w::CuArray{T, N},
    cdims::DepthwiseConvDims;
    algo=0
) where {T, N}
    workspace_size = Ref{Cint}()
    cudnnGetConvolutionForwardWorkspaceSize(
        TensorDesc(x),
        FilterDesc(w),
        ConvDesc(T, cdims),
        TensorDesc(y),
        algo,
        workspace_size
    )
    return Int(workspace_size[])
end

function cudnnConvolutionBackwardData(
    dx::CuArray{T, N},
    w::CuArray{T, N},
    dy::CuArray{T, N},
    cdims::DepthwiseConvDims;
    algo=0,
    workspace=CU_NULL,
    workspace_size=0,
    alpha=1,
    beta=0
) where {T, N}
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


function cudnnGetConvolutionBackwardDataWorkspaceSize(
    dx::CuArray{T, N},
    w::CuArray{T, N},
    dy::CuArray{T, N},
    cdims::DepthwiseConvDims;
    algo=0
) where {T, N}
    workspace_size = Ref{Cint}()
    cudnnGetConvolutionBackwardDataWorkspaceSize(
        FilterDesc(w),
        TensorDesc(dy),
        ConvDesc(T, cdims),
        TensorDesc(dx),
        algo,
        workspace_size
    )
    return Int(workspace_size[])
end


function cudnnConvolutionBackwardFilter(
    dw::CuArray{T, N},
    x::CuArray{T, N},
    dy::CuArray{T, N},
    cdims::DepthwiseConvDims;
    algo=0,
    workspace=CU_NULL,
    workspace_size=0,
    alpha=1,
    beta=0
) where {T, N}
    cudnnConvolutionBackwardFilter(
        Ref(T(alpha)),
        TensorDesc(x), x,
        TensorDesc(dy), dy,
        ConvDesc(T, cdims),
        algo,
        workspace,
        workspace_size,
        Ref(T(beta)),
        FilterDesc(dw), dw
    )
    return dw
end


function cudnnGetConvolutionBackwardFilterWorkspaceSize(
    dw::CuArray{T, N},
    x::CuArray{T, N},
    dy::CuArray{T, N},
    cdims::DepthwiseConvDims;
    algo=0
) where {T, N}
    workspace_size = Ref{Cint}()
    cudnnGetConvolutionBackwardFilterWorkspaceSize(
        TensorDesc(x),
        TensorDesc(dy),
        ConvDesc(T, cdims),
        FilterDesc(dw),
        algo,
        workspace_size
    )
    return Int(workspace_size[])
end


function depthwiseconv!(
    y::CuArray{T},
    x::CuArray{T},
    w::CuArray{T},
    cdims::DepthwiseConvDims;
    alpha=1,
    algo=0
) where T<:CUDNNFloat
    workspace_size = cudnnGetConvolutionForwardWorkspaceSize(y, x, w, cdims, algo=algo)
    CuVector{UInt8}(undef, workspace_size) do workspace
        cudnnConvolutionForward(
            y,
            x,
            w,
            cdims;
            alpha=alpha,
            algo=algo,
            workspace=workspace,
            workspace_size=workspace_size
        )
    end
end

function ∇depthwiseconv_filter!(
    dw::CuArray{T},
    x::CuArray{T},
    dy::CuArray{T},
    cdims::DepthwiseConvDims;
    alpha=1,
    algo=0
) where T<:CUDNNFloat
    workspace_size =
        cudnnGetConvolutionBackwardFilterWorkspaceSize(dw, x, dy, cdims, algo=algo)
    CuVector{UInt8}(undef, workspace_size) do workspace
        cudnnConvolutionBackwardFilter(
            dw,
            x,
            dy,
            cdims;
            alpha=alpha,
            algo=algo,
            workspace=workspace,
            workspace_size=workspace_size
        )
    end
end

function ∇depthwiseconv_data!(
    dx::CuArray{T},
    dy::CuArray{T},
    w::CuArray{T},
    cdims::DepthwiseConvDims;
    alpha=1,
    algo=0
) where T<:CUDNNFloat
    workspace_size =
        cudnnGetConvolutionBackwardDataWorkspaceSize(dx, w, dy, cdims; algo=algo)
    CuVector{UInt8}(undef, workspace_size) do workspace
        cudnnConvolutionBackwardData(
            dx,
            w,
            dy,
            cdims;
            alpha=alpha,
            algo=algo,
            workspace=workspace,
            workspace_size=workspace_size
        )
    end
end
