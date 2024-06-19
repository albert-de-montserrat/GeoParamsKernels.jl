# generic launchers

## single phase

function compute_fun!(A, fun::F, rheology, args) where F<:Function
    ni      = size(A)
    backend = KernelAbstractions.get_backend(A)
    kernel! = compute_fun_kernel!(backend, 256)
    kernel!(A, rheology, args, fun, ndrange = ni) 
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function compute_fun_kernel!(A, rheology, args, fun::F) where F<:Function
    I              = @index(Global, Linear)    
    args_ijk       = ntuple_idx(args, I)
    @inbounds A[I] = fun(rheology, args_ijk)
end

function compute_fun_arg1!(A, fun::F, rheology, arg1, args) where F<:Function
    ni      = size(A)
    backend = KernelAbstractions.get_backend(A)
    kernel! = compute_fun_arg1_kernel!(backend, 256)
    kernel!(A, rheology, arg1, args, fun, ndrange = ni) 
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function compute_fun_arg1_kernel!(A, rheology, arg1, args, fun::F) where F<:Function
    I              = @index(Global, Linear)    
    args_ijk       = ntuple_idx(args, I)
    @inbounds A[I] = fun(rheology, arg1, args_ijk)
end

## multi-phase

function compute_fun!(A, fun::F, phases, rheology, args) where F<:Function
    ni      = size(A)
    backend = KernelAbstractions.get_backend(A)
    kernel! = compute_fun_kernel!(backend, 256)
    kernel!(A, phases, rheology, args, fun, ndrange = ni) 
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function compute_fun_kernel!(A, phases, rheology, args, fun::F) where F<:Function
    I              = @index(Global, Linear)    
    args_ijk       = ntuple_idx(args, I)
    @inbounds A[I] = fun(rheology, phases[I], args_ijk)
end

function compute_fun_arg1!(A, fun::F, phases, rheology, arg1, args) where F<:Function
    ni      = size(A)
    backend = KernelAbstractions.get_backend(A)
    kernel! = compute_fun_arg1_kernel!(backend, 256)
    kernel!(A, phases, rheology, arg1, args, fun, ndrange = ni) 
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function compute_fun_arg1_kernel!(A, phases, rheology, arg1, args, fun::F) where F<:Function
    I              = @index(Global, Linear)    
    args_ijk       = ntuple_idx(args, I)
    @inbounds A[I] = fun(rheology, phases[I], arg1, args_ijk)
end


# generate kernels

import GeoParams: compute_density!, 
    compute_conductivity!,
    compute_dϕdT!,
    compute_heatcapacity!,
    compute_latent_heat!,
    compute_meltfraction!,
    compute_radioactive_heat!,
    compute_shearheating!,
    compute_wave_velocity!

fun_inplace = (
    :compute_density!, 
    :compute_conductivity!,
    :compute_dϕdT!,
    :compute_heatcapacity!,
    :compute_latent_heat!,
    :compute_meltfraction!,
    :compute_radioactive_heat!,
    :compute_shearheating!,
    :compute_wave_velocity!,
)
fun_local   = (
    :compute_density,
    :compute_conductivity,
    :compute_dϕdT,
    :compute_heatcapacity,
    :compute_latent_heat,
    :compute_meltfraction,
    :compute_radioactive_heat,
    :compute_shearheating,
    :compute_wave_velocity,
)

for (f, g) in zip(fun_inplace, fun_local)
    @eval function $f(A::AbstractArray, args::Vararg{Any, N}) where N 
        compute_fun!(A, $g, args...)
        return nothing
    end    
end

fun_arg1_inplace = (
    :compute_viscosity_τII!, 
    :compute_viscosity_εII!,
)
fun_arg1_local   = (
    :compute_viscosity_τII,
    :compute_viscosity_εII,
)

for (f, g) in zip(fun_arg1_inplace, fun_arg1_local)
    @eval function $f(A::AbstractArray, args::Vararg{Any, N}) where N 
        compute_fun_arg1!(A, $g, args...)
        return nothing
    end    
end