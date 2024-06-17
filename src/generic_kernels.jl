# generic launchers

## single phase

function compute_fun!(A, fun::F, rheology, args) where F<:Function
    ni      = size(A)
    backend = KernelAbstractions.get_backend(A)
    kernel! = compute_fun_kernel!(backend)
    kernel!(A, rheology, args, fun, ndrange = ni) 
    return nothing
end

@kernel function compute_fun_kernel!(A, rheology, args, fun::F) where F<:Function
    I              = @index(Global, Linear)    
    args_ijk       = ntuple_idx(args, I)
    @inbounds A[I] = fun(rheology, args_ijk)
end

## multi-phase

function compute_fun!(A, fun::F, phases, rheology, args) where F<:Function
    ni      = size(A)
    backend = KernelAbstractions.get_backend(A)
    kernel! = compute_fun_kernel!(backend)
    kernel!(A, phases, rheology, args, fun, ndrange = ni) 
    return nothing
end

@kernel function compute_fun_kernel!(A, phases, rheology, args, fun::F) where F<:Function
    I              = @index(Global, Linear)    
    args_ijk       = ntuple_idx(args, I)
    @inbounds A[I] = fun(rheology, phases[I], args_ijk)
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
    compute_wave_velocity!,

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