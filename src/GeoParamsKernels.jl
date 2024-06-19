module GeoParamsKernels

using GeoParams, KernelAbstractions

include("generic_kernels.jl")
export compute_density!, 
    compute_conductivity!,
    compute_dϕdT!,
    compute_heatcapacity!,
    compute_latent_heat!,
    compute_meltfraction!,
    compute_radioactive_heat!,
    compute_shearheating!,
    compute_wave_velocity!,
    compute_viscosity_τII!,
    compute_viscosity_εII!

include("buoyancy.jl")
export compute_buoyancy!

end # module GeoParamsKernels
