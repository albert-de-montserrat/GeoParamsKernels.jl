module GeoParamsKernels

using GeoParams, KernelAbstractions

include("generic_kernels.jl")
export compute_density!

include("buoyancy.jl")
export compute_buoyancy!

end # module GeoParamsKernels
