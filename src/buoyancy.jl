"""
    compute_buoyancy!(ρg, rheology, args)

Calculate the buoyance forces `ρg` for the given GeoParams.jl `rheology` object and correspondent arguments `args`.
"""
function compute_buoyancy!(ρg, rheology, args)
    ni = size(ρg)
    backend = KernelAbstractions.get_backend(ρg)
    kernel! = compute_ρg_kernel!(backend, 256)
    kernel!(ρg, rheology, args, ndrange = ni) 
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function compute_ρg_kernel!(ρg, rheology, args)
    I        = @index(Global, Linear)    
    args_ijk = ntuple_idx(args, I)
    ρg[I]    = compute_density(rheology, args_ijk) * compute_gravity(rheology)
end

"""
    compute_buoyancy!(ρg, phase_ratios, rheology, args)

Calculate the buoyance forces `ρg` for the given GeoParams.jl `rheology` object and correspondent arguments `args`. 
"""
function compute_buoyancy!(ρg, phases, rheology, args)
    ni = size(ρg)
    backend = KernelAbstractions.get_backend(ρg)
    kernel! = compute_ρg_kernel!(backend, 256)
    kernel!(ρg, phases, rheology, args, ndrange = ni) 
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function compute_ρg_kernel!(ρg, phases, rheology, args)
    I        = @index(Global, Linear)    
    args_ijk = ntuple_idx(args, I)
    ρg[I]    = compute_density(rheology, phases[I], args_ijk) * compute_gravity(rheology[1])
end