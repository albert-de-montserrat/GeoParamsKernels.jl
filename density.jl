using GeoParams, KernelAbstractions
using Test 

compute_ρ!(A::AbstractArray, args::Vararg{Any, N}) where N = compute_fun!(A, compute_density, args...)

function compute_fun!(A, fun::F, rheology, args) where F<:Function
    ni      = size(A)
    backend = KernelAbstractions.get_backend(A)
    kernel! = compute_fun_kernel!(backend)
    kernel!(A, rheology, args, fun, ndrange = ni) 
    return nothing
end

@kernel function compute_fun_kernel!(A, rheology, args, fun::F) where F<:Function
    I        = @index(Global, Linear)    
    args_ijk = ntuple_idx(args, I)
    A[I]     = fun(rheology, args_ijk)
end

rheology = PT_Density()
ni       = 5,5
ρ        = zeros(ni)
args     = (T=fill(1200, ni...), P=fill(1e9, ni...))
compute_ρ!(ρ, rheology, args)

@test all(x->x ≈ 5695.6, ρ)

function compute_ρg!(ρg, rheology, args)
    ni = size(ρg)
    backend = KernelAbstractions.get_backend(ρg)
    kernel! = compute_ρg_kernel!(backend)
    kernel!(ρg, rheology, args, ndrange = ni) 
    return nothing
end

@kernel function compute_ρg_kernel!(ρg, rheology, args)
    I        = @index(Global, Linear)    
    args_ijk = ntuple_idx(args, I)
    ρg[I]    = compute_density(rheology, args_ijk) * compute_gravity(rheology)
end

rheology = SetMaterialParams(; Density = PT_Density(), Gravity = ConstantGravity())
ρg       = zeros(ni)
compute_ρg!(ρg, rheology, args)

@test all(x->x ≈ 5695.6 * 9.81, ρg)


"""
    compute_ρg!(ρg, phase_ratios, rheology, args)

Calculate the buoyance forces `ρg` for the given GeoParams.jl `rheology` object and correspondent arguments `args`. 
The `phase_ratios` are used to compute the density of the composite rheology.
"""
function compute_ρ!(ρ, phases, rheology, args)
    ni      = size(ρ)
    backend = KernelAbstractions.get_backend(ρ)
    kernel! = compute_ρ_kernel!(backend)
    kernel!(ρ, phases, rheology, args, ndrange = ni) 
    return nothing
end

@kernel function compute_ρ_kernel!(ρ, phases, rheology, args)
    I = @index(Global, Linear)
    args_ijk = ntuple_idx(args, I)
    ρ[I] = compute_density(rheology, phases[I], args_ijk)
end
compute_ρ!(ρ, phases, rheologies, args)

rheologies = (
    SetMaterialParams(;
        Phase   = 1,
        Density = PT_Density(),
    ),
    SetMaterialParams(;
        Phase   = 2,
        Density = PT_Density(),
    ),
)
phases    = ones(Int, ni)
compute_ρ!(ρ, phases, rheologies, args)
@test all(x->x ≈ 5695.6, ρ)

function compute_ρg!(ρg, phases, rheology, args)
    ni = size(ρg)
    backend = KernelAbstractions.get_backend(ρg)
    kernel! = compute_ρg_kernel!(backend)
    kernel!(ρg, phases, rheology, args, ndrange = ni) 
    return nothing
end

@kernel function compute_ρg_kernel!(ρg, phases, rheology, args)
    I        = @index(Global, Linear)    
    args_ijk = ntuple_idx(args, I)
    ρg[I]    = compute_density(rheology, phases[I], args_ijk) * compute_gravity(rheology[1])
end

rheologies = (
    SetMaterialParams(; Phase = 1, Density = PT_Density(), Gravity = ConstantGravity()),
    SetMaterialParams(; Phase = 2, Density = PT_Density(), Gravity = ConstantGravity()),
)
ρg       = zeros(ni)
compute_ρg!(ρg, phases, rheologies, args)

@test all(x->x ≈ 5695.6 * 9.81, ρg)