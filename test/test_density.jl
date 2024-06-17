ni       = 5,5
args     = (T=fill(1200, ni...), P=fill(1e9, ni...))

@testset "Density" begin 
    # single phase
    rheology = PT_Density()
    ρ        = zeros(ni)
    compute_density!(ρ, rheology, args)
    @test all(x->x ≈ 5695.6, ρ)
    
    # multi-phase
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
    compute_density!(ρ, phases, rheologies, args)
    @test all(x->x ≈ 5695.6, ρ)
end 

@testset "Buoyancy forces" begin 
    # single phase
    phases   = ones(Int, ni) 
    rheology = SetMaterialParams(; Density = PT_Density(), Gravity = ConstantGravity())
    ρg       = zeros(ni)
    compute_buoyancy!(ρg, rheology, args)
    @test all(x->x ≈ 5695.6 * 9.81, ρg)
    
    # multi-phase
    rheologies = (
        SetMaterialParams(; Phase = 1, Density = PT_Density(), Gravity = ConstantGravity()),
        SetMaterialParams(; Phase = 2, Density = PT_Density(), Gravity = ConstantGravity()),
    )
    ρg       = zeros(ni)
    compute_buoyancy!(ρg, phases, rheologies, args)
    @test all(x->x ≈ 5695.6 * 9.81, ρg)
end
