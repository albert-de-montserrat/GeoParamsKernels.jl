ni       = 5,5
args     = (T=fill(1200, ni...), P=fill(1e9, ni...))

@testset "Conductivity" begin 
    # single phase
    rheology = ConstantConductivity()
    A        = zeros(ni)
    compute_conductivity!(A, rheology, args)
    @test all(x->x == 3, A)
    
    # multi-phase
    rheologies = (
        SetMaterialParams(;
            Phase        = 1,
            Conductivity = ConstantConductivity(),
        ),
        SetMaterialParams(;
            Phase        = 2,
            Conductivity = ConstantConductivity(),
        ),
    )
    phases    = ones(Int, ni)
    compute_conductivity!(A, phases, rheologies, args)
    @test all(x->x == 3, A)
end 

@testset "Heat capacity" begin 
    # single phase
    rheology = ConstantHeatCapacity()
    A        = zeros(ni)
    compute_heatcapacity!(A, rheology, args)
    @test all(x->x == 1050, A)
    
    # multi-phase
    rheologies = (
        SetMaterialParams(;
            Phase        = 1,
            HeatCapacity = ConstantHeatCapacity(),
        ),
        SetMaterialParams(;
            Phase        = 2,
            HeatCapacity = ConstantHeatCapacity(),
        ),
    )
    phases    = ones(Int, ni)
    compute_heatcapacity!(A, phases, rheologies, args)
    @test all(x->x == 1050, A)
end 

@testset "Radioactive heat" begin 
    # single phase
    rheology = ConstantRadioactiveHeat()
    A        = zeros(ni)
    compute_radioactive_heat!(A, rheology, args)
    @test all(x->x == 1e-6, A)
    
    # multi-phase
    rheologies = (
        SetMaterialParams(;
            Phase           = 1,
            RadioactiveHeat = ConstantRadioactiveHeat(),
        ),
        SetMaterialParams(;
            Phase           = 2,
            RadioactiveHeat = ConstantRadioactiveHeat(),
        ),
    )
    phases    = ones(Int, ni)
    compute_radioactive_heat!(A, phases, rheologies, args)
    @test all(x->x == 1e-6, A)
end

@testset "Latent heat" begin
    # single phase
    rheology = ConstantLatentHeat()
    A        = zeros(ni)
    compute_latent_heat!(A, rheology, args)
    @test all(x->x == 4e5, A)
    
    # multi-phase
    rheologies = (
        SetMaterialParams(;
            Phase      = 1,
            LatentHeat = ConstantLatentHeat(),
        ),
        SetMaterialParams(;
            Phase      = 2,
            LatentHeat = ConstantLatentHeat(),
        ),
    )
    phases    = ones(Int, ni)
    compute_latent_heat!(A, phases, rheologies, args)
    @test all(x->x == 4e5, A)
end 

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

@testset "Viscosity" begin 
    # single phase
    phases   = ones(Int, ni) 
    rheology = SetMaterialParams(; CompositeRheology = CompositeRheology((LinearViscous(),)))
    A       = zeros(ni)
    compute_viscosity_τII!(A, rheology, 1.0, args)
    @test all(x->x ≈ 1e20, A)
    compute_viscosity_εII!(A, rheology, 1.0, args)
    @test all(x->x ≈ 1e20, A)
    
    # multi-phase
    rheologies = (
        SetMaterialParams(; Phase = 1, CompositeRheology = CompositeRheology((LinearViscous(),))),
        SetMaterialParams(; Phase = 2, CompositeRheology = CompositeRheology((LinearViscous(),))),
    )
    A       .= zeros(ni)
    compute_viscosity_τII!(A, phases, rheologies, 1.0, args)
    @test all(x->x ≈ 1e20, A)
    compute_viscosity_εII!(A, phases, rheologies, 1.0, args)
    @test all(x->x ≈ 1e20, A)
end
