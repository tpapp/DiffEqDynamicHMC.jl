@info "loading packages - DiffEq ecosystem"
using OrdinaryDiffEq, ParameterizedFunctions, RecursiveArrayTools
@info "loading packages - DynamicHMC ecosystem"
using DiffEqDynamicHMC, TransformVariables, DynamicHMC
@info "loading packages - rest"
using  Test, Distributions, LinearAlgebra, Random

@info "starting tests"
Random.seed!(1)

@testset "Lotka-Volterra 1 parameter" begin

    # define model
    f1 = @ode_def LotkaVolterraTest1 begin
        dx = a*x - x*y
        dy = -3*y + x*y
    end a

    # generate data
    p = [1.5]
    u0 = [1.0, 1.0]
    tspan = (0.0, 10.0)
    prob1 = ODEProblem(f1, u0, tspan, p)
    σ = 0.01                         # noise, fixed for now
    t = collect(range(1, 10; length = 10))   # observation times
    sol = solve(prob1, Tsit5())
    randomized = VectorOfArray([(sol(t[i]) + σ * randn(2)) for i in 1:length(t)])
    data = convert(Array, randomized)

    # inference
    bayesian_result = dynamichmc_inference(prob1, t, data, (Normal(1.5, 1), ),
                                           as(Vector, asℝ₊, 1))
    # check
    @test mean(p.parameters[1] for p in bayesian_result.posterior) ≈ p[1] atol = 0.1
end

@testset "Lotka-Volterra 4 parameters" begin
    # define mdel
    f1 = @ode_def LotkaVolterraTest4 begin
        dx = a*x - b*x*y
        dy = -c*y + d*x*y
    end a b c d

    # generate data
    u0 = [1.0,1.0]
    tspan = (0.0,10.0)
    p = [1.5,1.0,3.0,1.0]
    prob1 = ODEProblem(f1,u0,tspan,p)
    sol = solve(prob1,Tsit5())
    t = collect(range(1,stop=10,length=10))
    randomized = VectorOfArray([(sol(t[i]) + .01randn(2)) for i in 1:length(t)])
    data = convert(Array,randomized)

    # inference
    priors = (a = Truncated(Normal(1.5,0.01), 0, 2),
              b = Truncated(Normal(1.0,0.01), 0, 1.5),
              c = Truncated(Normal(3.0,0.01), 0, 4),
              d = Truncated(Normal(1.0, 0.01), 0, 2))

    bayesian_result = dynamichmc_inference(prob1, t, data, priors, as(Vector, asℝ₊, 4))

    # check
    @test norm(mean([p.parameters for p in bayesian_result.posterior]) .- p, Inf) ≤ 0.1
end
