module DiffEqDynamicHMC

export DynamicHMCPosterior, dynamichmc_inference

using OrdinaryDiffEq, ParameterizedFunctions, RecursiveArrayTools, Parameters,
    Distributions, Random, DocStringExtensions, Distributions
using TransformVariables, LogDensityProblems, DynamicHMC
import ForwardDiff              # for the default AD

"""
$(TYPEDEF)

Defines a callable that returns the log density for given parameter values when called with
a `NamedTuple` `(parameters = ..., σ = ...)` where `parameters` is a vector of parameters,
and `σ` is the vector of noise scales.

For a common use case, see [`dynamichmc_inference`](@ref).

# Fields

$(FIELDS)
"""
Base.@kwdef struct DynamicHMCPosterior{TA,TP,TD,TT,TR,TS,TK}
    "Algorithm for the ODE solver."
    algorithm::TA
    "An ODE problem definition (`DiffEqBase.DEProblem`)."
    problem::TP
    "Time values at which the simulated path is compared to `data`."
    t::TT
    "Data, as a matrix with each time value in a column."
    data::TD
    "Priors for parameters, an iterable with the same length as the number of parameters."
    parameter_priors::TR
    """
    Priors for the noise scales (currently the standard deviation of a normal distribution),
    one for each variable.
    """
    σ_priors::TS
    "Keyword arguments passed on the the ODE solver `solve`."
    solve_kwargs::TK
end

function (P::DynamicHMCPosterior)(θ)
    @unpack parameters, σ = θ
    @unpack algorithm, problem, data, t, parameter_priors, σ_priors, solve_kwargs = P
    prob = remake(problem, u0 = convert.(eltype(parameters), problem.u0), p = parameters)
    solution = solve(prob, algorithm; solve_kwargs...)
    any((s.retcode ≠ :Success && s.retcode ≠ :Terminated) for s in solution) && return -Inf
    log_likelihood = sum(sum(logpdf.(Normal.(0.0, σ), solution(t) .- data[:, i]))
                         for (i, t) in enumerate(t))
    log_prior_parameters = mapreduce(logpdf, +, parameter_priors, parameters)
    log_prior_σ = mapreduce(logpdf, +, σ_priors, σ)
    log_likelihood + log_prior_parameters + log_prior_σ
end

"""
$(SIGNATURES)

Run MCMC for an ODE problem. Return a `NamedTuple`, which is similar to the one returned by
`DynamicHMC.mcmc_with_warmup`, with an added field `posterior` which contains a vector of
posterior values (transformed from `ℝⁿ`).

# Arguments

- `problem` is the ODE problem

- `t` is the time values at which the solution is compared to `data`

- `data` is a matrix of data, with one column for each element in `t`

- `parameter_priors` is an iterable with the length of the number of paramers, and is used
  as a prior on it, should have comparable structure.

- `parameter_transformations`: a `TransformVariables` transformation to mapping `ℝⁿ` to the
  vector of valid parameters.

# Keyword arguments

- `rng` is the random number generator used for MCMC. Defaults to the global one.

- `num_samples` is the number of MCMC draws (default: 1000)

- `algorithm` is the algorithm used

- `AD_gradient_kind` is passed on to `LogDensityProblems.ADgradient`, make sure to `import`
  the corresponding library.

- `solve_kwargs` is passed on to `solve`

- `mcmc_kwargs` are passed on as keyword arguments to `DynamicHMC.mcmc_with_warmup`
"""
function dynamichmc_inference(problem::DiffEqBase.DEProblem, t, data,
                              parameter_priors, parameter_transformations;
                              σ_priors = fill(Normal(0, 5), size(data, 1)),
                              rng = Random.GLOBAL_RNG, num_samples = 1000,
                              algorithm = Tsit5(),
                              AD_gradient_kind = Val(:ForwardDiff),
                              solve_kwargs = (), mcmc_kwargs = ())
    P = DynamicHMCPosterior(; algorithm = algorithm, problem = problem, t = t, data = data,
                            parameter_priors = parameter_priors, σ_priors = σ_priors,
                            solve_kwargs = solve_kwargs)
    trans = as((parameters = parameter_transformations,
                σ = as(Vector, asℝ₊, length(σ_priors))))
    ℓ = TransformedLogDensity(trans, P)
    ∇ℓ = LogDensityProblems.ADgradient(AD_gradient_kind, ℓ)
    results = mcmc_with_warmup(rng, ∇ℓ, num_samples; mcmc_kwargs...)
    merge((posterior = transform.(Ref(trans), results.chain), ), results)
end

end # module
