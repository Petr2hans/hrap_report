# Discover differential equations from synthetic data using sparse and symbolic regression. First report

Papers analysed:
1. Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. Proceedings of the National Academy of Sciences, 113(15), 3932–3937. https://doi.org/10.1073/pnas.1517384113
2. Cranmer, M. (2023). Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl (arXiv:2305.01582). arXiv. https://doi.org/10.48550/arXiv.2305.01582
3. Prokop, B., & Gelens, L. (2024). From biological data to oscillator models using SINDy. iScience, 27(4), 109316. https://doi.org/10.1016/j.isci.2024.109316
4. Schmidt, M., & Lipson, H. (2009). Distilling Free-Form Natural Laws from Experimental Data. Science, 324(5923), 81–85. https://doi.org/10.1126/science.1165893

Based on the findings of said papers, I was able to make several conclusions:
1. Symbolic regression is a powerful method for discovering algebraic equations. Yet, it scarcely used in discovering O/PDEs, for it lacks a toolkit to do so. Schmidt & Lipson suggest a decent workaround: finding derivatives manually and then plugging them into the script. I used this approach when working with PySR.
2. Aforementioned approach has a drawback: manual derivation (through np.gradient() or pysindy.FiniteDifference()) amplifies noise in the original data, for it is only a numerical approximation of the gradient. A solution would be to use pysindy.SmoothedFiniteDifference().
3. Both PySINDy and PySR are prone to overfitting: both algorithms attempt to minimise loss by introducing almost redundant constants for the sake of fitting more data points in the equation. A solution would be to adjust parsimony parameter, although this way one risks losing an important coefficient.
4. PySINDy is highly dependent on an input operator library: the algorithm uses all features given in the library, whereas PySR may neglect a given operator, if it is not applicable to the data. Therefore, quality of equations, discovered through PySINDy is entirely dependent on given features, as seen in Harmonic Oscillator section.
5. Wiener Filter has proven to be an effective method of reducing noise.

# Conclusion

Experimental results establish PySINDy as the fundamentally superior and physically reliable framework for differential equation discovery. 

Although PySR preformed well on in-sample trajectory fits under 10% noise in exponential decay (R² = 0.9851), out-of-sample extrapolation from y₀ = 20 ($R_{OOS}² = -3.0975 ± 7.0881$) revealed overfitting (discovered equation is not physically stable). Conversely, PySINDy succeeded in discovering an equation that remained stable during both in- and out-of-sample runs ($R_{OOS}² = 0.9528$).

PySR similarly broke down on the oscillator due to the severe noise amplification due to numerical second derivatives (θ̈). Conversely, formulating the pendulum as a coupled two-state system ([θ, ω]ᵀ) enabled PySINDy to recover the exact governing physics (θ̇ = 0.999ω, ω̇ = -3.267*sin θ). Furthermore, integrating calibrated Wiener filtering (window = 5) served as an essential regularizer under moderate-to-high noise, reducing PySINDy’s derivative MSE by over 60% (from 0.0122 to 0.0048) and lifting trajectory R² from -0.9784 to +0.5401 ($R_{OOS}² = 0.9652$) at 5% noise, while suppressing enormous divergence at 10% noise ($MSE_{traj} = 302.16$ versus 1.19 × 10⁶ unfiltered). 

Ultimately, both methods have their unique use-cases. PySR offers exploratory utility for completely unknown functional forms. It can be used to discover common terms related to equation's physics area. Simultaneously, PySINDy’s deterministic millisecond runtime and native handling of coupled multi-state vectors make it decisively more useful and accurate for dynamical system identification.

