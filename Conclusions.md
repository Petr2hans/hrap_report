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

## Conclusion (Reconciled)
Based on the experiments' outcomes, it can be concluded that neither method is universally superior. PySINDy is incredibly fast, deterministic, and mathematically rigorous. If the underlying physics is known for having some definite terms (as described in the original PySR paper or like knowing an oscillator will likely involve sine functions). PySINDy isolates the exact coefficients almost instantly. However, it's quite rigid. If the true dynamics fall outside your manually constructed library, it fails. The required careful set up might work as both advantage and disadvantage, for there is a wide toolkit for fine tuning and noise reduction, yet a small change can cause error in discovery (as seen by oscillator experiment). Explicitly stacking higher-order ODEs into coupled first-order systems for numerical integration might be an approach that would enable PySINDy to discover complex PDEs (I still would love to proceed with the heat equation). 

However, PySR is not affected by the need of careful setup: it is suitable for finding equations when underlying physics is completely unknown. Yet it's computationally heavy and its broad search space can occasionally produce equations that fit local points but are globally unstable, completely failing during forward simulation. Ultimately both methods can be utilised, but perform best in different conditions.
