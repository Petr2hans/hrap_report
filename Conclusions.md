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

## Conclusion (Review 1)

Both PySR and PySINDy can be utilised for discovering simple ODEs, although PySINDy appears to be more convenient and more suitable for this task. Both approaches are sensitive to noise; therefore, a combination of ps.SmoothedFiniteDifference and Wiener filter improves the results drastically.

The oscillator equation appears to be a failure point for PySR, perhaps, because of the presence second-order derivative there. Double derivation amplifies the noise significantly, hence forcing PySR to fit the data in any way available to the algorithm. It is also proven by my separate attempts of discovering a simple heat equation, which requires both temporal and spatial derivatives. PySR failed in these attempts even with a completely clean dataset purely due to the presence of noise from derivation.

The dependency of these methods on human-inputed operators appears to be their only flaw to me. Since the algorithms are not aware of underlying real-world conditions, misleading operators may result in an equation that fits a particular data sample, but fails to describe the system properly. At the first glance, Physics-Informed Neural Networks (PINNs) seem to be more appropriate in this sense.

## Conclusion (Review 2)

Fixing PySR experiments eliminated failure point of PySR in discovering the oscillator equation. However, this equation is still the most influenced by noise level at least visually. In other experiments performance of both methods is almost equal, if compared visually. Calculated MSE for all experiments proves this as well as it proves the claim that Wiener filter increases the accuracy of the prediction (the loss with filter is reduced by nearly 2 times).

Nevertheless, MSE shows a new breaking point in Logistic growth function. Metric's value reaches 90000 there for both methods – a mathematically impossible number. I presume, there is a code error either in my script or in used packages, although I cannot pinpoint the exact issue.

MSE and graphs both show that PySINDy performs slightly better than PySR in almost all cases, especially combined with noise filtering techniques. Additionally, it is much faster and is deprived of issues with randomness in its outcomes. I believe, PySINDy still remains to be the best option, although PySR seems to be a good alternative.

## Conclusion (Review 3)
Trajecotries' comparison reveals that PySR fails to produce a stable oscillator equation: its outputs cannot be integrated thus deriving a trajectory becomes impossible. Yet, PySR performs better in remaining cases, although by a low margin. A common failure point is a noisy data set for logistic growth equation: both methods fail with very high amount of loss and a negligible difference. Nevertheless, PySR's more accurate results in non-extreme cases suggest that it might be, in fact, slightly better than PyAINDy
