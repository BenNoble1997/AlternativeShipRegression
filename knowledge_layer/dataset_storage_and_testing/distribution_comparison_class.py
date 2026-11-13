import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from typing import Tuple, Dict


class DistributionComparison:
    """
    A class to compare distributions between conventional and alternative fuel ships.

    This implements various statistical tests to determine if two datasets
    come from the same underlying distribution or have significant differences.
    """

    def __init__(self, conventional_data: pd.DataFrame,
                 alternative_data: pd.DataFrame):
        """
        Initialize with two datasets to compare.

        Parameters:
        -----------
        conventional_data : pd.DataFrame
            Dataset of conventional ships
        alternative_data : pd.DataFrame
            Dataset of alternative fuel ships
        """
        self.conventional_data = conventional_data
        self.alternative_data = alternative_data

    def _get_clean_data(self, feature):
        """
        Helper method to get clean data with null handling.
        Returns cleaned data and validation status.
        """
        alt_data = self.alternative_data[feature].dropna()
        conv_data = self.conventional_data[feature].dropna()

        # Validation
        min_samples = 3  # Minimum for statistical tests

        if len(alt_data) < min_samples or len(conv_data) < min_samples:
            return None, None, {
                'valid': False,
                'reason': f'Insufficient data (alt: {len(alt_data)}, conv: {len(conv_data)})',
                'alt_n': len(alt_data),
                'conv_n': len(conv_data)
            }

        return alt_data, conv_data, {
            'valid': True,
            'alt_n': len(alt_data),
            'conv_n': len(conv_data)
        }

    def bootstrap_difference_ci(self,
                                feature='LOA',
                                n_bootstrap=10000,
                                ci_method:str='percentile',
                                confidence_level=0.95,
                                debug:bool=False,
                                plot:bool=False):
        """
        Bootstrap confidence interval for the difference in medians/means.

        Concept: We want to know "what's the difference in LOA between
        alternative and conventional ships, with uncertainty?"

        This resamples EACH dataset independently to get a distribution
        of the difference.

        :param feature: str the feature that the test is checking the difference of
        :param n_bootstrap: the number of bootstrap samples to use
        :param ci_method: str method for confidence interval. Can be:
            *Percentile - uses percentile method
            *bca - uses Bias and acceleration correction (DiCiccio & Efron)
            *studentised - Bootstraps the studentized statistic
        """
        alt_data, conv_data, status = self._get_clean_data(feature)

        if alt_data is None:
            print(f'insufficient alt data for {feature}')
            return None

        # Store bootstrap differences
        bootstrap_median_diffs = []
        bootstrap_mean_diffs = []

        for i in range(n_bootstrap):
            # Resample each dataset independently with replacement

            alt_resample = np.random.choice(alt_data, size=len(alt_data), replace=True)
            conv_resample = np.random.choice(conv_data, size=len(conv_data), replace=True)

            # Calculate difference in medians and means
            median_diff = np.median(alt_resample) - np.median(conv_resample)
            mean_diff = np.mean(alt_resample) - np.mean(conv_resample)

            bootstrap_median_diffs.append(median_diff)
            bootstrap_mean_diffs.append(mean_diff)

        bootstrap_median_diffs = np.array(bootstrap_median_diffs)
        bootstrap_mean_diffs = np.array(bootstrap_mean_diffs)

        # Observed differences
        observed_median_diff = np.median(alt_data) - np.median(conv_data)
        observed_mean_diff = np.mean(alt_data) - np.mean(conv_data)

        # Calculate confidence intervals (percentile method)

        if ci_method == 'percentile':
            median_lower_percentile, median_upper_percentile = self._percentile_ci(confidence_level=confidence_level)
            mean_lower_percentile = median_lower_percentile
            mean_upper_percentile = median_upper_percentile

        elif ci_method == 'bca':
            median_lower_percentile, median_upper_percentile = self._bca_ci(confidence_level=confidence_level,
                                                                            bootstrap_diffs=bootstrap_median_diffs,
                                                                            observed_diff=observed_median_diff,
                                                                            alt_data=alt_data,
                                                                            conv_data=conv_data,)

            mean_lower_percentile,mean_upper_percentile = self._bca_ci(confidence_level=confidence_level,
                                                                            bootstrap_diffs=bootstrap_mean_diffs,
                                                                            observed_diff=observed_mean_diff,
                                                                            alt_data=alt_data,
                                                                            conv_data=conv_data,
                                                                            statistic='mean')

        else:
            median_lower_percentile, median_upper_percentile = self._percentile_ci(confidence_level=confidence_level)
            mean_lower_percentile = median_lower_percentile
            mean_upper_percentile = median_upper_percentile

        median_ci = (
            np.percentile(bootstrap_median_diffs, median_lower_percentile),
            np.percentile(bootstrap_median_diffs, median_upper_percentile)
        )

        mean_ci = (
            np.percentile(bootstrap_mean_diffs, mean_lower_percentile),
            np.percentile(bootstrap_mean_diffs, mean_upper_percentile)
        )

        # CI doesn't contain 0 → significant difference
        median_significant = median_ci[0] > 0 or median_ci[1] < 0
        mean_significant = mean_ci[0] > 0 or mean_ci[1] < 0

        if debug:
            print(f"Median CI: [{median_ci[0]:.4f}, {median_ci[1]:.4f}]")
            print(f"Median significant: {median_significant}")
            print(f"Mean CI: [{mean_ci[0]:.4f}, {mean_ci[1]:.4f}]")
            print(f"Mean significant: {mean_significant}")

        if plot:

            # Visualization
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Median difference
            axes[0].hist(bootstrap_median_diffs, bins=50, alpha=0.7, edgecolor='black')
            axes[0].axvline(observed_median_diff, color='red', linestyle='--',
                            linewidth=2, label=f'Observed: {observed_median_diff:.2f}')
            axes[0].axvline(median_ci[0], color='green', linestyle=':', linewidth=2)
            axes[0].axvline(median_ci[1], color='green', linestyle=':', linewidth=2)
            axes[0].axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            axes[0].set_xlabel(f'Difference in Median {feature}')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title(
                f'Bootstrap Distribution: Median Difference\n95% CI: [{median_ci[0]:.2f}, {median_ci[1]:.2f}]')
            axes[0].legend()

            # Mean difference
            axes[1].hist(bootstrap_mean_diffs, bins=50, alpha=0.7, edgecolor='black')
            axes[1].axvline(observed_mean_diff, color='red', linestyle='--',
                            linewidth=2, label=f'Observed: {observed_mean_diff:.2f}')
            axes[1].axvline(mean_ci[0], color='green', linestyle=':', linewidth=2)
            axes[1].axvline(mean_ci[1], color='green', linestyle=':', linewidth=2)
            axes[1].axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            axes[1].set_xlabel(f'Difference in Mean {feature}')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(f'Bootstrap Distribution: Mean Difference\n95% CI: [{mean_ci[0]:.2f}, {mean_ci[1]:.2f}]')
            axes[1].legend()

            plt.tight_layout()
            plt.show()

        # Results
        results = {
            'observed_median_diff': observed_median_diff,
            'median_ci': median_ci,
            'median_significant': median_significant,
            'observed_mean_diff': observed_mean_diff,
            'mean_ci': mean_ci,
            'mean_significant': mean_significant,
            'bootstrap_median_diffs': bootstrap_median_diffs,
            'bootstrap_mean_diffs': bootstrap_mean_diffs
        }

        # Interpretation: if CI doesn't contain 0, significant at that confidence level
        print(f"\nBootstrap Results for {feature}:")
        print(f"{'=' * 60}")
        print(f"Median difference: {observed_median_diff:.2f}")
        print(f"  95% CI: [{median_ci[0]:.2f}, {median_ci[1]:.2f}]")
        print(f"  Significant: {'Yes' if median_ci[0] > 0 or median_ci[1] < 0 else 'No'}")
        print(f"\nMean difference: {observed_mean_diff:.2f}")
        print(f"  95% CI: [{mean_ci[0]:.2f}, {mean_ci[1]:.2f}]")
        print(f"  Significant: {'Yes' if mean_ci[0] > 0 or mean_ci[1] < 0 else 'No'}")

        return results

    def _percentile_ci(self,
                       confidence_level:float=0.95):

        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100


        return lower_percentile,upper_percentile

    def _bca_ci(self,
                bootstrap_diffs:list[float],
                observed_diff:float,
                alt_data:np.ndarray,
                conv_data:np.ndarray,
                confidence_level:float=0.95,
                statistic:str='median'):
        """
        Helper method for determing ci using bias and acceleration correction.
        """



        # Calculate bia correction z_0
        p_below = np.mean(bootstrap_diffs < observed_diff)
        z_0 = stats.norm.ppf(p_below)

        # Calculate acceleration 'a' using jackknife
        jackknife_diffs = []

        # Choose statistic function
        stat_func = np.median if statistic == 'median' else np.mean
        n1, n2 = len(alt_data), len(conv_data)

        # Empirical influence values
        L_all = []

        # Influence from alternative ships
        for i in range(n1):
            alt_jack = np.delete(alt_data, i)
            diff_jack = stat_func(alt_jack) - stat_func(conv_data)
            # Empirical influence: (n-1) * (full estimate - jackknife estimate)
            L_i = (n1 - 1) * (observed_diff - diff_jack)
            L_all.append(L_i)

        # Influence from conventional ships
        for i in range(n2):
            conv_jack = np.delete(conv_data, i)
            diff_jack = stat_func(alt_data) - stat_func(conv_jack)
            # Empirical influence: (n-1) * (full estimate - jackknife estimate)
            L_i = (n2 - 1) * (observed_diff - diff_jack)
            L_all.append(L_i)

        # Calculate acceleration from influence values
        L_all = np.array(L_all)
        L_mean = np.mean(L_all)

        numerator = np.sum((L_mean - L_all) ** 3)
        denominator = 6 * (np.sum((L_mean - L_all) ** 2)) ** 1.5
        a = numerator / denominator if denominator != 0 else 0

        alpha = 1 - confidence_level
        z_alpha_lower = stats.norm.ppf(alpha / 2)
        z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

        # Adjusted alpha levels
        percentile_lower = (stats.norm.cdf(z_0 + (z_0 + z_alpha_lower) / (1 - a * (z_0 + z_alpha_lower))))*100
        percentile_upper = (stats.norm.cdf(z_0 + (z_0 + z_alpha_upper) / (1 - a * (z_0 + z_alpha_upper))))*100

        return percentile_lower,percentile_upper

    def kolmogorov_smirnov_test(self, feature: str) -> Dict:
        """
        Perform Kolmogorov-Smirnov test on a single feature.

        Mathematics:
        ------------
        The K-S test compares the empirical CDFs of two samples.
        The test statistic D is:
            D = max|F₁(x) - F₂(x)|
        where F₁ and F₂ are the empirical CDFs.

        The CDF at point x is: F(x) = (number of observations ≤ x) / n

        Parameters:
        -----------
        feature : str
            Column name to test

        Returns:
        --------
        dict : Contains statistic, p-value, and interpretation
        """
        # Extract the feature data from both datasets
        alt_data, conv_data, status = self._get_clean_data(feature)

        # scipy.stats.ks_2samp computes:
        # 1. Sorts both arrays
        # 2. Computes empirical CDF for each (cumulative count / n)
        # 3. Finds maximum absolute difference between CDFs
        # 4. Computes p-value based on K-S distribution
        statistic, p_value = stats.ks_2samp(conv_data, alt_data)

        # Interpretation
        interpretation = (
            f"The maximum difference between CDFs is {statistic:.4f}. \n"
            f"P-value = {p_value:.4f}. "
        )

        if p_value < 0.05:
            interpretation += (
                "The distributions are significantly different (p < 0.05). \n"
                "This means at some point, the cumulative distributions differ substantially."
            )
        else:
            interpretation += (
                "No significant difference detected (p ≥ 0.05). \n"
                "The distributions appear similar."
            )

        return {
            'test': 'Kolmogorov-Smirnov',
            'feature': feature,
            'statistic': statistic,
            'p_value': p_value,
            'interpretation': interpretation,
            'n_conv': len(conv_data),
            'n_alt': len(alt_data)
        }

    def mann_whitney_test(self, feature: str) -> Dict:
        """
        Perform Mann-Whitney U test on a single feature.

        Mathematics:
        ------------
        The Mann-Whitney U test uses ranks instead of raw values.

        Steps:
        1. Combine both samples and rank from smallest (1) to largest (n₁+n₂)
        2. Sum ranks for sample 1: R₁
        3. Calculate U₁ = R₁ - n₁(n₁+1)/2
        4. Calculate U₂ = n₁×n₂ - U₁
        5. Test statistic is min(U₁, U₂)

        Under null hypothesis (same distribution), U follows a known distribution.

        Parameters:
        -----------
        feature : str
            Column name to test

        Returns:
        --------
        dict : Contains statistic, p-value, and interpretation
        """
        alt_data, conv_data, status = self._get_clean_data(feature)

        if alt_data is None:
            print(f'Insufficient data for parameter: {feature}')
            return {
            'test': 'Mann-Whitney U',
            'feature': feature,
            'statistic': 0.0,
            'p_value': 0.0,
            'median_conv': 'None',
            'median_alt': 'None',
            'interpretation': 'Insufficient data for this test'
        }

        # scipy.stats.mannwhitneyu performs:
        # 1. Ranks all observations from both samples combined
        # 2. Computes U statistic from rank sums
        # 3. Uses normal approximation for p-value (with continuity correction)
        # alternative='two-sided' tests if distributions differ in any direction
        statistic, p_value = stats.mannwhitneyu(
            conv_data,
            alt_data,
            alternative='two-sided'
        )

        # Calculate median for interpretation
        median_conv = np.median(conv_data)
        median_alt = np.median(alt_data)

        interpretation = (
            f"Mann-Whitney U statistic = {statistic:.2f}. P-value = {p_value:.4f}. \n"
            f"Median conventional: {median_conv:.2f}, Median alternative: {median_alt:.2f}. "
        )

        if p_value < 0.05:
            if median_alt > median_conv:
                interpretation += (
                    "Alternative fuel ships tend to have significantly larger values \n"
                    "for this feature (p < 0.05)."
                )
            else:
                interpretation += (
                    "Conventional ships tend to have significantly larger values \n"
                    "for this feature (p < 0.05)."
                )
        else:
            interpretation += (
                "No significant difference in central tendency (p ≥ 0.05)."
            )

        return {
            'test': 'Mann-Whitney U',
            'feature': feature,
            'statistic': statistic,
            'p_value': p_value,
            'median_conv': median_conv,
            'median_alt': median_alt,
            'interpretation': interpretation
        }

    def welch_t_test(self, feature: str) -> Dict:
        """
        Perform Welch's t-test on a single feature.

        Mathematics:
        ------------
        Welch's t-test compares means while allowing different variances.

        t = (x̄₁ - x̄₂) / sqrt(s₁²/n₁ + s₂²/n₂)

        where:
        - x̄₁, x̄₂ are sample means
        - s₁², s₂² are sample variances
        - n₁, n₂ are sample sizes

        Degrees of freedom use Welch-Satterthwaite equation:
        df ≈ (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]

        Parameters:
        -----------
        feature : str
            Column name to test

        Returns:
        --------
        dict : Contains statistic, p-value, effect size, and interpretation
        """
        conv_feature = self.conventional_data[feature].dropna()
        alt_feature = self.alternative_data[feature].dropna()

        if alt_feature is None:
            print(f'Insufficient data for parameter: {feature}')
            return {
            'test': 'Welch T-Test',
            'feature': feature,
            'statistic': 0.0,
            'p_value': 0.0,
            'median_conv': 'None',
            'median_alt': 'None',
            'interpretation': 'Insufficient data for this test'
        }

        # scipy.stats.ttest_ind with equal_var=False performs Welch's t-test
        # This computes the t-statistic using the formula above
        # and determines p-value from t-distribution with Welch-Satterthwaite df
        statistic, p_value = stats.ttest_ind(
            conv_feature,
            alt_feature,
            equal_var=False  # This makes it Welch's t-test
        )

        # Calculate descriptive statistics
        mean_conv = np.mean(conv_feature)
        mean_alt = np.mean(alt_feature)
        std_conv = np.std(conv_feature, ddof=1)
        std_alt = np.std(alt_feature, ddof=1)

        # Calculate Cohen's d (effect size)
        # Pooled standard deviation for effect size
        n1, n2 = len(conv_feature), len(alt_feature)
        pooled_std = np.sqrt(((n1 - 1) * std_conv ** 2 + (n2 - 1) * std_alt ** 2) / (n1 + n2 - 2))
        cohens_d = (mean_alt - mean_conv) / pooled_std

        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        interpretation = (
            f"t-statistic = {statistic:.4f}, p-value = {p_value:.4f}. \n"
            f"Mean conventional: {mean_conv:.2f} (±{std_conv:.2f}), \n"
            f"Mean alternative: {mean_alt:.2f} (±{std_alt:.2f}). \n"
            f"Cohen's d = {cohens_d:.3f} ({effect_interpretation} effect). \n"
        )

        if p_value < 0.05:
            interpretation += (
                f"The means are significantly different (p < 0.05). \n"
                f"Alternative fuel ships average {abs(mean_alt - mean_conv):.2f} units \n"
                f"{'higher' if mean_alt > mean_conv else 'lower'}."
            )
        else:
            interpretation += "No significant difference in means (p ≥ 0.05)."

        return {
            'test': "Welch's t-test",
            'feature': feature,
            'statistic': statistic,
            'p_value': p_value,
            'mean_conv': mean_conv,
            'mean_alt': mean_alt,
            'cohens_d': cohens_d,
            'effect_size': effect_interpretation,
            'interpretation': interpretation
        }

    def hotelling_t2_test(self, features: list) -> Dict:
        """
        Perform Hotelling's T² test on multiple features simultaneously.

        Mathematics:
        ------------
        Hotelling's T² is the multivariate generalization of the t-test.

        T² = (n₁×n₂)/(n₁+n₂) × (μ̄₁ - μ̄₂)ᵀ S⁻¹ (μ̄₁ - μ̄₂)

        where:
        - μ̄₁, μ̄₂ are mean vectors for each sample
        - S is the pooled covariance matrix
        - S⁻¹ is its inverse

        The pooled covariance matrix accounts for:
        1. Variances of each variable
        2. Correlations between variables

        F = [(n₁+n₂-p-1) / ((n₁+n₂-2)×p)] × T²
        follows an F-distribution with (p, n₁+n₂-p-1) degrees of freedom.

        Parameters:
        -----------
        features : list
            List of column names to test jointly

        Returns:
        --------
        dict : Contains test statistic, p-value, and interpretation
        """
        # Extract data for specified features, removing any rows with NaN
        conv_data = self.conventional_data[features].dropna()
        alt_data = self.alternative_data[features].dropna()

        n1, p = conv_data.shape
        n2, _ = alt_data.shape

        # Calculate mean vectors
        # μ̄₁ = [mean_length, mean_beam, mean_draft, ...]
        mean1 = conv_data.mean().values
        mean2 = alt_data.mean().values
        mean_diff = mean1 - mean2

        # Calculate pooled covariance matrix
        # S = [(n₁-1)S₁ + (n₂-1)S₂] / (n₁+n₂-2)
        # This weights each sample's covariance by its degrees of freedom
        cov1 = conv_data.cov().values
        cov2 = alt_data.cov().values
        pooled_cov = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)

        # Calculate T² statistic
        # The factor (n₁×n₂)/(n₁+n₂) adjusts for sample sizes
        # The middle term (μ̄₁ - μ̄₂)ᵀ S⁻¹ (μ̄₁ - μ̄₂) is a weighted distance
        # S⁻¹ weights by inverse covariance (Mahalanobis-like distance)
        try:
            pooled_cov_inv = np.linalg.inv(pooled_cov)
            t2_statistic = (n1 * n2) / (n1 + n2) * mean_diff @ pooled_cov_inv @ mean_diff

            # Convert to F-statistic for p-value calculation
            # This transformation makes T² follow a known F-distribution
            f_statistic = ((n1 + n2 - p - 1) / ((n1 + n2 - 2) * p)) * t2_statistic

            # Calculate p-value from F-distribution
            # df1 = p (number of variables)
            # df2 = n₁ + n₂ - p - 1
            df1 = p
            df2 = n1 + n2 - p - 1
            p_value = 1 - stats.f.cdf(f_statistic, df1, df2)

            interpretation = (
                f"Hotelling's T² = {t2_statistic:.4f}, F({df1},{df2}) = {f_statistic:.4f}, \n"
                f"p-value = {p_value:.4f}. \n"
                f"Testing {p} features jointly across {n1} conventional and {n2} alternative ships. "
            )

            if p_value < 0.05:
                interpretation += (
                    "The multivariate means are significantly different (p < 0.05). \n"
                    "The datasets differ in their overall pattern when considering all features together."
                )
            else:
                interpretation += (
                    "No significant multivariate difference detected (p ≥ 0.05). \n"
                    "The overall patterns are similar when considering all features jointly."
                )

            return {
                'test': "Hotelling's T²",
                'features': features,
                'T2_statistic': t2_statistic,
                'F_statistic': f_statistic,
                'df1': df1,
                'df2': df2,
                'p_value': p_value,
                'interpretation': interpretation
            }

        except np.linalg.LinAlgError:
            return {
                'test': "Hotelling's T²",
                'features': features,
                'error': 'Covariance matrix is singular (not invertible). This may occur if features are perfectly correlated or sample size is too small.',
                'interpretation': 'Cannot perform test due to singular covariance matrix.'
            }

    def maximum_mean_discrepancy(self, features: list, kernel_bandwidth: float = 1.0) -> Dict:
        """
        Compute Maximum Mean Discrepancy (MMD) between distributions.

        Mathematics:
        ------------
        MMD measures distance between distributions using kernel embeddings.

        MMD² = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]

        where:
        - x, x' are samples from distribution 1
        - y, y' are samples from distribution 2
        - k(·,·) is a kernel function (we use Gaussian/RBF kernel)

        Gaussian kernel: k(x,y) = exp(-||x-y||² / (2σ²))

        This measures similarity: k=1 if identical, k→0 if very different.

        Intuition:
        - E[k(x,x')] measures average similarity within conventional ships
        - E[k(y,y')] measures average similarity within alternative ships
        - E[k(x,y)] measures average similarity between the two groups
        - If MMD² is large, the groups are more similar to themselves than to each other

        Parameters:
        -----------
        features : list
            List of column names to compare
        kernel_bandwidth : float
            Bandwidth (σ) for Gaussian kernel. Larger = smoother comparison.

        Returns:
        --------
        dict : Contains MMD statistic and interpretation
        """
        # Extract and standardize data
        conv_data = self.conventional_data[features].dropna().values
        alt_data = self.alternative_data[features].dropna().values

        # Standardize features to have mean=0, std=1
        # This ensures all features contribute equally to distance calculations
        all_data = np.vstack([conv_data, alt_data])
        mean = all_data.mean(axis=0)
        std = all_data.std(axis=0)
        conv_data = (conv_data - mean) / (std + 1e-10)
        alt_data = (alt_data - mean) / (std + 1e-10)

        n1 = conv_data.shape[0]
        n2 = alt_data.shape[0]

        def gaussian_kernel(X, Y, bandwidth):
            """
            Compute Gaussian (RBF) kernel matrix between X and Y.

            K[i,j] = exp(-||x_i - y_j||² / (2σ²))

            This uses cdist to efficiently compute all pairwise distances.
            """
            # Compute pairwise squared Euclidean distances
            # ||x-y||² = (x₁-y₁)² + (x₂-y₂)² + ... + (xₚ-yₚ)²
            pairwise_sq_dists = cdist(X, Y, 'sqeuclidean')

            # Apply Gaussian kernel
            return np.exp(-pairwise_sq_dists / (2 * bandwidth ** 2))

        # Compute kernel matrices
        # K_xx: similarities within conventional ships
        K_xx = gaussian_kernel(conv_data, conv_data, kernel_bandwidth)

        # K_yy: similarities within alternative ships
        K_yy = gaussian_kernel(alt_data, alt_data, kernel_bandwidth)

        # K_xy: similarities between conventional and alternative ships
        K_xy = gaussian_kernel(conv_data, alt_data, kernel_bandwidth)

        # Calculate MMD² (unbiased estimator)
        # We exclude diagonal elements in K_xx and K_yy because k(x,x)=1 always
        # This gives an unbiased estimate of the expectations

        # E[k(x,x')] where x ≠ x'
        term1 = (K_xx.sum() - np.trace(K_xx)) / (n1 * (n1 - 1))

        # E[k(y,y')] where y ≠ y'
        term2 = (K_yy.sum() - np.trace(K_yy)) / (n2 * (n2 - 1))

        # -2 × E[k(x,y)]
        term3 = -2 * K_xy.sum() / (n1 * n2)

        # MMD² = term1 + term2 + term3
        mmd_squared = term1 + term2 + term3
        mmd = np.sqrt(max(0, mmd_squared))  # max with 0 to handle numerical issues

        # Interpretation (note: formal p-value requires permutation testing)
        interpretation = (
            f"MMD statistic = {mmd:.6f} (bandwidth σ = {kernel_bandwidth}). \n"
            f"Comparing {len(features)} features across {n1} conventional and {n2} alternative ships. "
        )

        # Rule of thumb: MMD > 0.1 suggests meaningful difference
        # (actual threshold depends on dimensionality and sample size)
        if mmd > 0.1:
            interpretation += (
                "The MMD value suggests the distributions differ. \n"
                "Ships within each fuel type are more similar to each other than to the other fuel type. \n"
                "This indicates distributional differences beyond just mean shifts."
            )
        else:
            interpretation += (
                "The MMD value is relatively small, suggesting similar distributions. \n"
                "Ships from both fuel types show similar patterns of variation."
            )

        return {
            'test': 'Maximum Mean Discrepancy (MMD)',
            'features': features,
            'mmd': mmd,
            'mmd_squared': mmd_squared,
            'kernel_bandwidth': kernel_bandwidth,
            'interpretation': interpretation,
            'note': 'For formal hypothesis testing, use permutation tests (not implemented here)'
        }

    def visualize_distribution_comparison(self, feature: str):
        """
        Create visualizations to compare distributions.

        This creates:
        1. Overlaid histograms
        2. Q-Q plot (quantile-quantile)
        3. Cumulative distribution functions (CDFs)
        """
        conv_feature = self.conventional_data[feature].dropna()
        alt_feature = self.alternative_data[feature].dropna()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 1. Overlaid histograms
        axes[0].hist(conv_feature, bins=30, alpha=0.5, label='Conventional', density=True)
        axes[0].hist(alt_feature, bins=30, alpha=0.5, label='Alternative', density=True)
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel('Density')
        axes[0].set_title('Distribution Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Q-Q plot
        # This plots quantiles of one distribution against the other
        # If distributions are identical, points fall on diagonal line
        conv_sorted = np.sort(conv_feature)
        alt_sorted = np.sort(alt_feature)

        # Interpolate to same length for Q-Q plot
        quantiles = np.linspace(0, 1, min(len(conv_sorted), len(alt_sorted)))
        conv_quantiles = np.quantile(conv_feature, quantiles)
        alt_quantiles = np.quantile(alt_feature, quantiles)

        axes[1].scatter(conv_quantiles, alt_quantiles, alpha=0.5, s=20)

        # Plot diagonal reference line
        min_val = min(conv_quantiles.min(), alt_quantiles.min())
        max_val = max(conv_quantiles.max(), alt_quantiles.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')

        axes[1].set_xlabel(f'Conventional {feature}')
        axes[1].set_ylabel(f'Alternative {feature}')
        axes[1].set_title('Q-Q Plot')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. Empirical CDFs
        # CDF at x = proportion of data ≤ x
        conv_sorted = np.sort(conv_feature)
        alt_sorted = np.sort(alt_feature)

        # Create CDF: for sorted data, CDF[i] = i/n
        conv_cdf = np.arange(1, len(conv_sorted) + 1) / len(conv_sorted)
        alt_cdf = np.arange(1, len(alt_sorted) + 1) / len(alt_sorted)

        axes[2].plot(conv_sorted, conv_cdf, label='Conventional', linewidth=2)
        axes[2].plot(alt_sorted, alt_cdf, label='Alternative', linewidth=2)
        axes[2].set_xlabel(feature)
        axes[2].set_ylabel('Cumulative Probability')
        axes[2].set_title('Empirical CDFs (K-S Test Compares These)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def run_all_tests(self,
                      features: list,
                K_S_test:bool=True,
                M_W_test:bool=True,
                welch_test:bool=True,
                      ) -> pd.DataFrame:
        """
        Run all univariate tests on each feature and return summary.

        Parameters:
        -----------
        features : list
            List of features to test

        Returns:
        --------
        pd.DataFrame : Summary of all test results
        """
        results = []

        for feature in features:

            if K_S_test:
                # K-S test
                ks_result = self.kolmogorov_smirnov_test(feature)
                results.append({
                    'Feature': feature,
                    'Test': 'K-S',
                    'Statistic': ks_result['statistic'],
                    'P-value': ks_result['p_value'],
                    'Significant': ks_result['p_value'] < 0.05
                })

            # Mann-Whitney test
            if M_W_test:

                mw_result = self.mann_whitney_test(feature)
                results.append({
                    'Feature': feature,
                    'Test': 'Mann-Whitney',
                    'Statistic': mw_result['statistic'],
                    'P-value': mw_result['p_value'],
                    'Significant': mw_result['p_value'] < 0.05
                })

            if welch_test:

                # Welch's t-test
                t_result = self.welch_t_test(feature)
                results.append({
                    'Feature': feature,
                    'Test': "Welch's t",
                    'Statistic': t_result['statistic'],
                    'P-value': t_result['p_value'],
                    'Significant': t_result['p_value'] < 0.05,
                    'Cohens_d': t_result['cohens_d']
                })

        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    # Create example datasets (replace with your actual data)
    np.random.seed(42)

    # Simulate conventional ships (slightly smaller on average)
    n_conv = 150
    conventional_ships = pd.DataFrame({
        'Length': np.random.normal(180, 30, n_conv),
        'Beam': np.random.normal(28, 5, n_conv),
        'Draft': np.random.normal(10, 2, n_conv),
        'DWT': np.random.normal(50000, 15000, n_conv)
    })

    # Simulate alternative fuel ships (slightly larger on average)
    n_alt = 80
    alternative_ships = pd.DataFrame({
        'Length': np.random.normal(190, 32, n_alt),
        'Beam': np.random.normal(29.5, 5.2, n_alt),
        'Draft': np.random.normal(10.3, 2.1, n_alt),
        'DWT': np.random.normal(53000, 16000, n_alt)
    })

    # Initialize comparison
    comparison = DistributionComparison(conventional_ships, alternative_ships)

    # Run individual tests
    print("=" * 80)
    print("KOLMOGOROV-SMIRNOV TEST (Length)")
    print("=" * 80)
    ks_result = comparison.kolmogorov_smirnov_test('Length')
    print(ks_result['interpretation'])
    print()

    print("=" * 80)
    print("MANN-WHITNEY U TEST (Length)")
    print("=" * 80)
    mw_result = comparison.mann_whitney_test('Length')
    print(mw_result['interpretation'])
    print()

    print("=" * 80)
    print("WELCH'S T-TEST (Length)")
    print("=" * 80)
    t_result = comparison.welch_t_test('Length')
    print(t_result['interpretation'])
    print()

    print("=" * 80)
    print("HOTELLING'S T² TEST (All Features)")
    print("=" * 80)
    features = ['Length', 'Beam', 'Draft', 'DWT']
    hotelling_result = comparison.hotelling_t2_test(features)
    print(hotelling_result['interpretation'])
    print()

    print("=" * 80)
    print("MAXIMUM MEAN DISCREPANCY (All Features)")
    print("=" * 80)
    mmd_result = comparison.maximum_mean_discrepancy(features, kernel_bandwidth=1.0)
    print(mmd_result['interpretation'])
    print()

    # Run all tests and create summary table
    print("=" * 80)
    print("SUMMARY TABLE (All Univariate Tests)")
    print("=" * 80)
    summary = comparison.run_all_tests(features)
    print(summary.to_string(index=False))

    # Create visualizations
    fig = comparison.visualize_distribution_comparison('Length')
    plt.savefig('distribution_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'distribution_comparison.png'")