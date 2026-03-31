import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
from knowledge_layer.parametric_regression.piecewise_model_wrapper import _PiecewiseModelWrapper
from knowledge_layer.parametric_regression.multivariable_fractional_polynomials_class import MFPRegressor

class ModelFactory:
    """
    Centralized model creation and prediction class

    """

    # ========================================================================
    # PIECEWISE_ALLOWED_BASE_MODELS
    # ========================================================================

    # Base models that can be made piecewise
    PIECEWISE_ALLOWED_BASE_MODELS = [
        'linear',
        'exponential',
        'exponential-offset',
        'logarithmic',
        'logarithmic-power',
        'power',
        'power-offset'
    ]

    # ========================================================================
    # NON-LINEAR FUNCTIONS
    # ========================================================================

    @staticmethod
    def exponential(x, a, b):
        """Exponential function: y = a * exp(b * x)"""
        return a * np.exp(b * x)

    @staticmethod
    def logarithmic(x, a, b):
        """Logarithmic function: y = a * ln(x) + b"""
        return a * np.log(x) + b

    @staticmethod
    def logarithmic_power(x,a,b):
        """Logarithmic power function: y = a * (ln(x) ** b)"""
        return a * (np.log(x) ** b)

    @staticmethod
    def power(x, a, b):
        """Power function: y = a * x^b"""
        return a * x ** b

    @staticmethod
    def exponential_offset(x, a, b, c):
        """Exponential with offset: y = a * exp(b*x) + c
        Useful for curves with horizontal asymptote"""
        return a * np.exp(b * x) + c

    @staticmethod
    def power_offset(x, a, b, c):
        """Power with offset: y = a * x^b + c
        Use with caution - may have convergence issues"""
        return a * (x ** b) + c

    @staticmethod
    def fit_log_power_multivariate(X, y):
        """Fit power model via log transformation"""
        # Transform to log space
        X_log = np.log(X)
        y_log = np.log(y)

        # Fit linear model
        model = LinearRegression()
        model.fit(X_log, y_log)

        # Extract original parameters
        a = np.exp(model.intercept_)
        b = model.coef_  # [b₁, b₂, ...]

        return a, b

    # ========================================================================
    # MODEL CREATION AND FITTING
    # ========================================================================

    @staticmethod
    def create_and_fit(model_type, X_train, y_train,
                        bins=None, binning_variable=None,
                        alpha=None, cv_folds=5,
                       feature_names:tuple[str]|None=None,):
        """
        Create and fit a model of specified type.

        Parameters
        ----------
        model_type : str
            Model type: 'linear', 'ridge', 'lasso', 'ridge-cv', 'lasso-cv',
            'polynomial-2', 'polynomial-3', 'exponential', 'logarithmic', 'power'
        X_train : array-like, shape (n_samples, n_features)
            Training data
        y_train : array-like, shape (n_samples,)
            Target values
        bins : dict, optional
            Bin specification for piecewise models
            Example: {'DWT': [0, 1000, 5000, 10000, np.inf]}
        binning_variable : str, optional
        Variable to bin on (key in bins dict)
        alpha : float, optional
            Regularization strength for ridge/lasso (default: 1.0 for ridge, 0.1 for lasso)
        cv_folds : int, optional
            Number of CV folds for ridge-cv/lasso-cv (default: 5)

        Returns
        -------
        model : fitted model object or tuple
            For linear/polynomial/ridge/lasso: sklearn model object
            For non-linear: tuple (function_name, parameters)
        preprocessor : object or None
            PolynomialFeatures object for polynomial models, None otherwise

        Examples
        --------
        >>> # Ridge with custom alpha
        >>> model, prep = ModelFactory.create_and_fit('ridge', X, y, alpha=10.0)
        >>>
        >>> # Ridge with automatic alpha selection
        >>> model, prep = ModelFactory.create_and_fit('ridge-cv', X, y, cv_folds=10)
        >>>
        >>> # Lasso with automatic alpha selection
        >>> model, prep = ModelFactory.create_and_fit('lasso-cv', X, y)
        """
        # Clean data first - remove inf/nan
        mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train) |
                 np.isinf(X_train).any(axis=1) | np.isinf(y_train))
        X_train_clean = X_train[mask]
        y_train_clean = y_train[mask]

        if len(X_train_clean) < 3:
            raise ValueError(f"Insufficient clean data: {len(X_train_clean)} samples")

    # ========================================================================
    # Handle piecewise models
    # ========================================================================

        if model_type.endswith('-piecewise'):
            # Extract base model type
            base_model = model_type.replace('-piecewise', '')

            # Validate
            if bins is None or binning_variable is None:
                raise ValueError(
                    f"Piecewise model '{model_type}' requires both 'bins' and "
                    f"'binning_variable' parameters"
                )

            if base_model not in ModelFactory.PIECEWISE_ALLOWED_BASE_MODELS:
                raise ValueError(
                    f"Base model '{base_model}' cannot be made piecewise. "
                    f"Allowed: {ModelFactory.PIECEWISE_ALLOWED_BASE_MODELS}"
                )

            # Create and fit piecewise wrapper
            wrapper = _PiecewiseModelWrapper(
                base_model_type=base_model,
                bins=bins,
                binning_variable=binning_variable,
                min_samples_per_bin=10,
                verbose=False  # Set to True for debugging
            )

            wrapper.fit(X_train, y_train)

            # Return wrapper as "model" (has predict() method)
            return wrapper, None

        # ========================================================================
        # LINEAR AND REGULARIZED LINEAR MODELS
        # ========================================================================

        if model_type == 'linear':
            model = LinearRegression()
            model.fit(X_train_clean, y_train_clean)
            return model, None

        elif model_type == 'ridge':
            # Ridge with fixed alpha
            if alpha is None:
                alpha = 1.0  # Default alpha for Ridge
            model = Ridge(alpha=alpha)
            model.fit(X_train_clean, y_train_clean)
            return model, None

        elif model_type == 'lasso':
            # Lasso with fixed alpha
            if alpha is None:
                alpha = 0.1  # Default alpha for Lasso
            model = Lasso(alpha=alpha, max_iter=5000)  # Increased max_iter for convergence
            model.fit(X_train_clean, y_train_clean)
            return model, None

        elif model_type == 'ridge-cv':
            # Ridge with automatic alpha selection via CV
            alphas = np.logspace(-3, 3, 50)  # Test alphas from 0.001 to 1000
            model = RidgeCV(alphas=alphas, cv=cv_folds)
            model.fit(X_train_clean, y_train_clean)
            # Store the selected alpha for reference
            model.selected_alpha_ = model.alpha_
            return model, None

        elif model_type == 'lasso-cv':
            # Lasso with automatic alpha selection via CV
            alphas = np.logspace(-4, 1, 50)  # Test alphas from 0.0001 to 10
            model = LassoCV(alphas=alphas, cv=cv_folds, max_iter=5000)
            model.fit(X_train_clean, y_train_clean)
            # Store the selected alpha for reference
            model.selected_alpha_ = model.alpha_
            return model, None

        # ========================================================================
        # POLYNOMIAL MODELS (can also be regularized)
        # ========================================================================

        elif model_type.startswith('polynomial-'):
            try:
                degree = int(model_type.split('-')[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid polynomial model type: {model_type}")

            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(X_train_clean)

            model = LinearRegression()
            model.fit(X_train_poly, y_train_clean)
            return model, poly

        elif model_type.startswith('polynomial-') and 'ridge' in model_type:
            # Example: 'polynomial-2-ridge'
            try:
                parts = model_type.split('-')
                degree = int(parts[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid polynomial model type: {model_type}")

            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(X_train_clean)

            if alpha is None:
                alpha = 1.0
            model = Ridge(alpha=alpha)
            model.fit(X_train_poly, y_train_clean)
            return model, poly

        elif model_type.startswith('polynomial-') and 'lasso' in model_type:
            # Example: 'polynomial-2-lasso'
            try:
                parts = model_type.split('-')
                degree = int(parts[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid polynomial model type: {model_type}")

            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(X_train_clean)

            if alpha is None:
                alpha = 0.1
            model = Lasso(alpha=alpha, max_iter=5000)
            model.fit(X_train_poly, y_train_clean)
            return model, poly

        # ========================================================================
        # MFP (new function)
        # ========================================================================

        if model_type =='mfp':
            mfp = MFPRegressor(alpha1=0.05, alpha2=0.05, verbose=False)
            mfp_model = mfp.fit(X_train_clean, y_train_clean,feature_names)

            return mfp_model,None

        # ========================================================================
        # NON-LINEAR MODELS (unchanged from original)
        # ========================================================================

        elif model_type == 'exponential':
            if X_train_clean.shape[1] != 1:
                raise ValueError("Exponential model requires single predictor")

            x_train = X_train_clean.flatten()
            if not np.all(y_train_clean > 0):
                raise ValueError("Exponential model requires all y > 0")

            try:
                popt, _ = curve_fit(
                    ModelFactory.exponential,
                    x_train,
                    y_train_clean,
                    maxfev=5000,
                    bounds=([-np.inf, -10], [np.inf, 10])
                )
                return ('exponential', popt), None
            except (RuntimeError, ValueError) as e:
                raise ValueError(f"Exponential fit failed: {e}")

        elif model_type == 'logarithmic':
            if X_train_clean.shape[1] != 1:
                raise ValueError("Logarithmic model requires single predictor")

            x_train = X_train_clean.flatten()
            if not np.all(x_train > 0):
                raise ValueError("Logarithmic model requires all x > 0")

            popt, _ = curve_fit(ModelFactory.logarithmic, x_train, y_train_clean, maxfev=5000)
            return ('logarithmic', popt), None

        elif model_type == 'exponential-offset':
            if X_train_clean.shape[1] != 1:
                raise ValueError("Exponential model requires single predictor")

            x_train = X_train_clean.flatten()

            try:
                popt, _ = curve_fit(
                    ModelFactory.exponential_offset,
                    x_train,
                    y_train_clean,
                    maxfev=10000,  # Increase iterations
                    bounds=([-np.inf, -10, -np.inf], [np.inf, 10, np.inf])
                )
                return ('exponential-offset', popt), None
            except (RuntimeError, ValueError) as e:
                raise ValueError(f"Exponential-offset fit failed: {e}")

        elif model_type == 'power-log-multivariate':
            # Check constraints
            if not (np.all(X_train_clean > 0) and np.all(y_train_clean > 0)):
                raise ValueError("Power-log model requires all X > 0 and y > 0")

            # Transform to log space
            X_log = np.log(X_train_clean)
            y_log = np.log(y_train_clean)

            # Fit linear model
            model = LinearRegression()
            model.fit(X_log, y_log)

            # Store as tuple: (model_type, log_model)
            return ('power-log-multivariate', model), None

        elif model_type == 'exponential-log-multivariate':
            # y = a · exp(b₁·x₁ + b₂·x₂ + ...)
            # ln(y) = ln(a) + b₁·x₁ + b₂·x₂ + ...

            if not np.all(y_train_clean > 0):
                raise ValueError("Exponential-log model requires all y > 0")

            y_log = np.log(y_train_clean)
            model = LinearRegression()
            model.fit(X_train_clean, y_log)

            return ('exponential-log-multivariate', model), None

        elif model_type == 'power':
            if X_train_clean.shape[1] != 1:
                raise ValueError("Power model requires single predictor")

            x_train = X_train_clean.flatten()
            if not (np.all(x_train > 0) and np.all(y_train_clean > 0)):
                raise ValueError("Power model requires all x > 0 and y > 0")

            try:
                popt, _ = curve_fit(
                    ModelFactory.power,
                    x_train,
                    y_train_clean,
                    maxfev=5000,
                    bounds=([-np.inf, -5], [np.inf, 5])
                )
                return ('power', popt), None
            except (RuntimeError, ValueError) as e:
                raise ValueError(f"Power fit failed: {e}")

        elif model_type == 'power-offset':
            if X_train_clean.shape[1] != 1:
                raise ValueError("Power model requires single predictor")

            x_train = X_train_clean.flatten()
            if not (np.all(x_train > 0) and np.all(y_train_clean > 0)):
                raise ValueError("Power model requires all x > 0 and y > 0")

            try:
                popt, _ = curve_fit(
                    ModelFactory.power_offset,
                    x_train,
                    y_train_clean,
                    maxfev=10000,
                    bounds=([-np.inf, -5, -np.inf], [np.inf, 5, np.inf])
                )
                return ('power-offset', popt), None
            except (RuntimeError, ValueError) as e:
                raise ValueError(f"Power fit failed: {e}")

        elif model_type == 'logarithmic-power':
            if X_train_clean.shape[1] != 1:
                raise ValueError("Logarithmic-power model requires single predictor")

            x_train = X_train_clean.flatten()
            if not np.all(x_train > 0):
                raise ValueError("Logarithmic-power model requires all x > 0")

            try:
                popt, _ = curve_fit(
                    ModelFactory.logarithmic_power,
                    x_train,
                    y_train_clean,
                    maxfev=10000,
                    bounds=([-np.inf, 0.1], [np.inf, 5])  # b > 0 to avoid issues
                )
                return ('logarithmic-power', popt), None
            except (RuntimeError, ValueError) as e:
                raise ValueError(f"Logarithmic-power fit failed: {e}")

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    # ========================================================================
    # PREDICTION
    # ========================================================================

    @staticmethod
    def predict(model, preprocessor, X_test, model_type):
        """
        Make predictions using fitted model.

        Works with all model types including ridge, lasso, ridge-cv, lasso-cv.
        """

        # ========================================================================
        # Handle piecewise models
        # ========================================================================

        if model_type.endswith('-piecewise'):
            # Model is a _PiecewiseModelWrapper instance
            if not isinstance(model, _PiecewiseModelWrapper):
                raise TypeError(
                    f"Expected _PiecewiseModelWrapper for model_type '{model_type}', "
                    f"got {type(model)}"
                )

            # Wrapper has its own predict() method
            return model.predict(X_test)

        if model_type == 'power-log-multivariate':
            _, log_model = model
            X_log = np.log(X_test)
            y_log_pred = log_model.predict(X_log)
            return np.exp(y_log_pred)  # Back-transform

        elif model_type == 'exponential-log-multivariate':
            _, log_model = model
            y_log_pred = log_model.predict(X_test)
            return np.exp(y_log_pred)  # Back-transform

        # ========================================================================
        # Handle non-linear models
        # ========================================================================

        if model_type in ['exponential', 'logarithmic', 'power',
                          'exponential-offset', 'power-offset', 'logarithmic-power']:
            # Non-linear models
            func_name, popt = model
            x_test = X_test.flatten()

            if func_name == 'exponential':
                return ModelFactory.exponential(x_test, *popt)
            elif func_name == 'logarithmic':
                return ModelFactory.logarithmic(x_test, *popt)
            elif func_name == 'power':
                return ModelFactory.power(x_test, *popt)

            elif func_name == 'exponential-offset':
                return ModelFactory.exponential_offset(x_test, *popt)
            elif func_name == 'power-offset':
                return ModelFactory.power_offset(x_test, *popt)

            elif func_name == 'logarithmic-power':
                return ModelFactory.logarithmic_power(x_test, *popt)

            else:
                raise ValueError(f"Unknown non-linear model: {func_name}")

        elif preprocessor is not None:
            # Polynomial models - need to transform first
            X_test_transformed = preprocessor.transform(X_test)
            return model.predict(X_test_transformed)

        else:
            # Linear, Ridge, Lasso models - direct prediction
            return model.predict(X_test)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    @staticmethod
    def get_available_models(include_piecewise=False, include_mfp=False):
        """Return list of available model types."""
        base_models = [
            'linear',
            'ridge',
            'lasso',
            'ridge-cv',
            'lasso-cv',
            'polynomial-2',
            'polynomial-3',
            'polynomial-2-ridge',
            'polynomial-2-lasso',
            'polynomial-3-ridge',
            'polynomial-3-lasso',
            'exponential',
            'exponential-offset',
            'logarithmic',
            'logarithmic-power',
            'power',
            'power-offset',
            'power-log-multivariate',
            'exponential-log-multivariate',


        ]

        piecewise_models = []
        mfp_models = []

        if include_piecewise:

            # Add piecewise variants
            piecewise_models = [
                f"{base}-piecewise"
                for base in ModelFactory.PIECEWISE_ALLOWED_BASE_MODELS
            ]

        if include_mfp:
            mfp_models = ['mfp']

        return base_models + piecewise_models + mfp_models

    @staticmethod
    def get_model_description(model_type):
        """Get human-readable description of model type."""
        descriptions = {
            'linear': 'Linear regression: y = a₀ + a₁x₁ + a₂x₂ + ...',
            'ridge': 'Ridge regression (L2): Linear with coefficient shrinkage',
            'lasso': 'Lasso regression (L1): Linear with feature selection',
            'ridge-cv': 'Ridge with auto alpha selection (CV)',
            'lasso-cv': 'Lasso with auto alpha selection (CV)',
            'polynomial-2': 'Quadratic: y = a₀ + a₁x + a₂x² + a₃x₁x₂ + ...',
            'polynomial-3': 'Cubic: y = a₀ + a₁x + a₂x² + a₃x³ + ...',
            'polynomial-2-ridge': 'Quadratic with Ridge regularization',
            'polynomial-2-lasso': 'Quadratic with Lasso regularization',
            'polynomial-3-ridge': 'Cubic with Ridge regularization',
            'polynomial-3-lasso': 'Cubic with Lasso regularization',
            'exponential': 'Exponential: y = a·exp(bx) [single predictor only]',
            'logarithmic': 'Logarithmic: y = a·ln(x) + b [single predictor only]',
            'power': 'Power: y = a·x^b [single predictor only]',
            'exponential-offset': 'Exponential with offset: y = a·exp(bx) + c',
            'power-offset': 'Power with offset: y = a·x^b + c',
            'logarithmic-power': 'Logarithmic power: y = a·log(x)^b',
            # NEW: Multi-predictor descriptions
            'power-log-multivariate': 'Multiplicative power: y = a·x₁^b₁·x₂^b₂·... [multi-predictor]',
            'exponential-log-multivariate': 'Additive exponential: y = a·exp(b₁x₁+b₂x₂+...) [multi-predictor]',
            'mfp':'Multivariable Fractional Polynomials:Automatic variable selection and transformation'
        }
        return descriptions.get(model_type, 'Unknown model type')

    @staticmethod
    def get_model_coefficients(model, preprocessor, feature_names, model_type):
        """
        Extract and display model coefficients.

        Parameters
        ----------
        model : fitted model
        preprocessor : PolynomialFeatures or None
        feature_names : list of str
            Original feature names
        model_type : str
            Model type

        Returns
        -------
        dict : {feature_name: coefficient}
        """

        # ========================================================================
        # Handle MFP models - delegate to MFP's own representation
        # ========================================================================

        if model_type == 'mfp':
            # model is an MFPRegressor instance
            if not isinstance(model, MFPRegressor):
                raise TypeError(f"Expected MFPRegressor for model_type 'mfp', got {type(model)}")

            # Get transformations from MFP
            transformations = model.get_selected_transformations()

            # Build coefficient dictionary
            coeffs = {'intercept': model.intercept_}

            # Map coefficients to transformed variables
            coef_idx = 0
            for var_idx in sorted(model.transforms_.keys()):
                var_name = model.feature_names_in_[var_idx]
                transform_type, powers = model.transforms_[var_idx]

                if transform_type == 'excluded':
                    coeffs[f'{var_name} (excluded)'] = None
                    continue

                transform_str = transformations[var_name]

                # Determine number of terms
                if transform_type in ['linear', 'fp1']:
                    n_terms = 1
                elif transform_type == 'fp2':
                    n_terms = 2
                else:
                    n_terms = 1

                # Extract coefficients
                if n_terms == 1 and coef_idx < len(model.coef_):
                    coeffs[transform_str] = model.coef_[coef_idx]
                    coef_idx += 1
                elif n_terms == 2 and coef_idx + 1 < len(model.coef_):
                    # For FP2, show both terms
                    if len(powers) == 1:
                        # Repeated power: x^p and x^p·ln(x)
                        power = powers[0]
                        term1_str = model._format_power(power).replace('x', var_name)
                        if power == 0:
                            term2_str = f"ln({var_name})²"
                        else:
                            term2_str = f"{term1_str}·ln({var_name})"
                        coeffs[term1_str] = model.coef_[coef_idx]
                        coeffs[term2_str] = model.coef_[coef_idx + 1]
                    else:
                        # Different powers
                        p1, p2 = powers
                        term1_str = model._format_power(p1).replace('x', var_name)
                        term2_str = model._format_power(p2).replace('x', var_name)
                        coeffs[term1_str] = model.coef_[coef_idx]
                        coeffs[term2_str] = model.coef_[coef_idx + 1]
                    coef_idx += 2

            return coeffs

        # ========================================================================
        # Handle piecewise models
        # ========================================================================

        if model_type.endswith('-piecewise'):
            # model is a _PiecewiseModelWrapper instance
            if not isinstance(model, _PiecewiseModelWrapper):
                raise TypeError(
                    f"Expected _PiecewiseModelWrapper for model_type '{model_type}', "
                    f"got {type(model)}"
                )

            # Get diagnostics
            diag = model.get_bin_diagnostics()

            coeffs = {
                'model_type': 'piecewise',
                'base_model': model.base_model_type,
                'binning_variable': model.binning_variable,
                'n_bins_fitted': model.n_bins_fitted,
                'n_bins_total': model.n_bins_total,
                'bins': {}
            }

            # Get coefficients for each fitted bin
            for bin_idx, (bin_model, bin_preprocessor) in model.bin_models.items():
                bin_label = model.bin_labels[bin_idx]
                n_samples = model.bin_sample_counts[bin_idx]

                # Recursively get coefficients for the base model in this bin
                base_coeffs = ModelFactory.get_model_coefficients(
                    bin_model,
                    bin_preprocessor,
                    feature_names,
                    model.base_model_type
                )

                coeffs['bins'][bin_label] = {
                    'n_samples': n_samples,
                    'coefficients': base_coeffs
                }

            return coeffs

        # ========================================================================
        # Handle all other model types (unchanged from original)
        # ========================================================================

        if model_type in ['exponential', 'logarithmic', 'power',
                          'exponential-offset', 'power-offset', 'logarithmic-power']:
            # Non-linear models
            func_name, popt = model
            return {f'{func_name}_param_{i}': val for i, val in enumerate(popt)}

        elif model_type == 'power-log-multivariate':
            # y = a · x₁^b₁ · x₂^b₂ · ...
            _, log_model = model
            a = np.exp(log_model.intercept_)
            b = log_model.coef_

            coeffs = {'a (scale)': a}
            for i, (name, exp) in enumerate(zip(feature_names, b)):
                coeffs[f'b_{i + 1} ({name} exponent)'] = exp
            return coeffs

        elif model_type == 'exponential-log-multivariate':
            # y = a · exp(b₁x₁ + b₂x₂ + ...)
            _, log_model = model
            a = np.exp(log_model.intercept_)
            b = log_model.coef_

            coeffs = {'a (scale)': a}
            for i, (name, coef) in enumerate(zip(feature_names, b)):
                coeffs[f'b_{i + 1} ({name} coefficient)'] = coef
            return coeffs

        elif preprocessor is not None:
            # Polynomial models
            poly_features = preprocessor.get_feature_names_out(feature_names)
            return dict(zip(poly_features, model.coef_))

        else:
            # Linear, Ridge, Lasso
            return dict(zip(feature_names, model.coef_))

    @staticmethod
    def get_model_summary(model, preprocessor, feature_names, model_type):
        """
        Get complete formatted summary of fitted model.

        This is a new method that provides a unified interface for getting
        detailed model summaries, delegating to model-specific summary methods
        where available.

        Parameters
        ----------
        model : fitted model
            The fitted model object
        preprocessor : object or None
            Preprocessor (e.g., PolynomialFeatures)
        feature_names : list of str
            Original feature names
        model_type : str
            Model type

        Returns
        -------
        str
            Formatted model summary
        """

        # ========================================================================
        # Handle MFP models - delegate to MFP's own summary
        # ========================================================================

        if model_type == 'mfp':
            if not isinstance(model, MFPRegressor):
                raise TypeError(f"Expected MFPRegressor for model_type 'mfp', got {type(model)}")

            # Use MFP's built-in summary method
            return model.get_model_summary()

        # ========================================================================
        # Handle piecewise models
        # ========================================================================

        if model_type.endswith('-piecewise'):
            if not isinstance(model, _PiecewiseModelWrapper):
                raise TypeError(
                    f"Expected _PiecewiseModelWrapper for model_type '{model_type}', "
                    f"got {type(model)}"
                )

            lines = []
            lines.append("=" * 80)
            lines.append(f"PIECEWISE MODEL SUMMARY: {model.base_model_type.upper()}")
            lines.append("=" * 80)
            lines.append(f"\nBinning Variable: {model.binning_variable}")
            lines.append(f"Bins Fitted: {model.n_bins_fitted}/{model.n_bins_total}")

            if model.skipped_bins:
                lines.append(f"Skipped Bins: {model.skipped_bins} (insufficient samples)")

            lines.append("\n" + "-" * 80)
            lines.append("BIN DETAILS:")
            lines.append("-" * 80)

            # Show each bin
            for bin_idx in range(model.n_bins_total):
                bin_label = model.bin_labels[bin_idx]
                n_samples = model.bin_sample_counts.get(bin_idx, 0)

                if bin_idx in model.bin_models:
                    lines.append(f"\nBin {bin_idx}: {bin_label}")
                    lines.append(f"  Samples: {n_samples}")

                    # Get base model coefficients
                    bin_model, bin_preprocessor = model.bin_models[bin_idx]
                    base_coeffs = ModelFactory.get_model_coefficients(
                        bin_model, bin_preprocessor, feature_names, model.base_model_type
                    )

                    lines.append(f"  Model: {model.base_model_type}")
                    lines.append("  Coefficients:")
                    for coef_name, coef_value in base_coeffs.items():
                        if coef_value is not None:
                            lines.append(f"    {coef_name}: {coef_value:.6f}")
                else:
                    lines.append(f"\nBin {bin_idx}: {bin_label}")
                    lines.append(f"  SKIPPED (only {n_samples} samples)")

            # Add discontinuity analysis if possible
            try:
                discontinuities = model.calculate_boundary_discontinuities()
                if discontinuities:
                    lines.append("\n" + "-" * 80)
                    lines.append("BOUNDARY DISCONTINUITIES:")
                    lines.append("-" * 80)
                    for boundary_idx, info in discontinuities.items():
                        lines.append(
                            f"\nBoundary {boundary_idx} (at {model.binning_variable} = {info['boundary_value']:.2f}):")
                        lines.append(f"  Prediction from left bin:  {info['pred_left']:.4f}")
                        lines.append(f"  Prediction from right bin: {info['pred_right']:.4f}")
                        lines.append(f"  Absolute jump: {info['absolute_jump']:.4f}")
                        lines.append(f"  Relative jump: {info['relative_jump_pct']:.2f}%")
            except Exception:
                pass  # Skip discontinuity analysis if it fails

            lines.append("\n" + "=" * 80)
            return "\n".join(lines)

        # ========================================================================
        # Handle all other model types - build generic summary
        # ========================================================================

        lines = []
        lines.append("=" * 80)
        lines.append(f"MODEL SUMMARY: {model_type.upper()}")
        lines.append("=" * 80)

        # Get model description
        desc = ModelFactory.get_model_description(model_type)
        lines.append(f"\nModel Type: {desc}")

        # Get coefficients
        coeffs = ModelFactory.get_model_coefficients(model, preprocessor, feature_names, model_type)

        lines.append("\nFitted Parameters:")
        lines.append("-" * 80)

        for name, value in coeffs.items():
            if value is not None:
                if isinstance(value, (int, float)):
                    lines.append(f"  {name:40s}: {value:12.6f}")
                else:
                    lines.append(f"  {name:40s}: {value}")

        # Add model-specific info
        if hasattr(model, 'intercept_'):
            lines.append(f"\n  {'Intercept':40s}: {model.intercept_:12.6f}")

        if hasattr(model, 'selected_alpha_'):
            lines.append(f"\n  {'Selected Alpha (CV)':40s}: {model.selected_alpha_:12.6f}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)
    # ========================================================================
    # MODEL APPLICABILITY CHECKING
    # ========================================================================

    @staticmethod
    def _clean_data_for_check(X, y):
        """
        Clean data by removing NaN and Inf values.

        This mirrors the cleaning done in create_and_fit() so that
        applicability checks are performed on the same data that
        would actually be used for fitting.
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) |
                 np.isinf(X).any(axis=1) | np.isinf(y))
        X_clean = X[mask]
        y_clean = y[mask]

        return X_clean, y_clean

    # @staticmethod
    # def _estimate_polynomial_features(n_predictors, degree):
    #     """
    #     Calculate number of features in polynomial expansion.
    #
    #     Uses combinatorial formula: sum of C(n+d-1, d) for d from 0 to degree
    #
    #     Parameters
    #     ----------
    #     n_predictors : int
    #         Number of input features
    #     degree : int
    #         Polynomial degree
    #
    #     Returns
    #     -------
    #     int
    #         Number of polynomial features
    #
    #     """
    #     from math import comb
    #     return sum(comb(n_predictors + d - 1, d) for d in range(degree + 1))

    @staticmethod
    def get_actual_n_features(model_type, X):
        """
        Determine actual number of features after transformation.

        Parameters
        ----------
        X : array-like
            Input features
        model_type : str
            Model type

        Returns
        -------
        int
            Actual number of features
        """

    # ========================================================================
    # Handle piecewise models
    # ========================================================================

        if model_type.endswith('-piecewise'):
            # X is actually the fitted wrapper in this context
            # (when called from metrics calculation)
            if isinstance(X, _PiecewiseModelWrapper):
                return X.get_n_parameters()

            # If X is actual data, we need the wrapper
            # This case shouldn't happen in normal flow, but handle gracefully
            base_model = model_type.replace('-piecewise', '')
            # Return base model params as approximation
            # (actual calculation requires fitted wrapper)
            base_params = ModelFactory.get_actual_n_features(base_model, X)
            return base_params  # Will be corrected when wrapper is available


        n_input_features = X.shape[1]

        if model_type.startswith('polynomial-'):
            # Extract degree
            try:
                degree = int(model_type.split('-')[1])
            except (IndexError, ValueError):
                return n_input_features

            # Calculate number of polynomial features
            # Formula: (n + d)! / (n! * d!) where n=features, d=degree
            from math import comb
            actual_features = sum(comb(n_input_features + i - 1, i)
                                  for i in range(degree + 1))
            return actual_features

        elif model_type in ['exponential', 'logarithmic', 'power',
                            'exponential-offset', 'power-offset', 'logarithmic-power']:
            # Non-linear models have fixed parameter counts
            if model_type in ['exponential', 'logarithmic', 'power']:
                return 2  # (a, b)
            else:
                return 3  # (a, b, c) for offset models

        # NEW: Multi-predictor non-linear models
        elif model_type == 'power-log-multivariate':
        # y = a · x₁^b₁ · x₂^b₂ · ...
        # Parameters: a (intercept) + b₁, b₂, ... (one per predictor)
            return n_input_features + 1

        elif model_type == 'exponential-log-multivariate':
            # y = a · exp(b₁x₁ + b₂x₂ + ...)
            # Parameters: a (intercept) + b₁, b₂, ... (one per predictor)
            return n_input_features + 1

        elif model_type == 'mfp':
            n_input_features = X.shape[1] if len(X.shape) > 1 else 1
            return 2 * n_input_features + 1

        else:
            # Linear, ridge, lasso
            return n_input_features

    @staticmethod
    def is_model_applicable(model_type, X, y, clean_first=True, min_samples=5):
        """
        Check if a model type can be applied to given data.

        This method checks all mathematical and structural constraints
        required by each model type, including:
        - Number of predictors (some models require single predictor)
        - Value constraints (positivity requirements)
        - Sample size requirements

        Parameters
        ----------
        model_type : str
            Model type to check (e.g., 'linear', 'polynomial-2', 'power')
        X : array-like, shape (n_samples, n_features)
            Predictor data
        y : array-like, shape (n_samples,)
            Target data
        clean_first : bool, default=True
            Whether to clean data (remove NaN/Inf) before checking.
            Should match behavior of create_and_fit().
        min_samples : int, default=5
            Minimum number of samples required (after cleaning)

        Returns
        -------
        tuple : (bool, str)
            - is_applicable : Whether the model can be used
            - reason : Explanation if not applicable (empty string if applicable)


        """

        # Handle unknown model types
        available_models = ModelFactory.get_available_models(include_piecewise=True)
        if model_type not in available_models:
            return False, f"unknown model type '{model_type}'"

        # ========================================================================
        # Handle piecewise models
        # ========================================================================


        if model_type.endswith('-piecewise'):
            # Check if base model is applicable
            base_model = model_type.replace('-piecewise', '')

            # Piecewise models currently only support single predictor
            if len(X.shape) > 1 and X.shape[1] > 1:
                return False, "piecewise models currently require single predictor"

            # Check base model applicability
            base_applicable, base_reason = ModelFactory.is_model_applicable(
                base_model, X, y, clean_first, min_samples
            )

            if not base_applicable:
                return False, f"base model '{base_model}' not applicable: {base_reason}"

            return True, ""

        # ========================================================================
        # Handle other models
        # ========================================================================

        # Clean data if requested
        if clean_first:
            X_check, y_check = ModelFactory._clean_data_for_check(X, y)
        else:
            X_check, y_check = X.copy(), y.copy()
            if len(X_check.shape) == 1:
                X_check = X_check.reshape(-1, 1)

        n_samples = len(X_check)
        n_predictors = X_check.shape[1]

        # Universal constraint: minimum sample size
        if n_samples < min_samples:
            return False, f"insufficient samples ({n_samples} < {min_samples})"

        # ====================================================================
        # Define model categories for constraint checking
        # ====================================================================

        NONLINEAR_SINGLE_PREDICTOR = [
            'exponential', 'logarithmic', 'power',
            'exponential-offset', 'power-offset', 'logarithmic-power'
        ]

        REQUIRES_POSITIVE_Y = [
            'exponential',
            'exponential-offset',
            'power-log-multivariate',
            'exponential-log-multivariate',

            # Can handle some negative with offset, but safer to require positive
        ]

        REQUIRES_POSITIVE_X = [
            'logarithmic',
            'logarithmic-power',
            'power-log-multivariate',
        ]

        REQUIRES_POSITIVE_X_AND_Y = [
            'power',
            'power-offset'
        ]

        # ====================================================================
        # Check predictor count constraints
        # ====================================================================

        if model_type in NONLINEAR_SINGLE_PREDICTOR:
            if n_predictors != 1:
                return False, f"requires single predictor (got {n_predictors})"
            # Flatten for value checking
            x_values = X_check.flatten()

        else:
            x_values = X_check

        # ====================================================================
        # Check value constraints
        # ====================================================================

        # Check positive Y requirement
        if model_type in REQUIRES_POSITIVE_Y:
            if not np.all(y_check > 0):
                min_y = np.min(y_check)
                return False, f"requires all y > 0 (found min y = {min_y:.4f})"

        # Check positive X requirement
        if model_type in REQUIRES_POSITIVE_X:
            if not np.all(x_values > 0):
                min_x = np.min(x_values)
                return False, f"requires all x > 0 (found min x = {min_x:.4f})"

        # Check positive X AND Y requirement
        if model_type in REQUIRES_POSITIVE_X_AND_Y:
            if not np.all(x_values > 0):
                min_x = np.min(x_values)
                return False, f"requires all x > 0 (found min x = {min_x:.4f})"
            if not np.all(y_check > 0):
                min_y = np.min(y_check)
                return False, f"requires all y > 0 (found min y = {min_y:.4f})"

        # ====================================================================
        # Check sample size relative to model complexity
        # ====================================================================

        # Polynomial models: need more samples than features
        if model_type.startswith('polynomial-'):
            try:
                # Extract degree from model name (e.g., 'polynomial-2' -> 2)
                parts = model_type.split('-')
                degree = int(parts[1])
                #
                # n_poly_features = ModelFactory._estimate_polynomial_features(n_predictors, degree)

                n_poly_features = ModelFactory.get_actual_n_features(model_type=model_type, X=X)

                # Need at least n_features + 1 samples (preferably more)
                if n_samples <= n_poly_features:
                    return False, (f"insufficient samples for polynomial degree {degree} "
                                   f"({n_samples} samples ≤ {n_poly_features} features)")

            except (IndexError, ValueError):
                return False, f"invalid polynomial model type format: '{model_type}'"

        # Linear models with regularization: need enough samples
        elif model_type in ['linear', 'ridge', 'lasso', 'ridge-cv', 'lasso-cv']:
            if n_samples <= n_predictors:
                return False, f"insufficient samples ({n_samples} ≤ {n_predictors} predictors)"

        # Non-linear models: already checked sample size with min_samples

        # ====================================================================
        # All checks passed
        # ====================================================================

        return True, ""

    @staticmethod
    def get_applicable_models(X, y, exclude_models=None, clean_first=True, min_samples=5):
        """
        Get list of all models applicable to given data.

        Filters the full list of available models to only those that
        can be applied to the provided data based on structural and
        mathematical constraints.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Predictor data
        y : array-like, shape (n_samples,)
            Target data
        exclude_models : list of str, optional
            Model types to exclude from consideration
        clean_first : bool, default=True
            Whether to clean data before checking applicability
        min_samples : int, default=5
            Minimum number of samples required

        Returns
        -------
        list of str
            Model type names that are applicable

        """
        all_models = ModelFactory.get_available_models()

        # Apply user exclusions
        if exclude_models:
            all_models = [m for m in all_models if m not in exclude_models]

        # Filter to applicable models
        applicable = []
        for model_type in all_models:
            is_applicable, _ = ModelFactory.is_model_applicable(
                model_type, X, y,
                clean_first=clean_first,
                min_samples=min_samples
            )
            if is_applicable:
                applicable.append(model_type)

        return applicable

    @staticmethod
    def get_inapplicable_models_with_reasons(X, y, exclude_models=None, clean_first=True, min_samples=5):
        """
        Get dictionary of inapplicable models with reasons why.

        Useful for debugging or verbose output to understand why
        certain models were not tested.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Predictor data
        y : array-like, shape (n_samples,)
            Target data
        exclude_models : list of str, optional
            Model types to exclude from consideration
        clean_first : bool, default=True
            Whether to clean data before checking
        min_samples : int, default=5
            Minimum number of samples required

        Returns
        -------
        dict : {model_type: reason}
            Dictionary mapping inapplicable model types to reasons

        """
        all_models = ModelFactory.get_available_models()

        # Apply user exclusions
        if exclude_models:
            all_models = [m for m in all_models if m not in exclude_models]

        # Collect inapplicable models with reasons
        inapplicable = {}
        for model_type in all_models:
            is_applicable, reason = ModelFactory.is_model_applicable(
                model_type, X, y,
                clean_first=clean_first,
                min_samples=min_samples
            )
            if not is_applicable:
                inapplicable[model_type] = reason

        return inapplicable

    @staticmethod
    def is_piecewise_model(model_type):
        """Check if model type is piecewise."""
        return model_type.endswith('-piecewise')

    @staticmethod
    def get_base_model_type(model_type):
        """
        Get base model type from potentially piecewise model.

        Examples
        --------
        >>> ModelFactory.get_base_model_type('power')
        'power'
        >>> ModelFactory.get_base_model_type('power-piecewise')
        'power'
        """
        if model_type.endswith('-piecewise'):
            return model_type.replace('-piecewise', '')
        return model_type

    @staticmethod
    def get_piecewise_diagnostics(model):
        """
        Get diagnostics for a fitted piecewise model.

        Parameters
        ----------
        model : _PiecewiseModelWrapper
            Fitted piecewise model

        Returns
        -------
        dict
            Diagnostic information

        Raises
        ------
        TypeError
            If model is not a piecewise wrapper
        """
        if not isinstance(model, _PiecewiseModelWrapper):
            raise TypeError(
                f"Expected _PiecewiseModelWrapper, got {type(model)}"
            )

        return model.get_bin_diagnostics()