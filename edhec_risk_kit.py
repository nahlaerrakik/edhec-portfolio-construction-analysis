__author__ = 'nahla.errakik'

import numpy as np
import pandas as pd
import scipy.stats
from scipy.optimize import minimize


def get_data(path, date_format, sub_column=None, returns=True):
    """Loads the Dataset for the returns of the top and bottom deciles by MarketCap

    :type
        path: str
        sub_column: list
    :param
        path: location of the csv file containing the data.
        date_format: date format of the index column
        sub_column: subtract the desired columns from the DataFrame if provided.
    :return
        the returns DataFrame."""

    if sub_column is None:
        sub_column = []

    data = pd.read_csv(path, header=0, index_col=0, na_values=-99.99)
    data.columns = data.columns.str.strip()
    if len(sub_column) > 0:
        data = data[sub_column]

    data = data / 100 if returns else data
    data.index = pd.to_datetime(data.index, format=date_format).to_period('M')

    return data


def annualized_return(returns, period_per_year):
    """Calculate the annualized return of a portfolio

    :type
        returns: pandas.DataFrame/pandas.Series
        periods_per_year: int
    :param
        returns: DataFrame/Series of the returns observed over a given period of time.
        base_period: base number of observation in a year. Ex: for monthly returns, base_period =12.
    :return
        the annualized return.
    :raises
        a TypeError exception if the returns is not a Dataframe or Series."""
    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(annualized_return, base_period=period_per_year)
    elif isinstance(returns, pd.Series):
        nbr_of_periods = returns.shape[0]
        return (returns + 1).prod() ** (period_per_year / nbr_of_periods) - 1
    else:
        raise TypeError("Expected returns to be a DataFrame or Series")


def annualized_volatility_from_periodic_volatility(daily_volatility):
    """Calculate the annualized volatility based on the daily volatility.
    Volatility is the standard deviation from the mean of a distribution."""

    return daily_volatility * 252


def annualized_volatility_from_df_detailed(returns, base_period, number_of_periods):
    """Calculate the annualized volatility of portfolio

    :type:
        returns: pandas.DataFrame
        base_period: int
        number_of_periods: int
    :param
        returns_df: DataFrame of the returns observed over a given period of time.
        number_of_periods: number of periods that exist in one year. For example: if the returns are observed monthly,
         the number of periods would be 12
    :return
        the annualized volatility of a portfolio"""

    number_of_obs = returns.shape[0]
    deviations = returns - returns.mean()
    squared_deviations = deviations ** 2
    variance = squared_deviations.sum() / (number_of_obs - 1)
    period_volatility = variance ** 0.5

    return period_volatility * np.sqrt(number_of_periods / base_period)


def annualized_volatility(returns, period_per_year):
    """Another simplified way to calculate the annualized volatility of a portfolio leveraging on the built-in Dataframe functions.

    :type
        returns: pandas.DataFrame/pandas.Series
        periods_per_year: int
    :param
        returns_df: DataFrame/Series of the returns observed over a given period of time
        periods_per_year: number of periods that exist in one year. For example: if the returns are observed monthly,
             the number of periods would be 12
    :return
        the annualized volatility of a portfolio.
    :raises
        a TypeError exception if the returns is not a Dataframe or Series."""

    if isinstance(returns, pd.DataFrame) or isinstance(returns, pd.Series):
        return returns.std() * np.sqrt(period_per_year)
    else:
        raise TypeError("Expected returns to be a DataFrame or Series")


def sharpe_ratio(returns, risk_free_rate, periods_per_year):
    """The sharpe ratio is the excess return that you would get over what you could get with no risk.
     In other words, the excess return over the risk-free rate per unit of volatility."""

    # convert the annual risk-free rate to per period
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_ret = returns - rf_per_period
    ann_ex_ret = annualized_return(excess_ret, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    return ann_ex_ret / ann_vol


def compute_drawdown(return_series):
    """The maximum drawdown is the maximum loss that you could have experienced, if you had been unlucky enough to
    buy the asset or the strategy or whatever you're looking at. At its very peak, and you sold it at the very bottom
    It is the worst return of the peak to trough that you could have experienced over the time that the return series
    are being analyzed.

    BUY at its highest value - SOLD ath the bottom.

    The worst possible return you could have seen if you bought high and sold low.

    How to convert a return series to a drawdown:
    step 1: construct a wealth index hypothetical BUY-AND-HOLD investment in the asset
            (measures what would happen if we take an amount and invested it over a time period).
    step 2: look at the prior peak (highest value) at any point in time.
    step 3: calculate the distance between the current value and the prior peak at any giver point in time

    It is important to note that they are far from the perfect measure:
     For instance, they are entirely defined by 2 points and hence are very sensitive to outliers.
     Second, they depend on the frequency of observations. For example: a very deep drawdown on a daily or weekly basis
      might almost completely disappear or move to very different location based on a monthly data.

    :type
        return_series: pandas.Series
    :param
        return_series: times series of asset returns.
    :return
        a DataFrame that contains: the wealth index, the previous peaks and the percent drawdowns.
    :raises
        a TypeError exception if the returns is not a Series."""

    if not isinstance(return_series, pd.Series):
        raise TypeError("Expected returns to be a Series")

    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    return pd.DataFrame({
        'wealth': wealth_index,
        'peaks': previous_peaks,
        'drawdown': drawdowns
    })


def get_max_drawdown(return_series):
    """Gets the maximum drawdown.

        :type
            return_series: pandas.Series
        :param
            returns_series: times series of asset returns.
        :return
            the max_drawdown observed in a time series of returns.
        """
    df = compute_drawdown(return_series)
    return df['drawdown'].min()


def get_max_drawdown_period(return_series):
    """Gets when the maximum drawdown occurred.

    :type
        return_series: pandas.Series
    :param
        returns_series: times series of asset returns.
    :return
        the period of the max_drawdown observed in a time series of returns.
    """
    df = compute_drawdown(return_series)
    return df['drawdown'].idxmin()


def skewness_kurtosis(returns, skewness: bool = False, kurtosis: bool = False):
    """Alternative to scipy.stats.skew() / scipy.stats.kurtosis
    Computes the skewness and/or kurtosis of a given Series or DataFrame.

    :type
        returns: pandas.DataFrame/pandas.Series
        skewness: boolean
        kurtosis: boolean
    :param
        returns: DataFrame/Series of the asset returns observed over a given period of time.
    :return
        a series or a tuple of skewness/kurtosis data.
    :raises
        a TypeError exception if the returns is not a Series or DataFrame."""

    if not isinstance(returns, pd.Series) and not isinstance(returns, pd.DataFrame):
        raise TypeError("Expected returns to be a DataFrame or Series")

    skewness_value = None
    kurtosis_value = None

    demeaned_r = returns - returns.mean()
    # use of the population standard deviation so we set ddof=0
    sigma_r = returns.std(ddof=0)
    if skewness:
        skewness_exp = (demeaned_r ** 3).mean()
        skewness_value = skewness_exp / sigma_r ** 3
    if kurtosis:
        kurtosis_exp = (demeaned_r ** 4).mean()
        kurtosis_value = kurtosis_exp / sigma_r ** 4

    return skewness_value, kurtosis_value


def is_normal(returns, level=0.01):
    """Applies the Jarque-Bera test to determine if a Series is normal or not.
    The test is applied at the 1% level by default.

    :type
        returns: pandas.DataFrame/pandas.Series
        level: float
    :param
        returns: DataFrame/Series of the asset returns observed over a given period of time.
    :return
        True is the hypothesis of normality is accepted, False otherwise
    :raises
        a TypeError exception if the returns is not a Series or DataFrame."""

    if not isinstance(returns, pd.Series) or not isinstance(returns, pd.DataFrame):
        raise TypeError("Expected returns to be a DataFrame or Series")

    statistic, pv_value = scipy.stats.jarque_bera(returns)
    return pv_value > level


def semi_deviation(returns):
    """Semi deviation is the volatility of the sub-sample of below-average or below-zero returns.

    :param
        returns: DataFrame of the asset returns observed over a given period of time.
    :return
        the semi-deviation aka negative semi-deviation of asset returns
    """

    condition = returns < 0
    return returns[condition].std(ddof=0)


def var(returns, level=0.05, is_historic=False, is_gaussian=False, is_cornish_fisher=False):
    """The Value at Risk (VaR) represents the maximum expected loss over a given time period.
    It is also defined as the Maximum potential loss threshold at a specified confidence level over a specified holding period.

    There are at least 3 standard methods for calculating VaR:
        - Method 1: Historical (non parametric).
                    Calculation of VaR based on the distribution of historical changes in the value of the current portfolio under market prices
                    over the specified historical observation window.
                    Advantages: There is no assumption about asset return distributions so we don't have to worry about specifying the model.
                    Drawbacks: With lack of assumption, we are relying on historical data so the estimate is sensitive to the sample period.
        - Method 2: Variance-Covariance (parametric gaussian).
                    Calculation of VaR based on portfolio volatility; on volatilities and correlation of components.
                    We assume that the distribution is a Gaussian distribution
                    VaR = -(mean+z*volatility) with z is the alpha-quantile of the return distribution.
                    Drawbacks: There is massive amount of risk because we are assuming a Gaussian distribution and it is widely known that asset returns
                    does not follow a Gaussian distribution 99% of the times, so in consequence we might end up underestimating the actual risk in our portfolio.
    The problem with using a parametric method, we are running the risk of specifying a wrong model called a specification risk.

        - Method 3: Cornish-Fisher (semi parametric)
                    Allows to relate the alpha-quantile of a non Gaussian distribution to the alpha-quantile of the Gaussian distribution
                    VaR(1-alpha) = -(mean+Z_bar*volatility)
                        with z_bar = z + (1/6 * (z**2)*skewness) + (1/24 *(z**3 - 3*Z)*(kurtosis - 3)) - (1/36*(2*z**3 - (5*z)*skewness**2)
                    If the skewness is negative and the kurtosis is higher than 3, this adjustment will give a vaR estimate that will be different
                    and higher than the Gaussian estimate.
    :type
        returns: pandas.DataFrame/pandas.Series
        level: float/int
        is_historic: boolean
        is_gaussian: boolean
        is_cornish_fisher: boolean
    :param
        returns: DataFrame/Series of the asset returns observed over a given period of time.
        level: must be between 0-100.
        is_historic: when set to true calculate the VaR using the method 1 (Historical).
        is_gaussian: when set to true calculate the VaR using the method 2 (Variance-Covariance Gaussian).
        is_cornish_fisher: when set to true calculate the VaR using the method 3 (Cornish-Fisher)
    :return
        a dictionary of the VaR calculated using the 3 methods described above.
    :raises
        a TypeError exception if the returns is not a Series or DataFrame."""

    historic_var = None
    gaussian_var = None
    cornish_fisher_var = None

    if isinstance(returns, pd.DataFrame):
        if is_historic:
            historic_var = -returns.aggregate(np.percentile, q=level)
    elif isinstance(returns, pd.Series):
        if is_historic:
            historic_var = -np.percentile(returns, level)
    else:
        raise TypeError("Expected returns to be Series or Dataframe")

    # Compute the Z score assuming the returns follow a Gaussian distribution
    z = scipy.stats.norm.ppf(level)
    if is_gaussian:
        gaussian_var = -(returns.mean() + z * returns.std(ddof=0))
    if is_cornish_fisher:
        # Calculate the Z score based on observed skewness and kurtosis
        s, k = skewness_kurtosis(returns, True, True)
        z = (z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * (k - 3) / 24 - (2 * z ** 3 - 5 * z) * (s ** 2) / 36)
        cornish_fisher_var = -(returns.mean() + z * returns.std(ddof=0))

    return {'Historic': historic_var, 'Gaussian': gaussian_var, 'Cornish-Fisher': cornish_fisher_var}


def cvar(returns, level=0.05):
    """The conditional Value at Risk (Expected loss beyond VaR) is the expected return, conditional upon the return being
    less than the value at risk number E(R|R<=VaR).
    Interpretation: That number says, if that level percent chance happens, that is the worst level percent of the possible cases.
    When those things happen, the average of that is a cvar value percent loss in that given period.

    :param
        returns: DataFrame/Series of the asset returns observed over a given period of time.
        level: must be between 0-100.
    :return
        the CVaR Value.
    :raises
        a TypeError exception if the returns is not a Series or DataFrame."""

    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(cvar, level=level)
    elif isinstance(returns, pd.Series):
        is_beyond = returns <= -var(returns, level=level, is_historic=True)['Historic']
        return -returns[is_beyond].mean()
    else:
        raise TypeError("Expected returns to be Series or Dataframe")


def portfolio_return(weights, returns):
    """Calculate the portfolio return with a given assets weights vector.

    :type
        weights: numpy.array.
        returns: pandas.Dataframe.
    :param
        weights: the weights of the assets in the portfolio.
        returns: the returns of the assets in the portfolio.
    :return
        the returns of each asset composing the portfolio.
    :raises
        a TypeError if weights and returns are not a DataFrame."""

    return weights.T @ returns


def portfolio_volatility(weights, covariance_matrix):
    """Calculate the portfolio volatility.

    :type
        weights: numpy.array.
        covariance_matrix: pandas.DataFrame.
    :param
        weights: the weights of the assets in the portfolio.
        covariance_matrix: the correlation matrix between the assets of the portfolio.
    :return
        the volatility of each asset composing the portfolio.
    :raises
        a TypeError if the weights and covariance_matrix are not a DataFrame"""

    return (weights.T @ covariance_matrix @ weights) ** 0.5


def compute_efficient_frontier(returns, n_points, risk_free_rate=0, show_cml=False, show_ew=False, show_gmv=False):
    """
    - Given a set of asset returns volatilities and correlations, we can plot the efficient frontier.
    - The efficient frontier represents the portfolios that offers the highest possible return for a for a given level of volatility.
    - Portfolio diversification is mixing 2 or more assets, that must be decorrelated in a certain ratio to end up with a portfolio
        that has lower volatility than any volatility of the assets of the portfolio.
    - It is not wise to hold a portfolio inside the efficient frontier region because the best portfolios in terms of expected return and volatility
        are the ones that are sitting on the edge of the region.
    - In order to find the efficient frontier, recall that the portfolio on the efficient frontier is the one with the minimum volatility
        for a certain level of return. So, we need to minimise the volatility by minimizing the variance which is minimizing W transpose sigma W:
        1/2*W(T)*sum(W). Here, we divide by 2 to obtain a quadratic form because we will be using a quadratic optimize that expects quadratic inputs.
    - We supply a set of constraints to the quadratic optimizer:
        1. The return must be at a certain level.
        2. All the assets weights must be greater than zero (no shorting).
        3. The sum of weights must be equal to 1.

    :type
        returns: pandas.DataFrame
        n_points: int
        risk_free_rate: float
        show_cml: boolean
        show_ew: boolean
        show_gmv: boolean
    :param
        returns: the assets returns observed over a given period of time.
        n_points: number of possible portfolios composition that we want to generate.
        risk_free_rate: default value is 0 meaning that they are no risk free asset in our portfolio.
        show_cml: indicates if the Capital Market Line should be plotted (default value is False).
        show_ew: indicates if the equally weighted portfolio should be plotted (default value is False).
        show_gmv: indicates if the global minimum variance portfolio should be plotted (default value is False).
    :return:
        the efficient frontier.
    :raises
        a TypeError if returns is not a DataFrame."""

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("Expected returns to be a DataFrame")

    result = {}

    expected_returns = annualized_return(returns=returns, period_per_year=12)
    cov_matrix = returns.cov()

    # weights = [np.random.dirichlet(np.ones(n_assets),size=1)[0] for n in range(n_points)]
    # weights.sort(key=lambda x: x[0])
    # weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    targets_returns = np.linspace(expected_returns.min(), expected_returns.max(), n_points)
    weights = [minimize_volatility(tg, expected_returns, cov_matrix) for tg in targets_returns]

    portfolio_returns = [portfolio_return(w, expected_returns) for w in weights]
    portfolio_volatilities = [portfolio_volatility(w, cov_matrix) for w in weights]

    efficient_frontier_df = pd.DataFrame({"Return": portfolio_returns, "Volatility": portfolio_volatilities})
    ef_plotted = efficient_frontier_df.plot.line(x="Volatility", y="Return", style=".-")

    # Plot the Capital Market Line if show_cml is True
    if show_cml:
        w_msr = maximum_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate)
        r_msr = portfolio_return(w_msr, expected_returns)
        v_msr = portfolio_volatility(w_msr, cov_matrix)

        cml_x = [0, v_msr]
        cml_y = [risk_free_rate, r_msr]
        ef_plotted.set_xlim(left=0)
        ef_plotted = \
            ef_plotted.plot(cml_x, cml_y, color="green", marker="o", linestyle="dashed", markersize=8, linewidth=2)[
                0].axes
        result.update({'cml': (w_msr, r_msr, v_msr)})

    # Plot the equally weighed portfolio if show_ew is True
    if show_ew:
        n_assets = expected_returns.shape[0]
        w_ew = np.repeat(1 / n_assets, n_assets)
        r_ew = portfolio_return(w_ew, expected_returns)
        v_ew = portfolio_volatility(w_ew, cov_matrix)

        ew_x = [v_ew]
        ew_y = [r_ew]
        ef_plotted = ef_plotted.plot(ew_x, ew_y, color="orange", marker="o", markersize=8)[0].axes
        result.update({'ew': (w_ew, r_ew, v_ew)})

    # Plot the Global Minimum Variance portfolio is show_gmv is True
    if show_gmv:
        w_gmv = global_minimum_variance(covariance_matrix=cov_matrix)
        r_gmv = portfolio_return(w_gmv, expected_returns)
        v_gmv = portfolio_volatility(w_gmv, cov_matrix)

        gmv_x = [v_gmv]
        gmv_y = [r_gmv]
        ef_plotted = ef_plotted.plot(gmv_x, gmv_y, color="red", marker="o", markersize=8)[0].axes
        result.update({'gmv': (w_gmv, r_gmv, v_gmv)})

    result.update({'ef_plotted': ef_plotted})

    return result


def minimize_volatility(target_return, expected_return, covariance_matrix):
    """Generates an optimized weight vector that gives the target return using a scipy built in optimizer.
    We define the following list of constraints:
    1 - The weights must be between 0 and 1.
    2 - The generated weights using the optimizer must add up to 1.
    3 - The generated return from the weights must meet the target return.

    :type
        target_return: float
        expected_return: float
        covariance_matrix: pandas.Dataframe
    :param
        target_return: the portfolio's target return.
        expected_return: the expected return for each asset composing the portfolio.
        covariance_matrix: the portfolio's covariance matrix.
    :return
        an array containing the optimized weight vector."""

    n_assets = expected_return.shape[0]
    init_guess = np.repeat(1 / n_assets,
                           n_assets)  # initial guess is that the portfolio composition is equally distributed between the assets
    weights_bound = ((.0,
                      1.),) * n_assets  # adding "," to construct a tuple of tuples: generates a constraint for every asset in the portfolio
    return_is_target = {
        'type': 'eq',
        'args': (expected_return,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    # Calling the scipy optimizer
    results = minimize(portfolio_volatility, init_guess,
                       args=(covariance_matrix,),
                       method='SLSQP',
                       options={'disp': False},
                       bounds=weights_bound,
                       constraints=(return_is_target, weights_sum_to_1))

    return results.x


def maximum_sharpe_ratio(expected_returns, covariance_matrix, risk_free_rate):
    """
    - The efficient frontier dramatically changes shape when a risk-free asset is introduced and becomes a straight line called the Capital Market Line.
        When there are only risky assets in our portfolio, we get the usual shape of the Markowitz efficient frontier. But when we introduce a risk-free asset,
        the slop of the tangency line all the way until there is still one intersection point. This intersection point is what we call the MAXIMUM SHARPE RATIO PORTFOLIO.
    - The special thing about MSR portfolios is that they contains no exposure to specific risks because they are diversified away, so all investors should
        hold a combination of the risk-free asset and the portfolio that maximizes the reward-per-risk ratio.

    :type
        expected_returns: pandas.Series
        covariance_matrix: pandas.DataFrame
        risk_free_rate: float
    :param
        expected_return: the expected return for each asset composing the portfolio.
        covariance_matrix: the portfolio's covariance matrix.
        risk_free_rate: the risk free rate.
    :returns
        the maximum sharpe ratio return."""

    n_assets = expected_returns.shape[0]
    init_guess = np.repeat(1 / n_assets,
                           n_assets)  # initial guess is that the portfolio composition is equally distributed between the assets
    weights_bound = ((.0,
                      1.),) * n_assets  # adding "," to construct a tuple of tuples: generates a constraint for every asset in the portfolio
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    def neg_sharpe_ratio(w, rfr, er, cov_m):
        """Returns the negative of the sharpe ration with given weights.
        :param
            w: the assets weight vector in the portfolio.
            rfr: the risk free rate.
            er: the portfolio expected return.
            cov_m: the portfolio's covariance matrix.
        :return
            the negative sharpe ratio."""
        ret = portfolio_return(w, er)
        vol = portfolio_volatility(w, cov_m)
        return - (ret - rfr) / vol

    # Calling the scipy optimizer to maximum the sharpe ratio by minimizing the negative sharpe ratio
    results = minimize(neg_sharpe_ratio, init_guess,
                       args=(risk_free_rate, expected_returns, covariance_matrix),
                       method='SLSQP',
                       options={'disp': False},
                       bounds=weights_bound,
                       constraints=(weights_sum_to_1,))

    return results.x


def global_minimum_variance(covariance_matrix):
    """Lack of robustness of Markowitz analysis:
        Estimation error is the key challenge of portfolio optimization. If we feed an optimizer with parameters that are severely miss-estimated,
        with lot of estimation errors embedded in them, we're going to get a portfolio that's not going to be a very meaningful portfolio.
        In portfolio optimization, it is possible to estimate a good covariance matrix parameter, however the excepted returns are much harder
        to obtain with a good degree of accuracy. Therefore, we can focus on the GLOBAL MINIMUM VARIANCE PORTFOLIO which is a portfolio that requires
        no expected return estimates.

    :type:
        covariance_matrix: pandas.Dataframe.
    :param
        covariance_matrix: the portfolio covariance matrix.
    :return
        the weight vector of the Global Minimum Variance portfolio.
    """

    # We call the maximum sharpe_ratio to minimize the volatility with the expected returns equal to one.
    # When the optimizer is given a set of expected returns that are all the same, it will improve the Sharpe ratio by dropping the volatility only
    n_assets = covariance_matrix.shape[0]
    return maximum_sharpe_ratio(expected_returns=np.repeat(1, n_assets), covariance_matrix=covariance_matrix,
                                risk_free_rate=0)


def market_capitalization(nbr_shares, price):
    """Returns the market capitalization"""
    return nbr_shares * price


def study_limits_of_diversification():
    ind_returns = get_data(path='./data/ind30_m_vw_rets.csv', date_format="%Y%m")
    ind_nfirms = get_data(path='./data/ind30_m_nfirms.csv', date_format="%Y%m", returns=False)
    ind_size = get_data(path='./data/ind30_m_size.csv', date_format="%Y%m", returns=False)

    ind_maktcap = ind_nfirms * ind_size
    total_mktcap = ind_maktcap.sum(axis="columns")

    ind_capweight = ind_maktcap.divide(total_mktcap, axis="rows")
    total_market_return = (ind_capweight * ind_returns).sum(axis="columns")

    # total_market_index = compute_drawdown(total_market_return).wealth
    # total_market_index.plot().figure.savefig("myfig.png")

    # compute a new time series out of a window of 36 months
    # total_market_index["1980":].rolling(window=36).mean().plot().figure.savefig("myfig.png")

    # Rolling returns out of window of 36 mounts
    # total 36 trailing market index returns
    tmi_tr36_rets = total_market_return.rolling(window=36).aggregate(annualized_return, periods_per_year=12)
    tmi_tr36_rets.plot(label='Tr36 Mo Rets', legend=True).figure.savefig("Tr36_Mo_rets_corr.png")

    # Rolling correlation out of window of 36 mounts along with multi indexing and group by
    ts_corr = ind_returns.rolling(window=36).corr()
    ts_corr.index.names = ['date', 'industry']

    ind_tr36m_corr = ts_corr.groupby(level='date').apply(lambda corr_mat: corr_mat.values.mean())
    ind_tr36m_corr.plot(label='Tr36 Mo Corr', legend=True, secondary_y=True).figure.savefig("Tr36_Mo_rets_corr.png")
    """We notice that when the market is falling down, correlation spikes up, and the market starts to rise up, correlation goes back to normal levels.
    This is when diversification is not very helpful can be a bad investment strategy."""


def cppi(risky_returns, safe_returns=None, multiplier=3, start=1000, floor=0.8, risk_free_rate=0.03, drawdown=None):
    """Part I:
        Constant Proportion Portfolio Insurance (CPPI) strategies was introduced by Black & Jones in 1987. It allows to generate convex option-like payoffs
    without using options. This procedure dynamically allocates total assets to a risky asset and a safe asset.
    The principle is to allocate to a risky asset in multiple M of the difference between your asset value and a given floor (cushion).
    The given floor is the minimum asset level that you don't want to go below it. As we get closer to the floor, the allocation is reduced and as we hit the floor,
    it goes to zero.

    cushion(c) = cppi - floor(f)
    riksy_asset(E) = Multiplier(M) * C
    cppi = E + riskless asset(B)

    Advantages: When implemented carefully and when willing to trade extremely often, nothing can go wrong.
    Downside: In practise, willing to trade on a daily basis can generate lots of transaction costs. Even when trading on a monthly or a quarterly basis,
    it could happen that between 2 trading dates, the loss in the risky component is so large that you get below the floor before having time to trade.

    The risk of breaching the floor because of discrete trading in CPPI strategies is known as gap risk. Gap risk materializes IF AND ONLY IF the loss on
    the risky portfolio relative to the safe portfolio exceeds 1/M within the trading interval.

    1 - There are suitable extensions that can accommodate the presence of maximum drawdown constraints. The principal is to protect a floor which is the
    maximum drawdown floor.
    The maximum drawdown constraints is : V(t) > alpha * M(t), where:
        - V(t): The value of the portfolio at time t.
        - M(t): The maximum value of the portfolio between 0 and time t.
        - 1-alpha: The maximum acceptable drawdown.

    By imposing that the value of your portfolio is greater than alpha percent of the maximum value, then this allows you to protect one minus alpha as a maximum drawdown.
    The max drawdown floor is as we say in probability theory a never decreasing process, it either increases or stay flat but never decreases.
    To protect maximum drawdown floor, we just pick a multiplier and invest in the performance, or by contradiction we can also invest, in the risky portfolio,
    a multiplier times the distance between the asset value and the floor, which is given in this case by the max drawdown floor.

    2 - another extension of the CPPI strategy is to have a floor (which is the minimum ammount of wealth tht an investor want to protect) but also a cap
    (which is the maximum amount of wealth that the asset owner would like to protect). This allows to protect the downside of a portfolio.
    In practise, this is done by defining a threshold level. If the asset value is between the floor and the threshold, the allocation to risky asset should be
    the multiplier times the asset value minus the floor ( F(t) <= A(t) <= T(t) = m*(A(t) - F(t)) ).
    When the asset is above the floor, we have to switch gears and make sure to not hit the cap by allocating the  the allocation should be a multiplier times
    the cap value minus the asset value ( T(t) <= A(t) <= C(t) = m*(C(t) - A(t)) ).

    **HOW TO PICK THE THRESHOLD VALUE? : By imposing the smooth-pasting condition : A(t) = T(t) ==> m*(T(t) - F(t)) = m*(C(t) - T(t)). This means that the allocation should
    be exactly the same looking up and looking down when you are at the threshold value.

    Run a backtest of the CPPI strategy, given a set of returns for the risky asset

    :type
        risky_returns: pd.Dataframe/pd.Series
        safe_returns: pd.Dataframe/pd.Series
        multiplier: int
        start: int/float
        floor: float
        risk_free_rate: float
    :return
        a dictionary containing: Asset Value History, Risk Budget History and Risky Weight History
    """

    # Set up the CPPI parameters
    dates = risky_returns.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = start

    if isinstance(risky_returns, pd.Series):
        risky_returns = pd.DataFrame(risky_returns, columns=['R'])

    if safe_returns is None:
        safe_returns = pd.DataFrame().reindex_like(risky_returns)
        safe_returns.values[:] = risk_free_rate / 12

    # Set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_returns)
    cushion_history = pd.DataFrame().reindex_like(risky_returns)
    risky_weight_history = pd.DataFrame().reindex_like(risky_returns)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_history)
            floor_value = peak * (1 - drawdown)
        cushion = (account_value - floor_value) / account_value  # in percentage
        risky_weight = multiplier * cushion
        risky_weight = np.minimum(risky_weight, 1)
        risky_weight = np.maximum(risky_weight, 0)
        safe_weight = 1 - risky_weight

        risk_allocation = account_value * risky_weight
        safe_allocation = account_value * safe_weight

        # Recompute the new value at the end of the step
        account_value = risk_allocation * (1 + risky_returns.iloc[step]) + safe_allocation * (
                1 + safe_returns.iloc[step])

        # Save the histories for analysis and plotting
        account_history.iloc[step] = account_value
        cushion_history.iloc[step] = cushion
        risky_weight_history.iloc[step] = risky_weight

    risky_wealth = start * (1 + risky_returns).cumprod()

    return {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_weight_history,
        "multiplier": multiplier,
        "stat": start,
        "floor": floor,
        "risky_returns": risky_returns,
        "safe_returns": safe_returns
    }


def summary_stats(r, risk_free_rate=0.03):
    """Return a Dataframe that contains aggregated summary stats fo the returns in the columns of r"""

    ann_r = r.aggregate(annualized_return, period_per_year=12)
    ann_v = r.aggregate(annualized_volatility, period_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, risk_free_rate=risk_free_rate, periods_per_year=12)
    dd = r.aggregate(lambda a: compute_drawdown(a)['drawdown'].min())
    skew = r.aggregate(lambda b: skewness_kurtosis(b, skewness=True)[0])
    kurt = r.aggregate(lambda c: skewness_kurtosis(c, kurtosis=True)[1])
    var_cf = r.aggregate(lambda d: var(d, is_cornish_fisher=True)['Cornish-Fisher'])
    hist_cva = r.aggregate(cvar)

    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_v,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": var_cf,
        "Historic CVaR (5%)": hist_cva,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })


def geometric_brownian_motion(n_years=10, n_scenarios=1000, u=0.07, sigma=0.15, steps_per_year=12, s_0=100.0):
    """Asset returns are often assumed to follow a RANDOM WALK. We assume that asset returns are NORMALLY DISTRIBUTED, with ZERO SERIAL CORRELATION and a VARIANCE
    PROPORTIONAL TO TIME. It is possible to generate meaningful, reasonable scenarios for asset returns using a random walk model in continuous.
    The return process is modelled as :
        return = dS(t)/S(t) = u*d(t) + sigma*dW(t)
        with u = (r + sigma*lambda)

        dS(t): the change in value of the asset (stock index).
        u: the expected return decomposed into two components.
        r: risk free rate.
        sigma: the volatility of the asset (stock index).
        lambda: the sharpe ratio on the asset (stock index).
        dt: the interval of time.
        dW(t): standard brownian motion process equivalent to a random walk in continuous time.
        The brownian motion follows a normal distribution with zero MEAN and VARIANCE dt, u is the ANNUALIZED EXPECTED RETURN and sigma is the ANNUALIZED VOLATILITY;
        What's important about this Brownian Motion is the concept of independent increments which mean that at all points, the probability to go up or to go down
        these Brownian motion is always half (there is no serial correlation, just a random walk). This has been recognized as a fair approximation of the returns
        on securities. Of course if you look closely you could find some serial correlation but at least it's a reasonable model for reasonable situation when we
        have to simulate as written.

    In practise, everything is time-changing, this means that r(risk free rate), sigma (volatility and can be considered as the square root of the variance),
    lambda(sharpe ratio) shouldn't be regarded as a constant and should be regarded as a time varying quantity.
        dr(t,r) = a(b - r(t))dt + sigma(r) * dW(t,r) (*check function cir)
        dV(t,V) = alpha(V - V(t))dt + sigma(V) * square(V(t)) * dW(t,V)
        Note: Since the we know that Variance>0, we are introducing a slight modification in the process by introducing a square root of variance before the Brownian
        perturbation. So that as we get close to a zero value, and eventually when we reach a zero value for variance, the Brownian perturbation goes away so that we
        cannot go below zero.
    """

    dt = 1 / steps_per_year
    n_steps = int(n_years * steps_per_year)
    """Way 1:"""
    """xi = np.random.normal(size=(n_steps, n_scenarios))
    returns = u * dt + sigma * np.sqrt(dt) * xi
    returns[0] = s_0
    returns = pd.DataFrame(returns)
    prices = s_0 * (1 + returns).cumprod()"""

    """Way 2: ( more efficient)"""
    returns_plus_1 = np.random.normal(loc=u * dt,  # mean
                                      scale=sigma * np.sqrt(dt),  # volatility
                                      size=(n_steps, n_scenarios))
    returns_plus_1[0] = s_0
    returns_plus_1 = pd.DataFrame(returns_plus_1)
    prices = s_0 * returns_plus_1.cumprod()

    return prices


def discount(time, interest_rate):
    """Compute the price of a pure discount bond that pays a dollar at time given an interest rate"""
    return 1 / (1 + interest_rate) ** time


def present_value(liabilities, interest_rate):
    """Computes the present value of a sequence of liabilities
    
    :type
        liability: pd.Series
        interest_rate: float"""

    dates = liabilities.index
    discounts = discount(dates, interest_rate)
    pv = (discounts * liabilities).sum()

    return pv


def funding_ratio(assets, liabilities, interest_rate):
    """"In Asset-Liability Management, what is important is the ASSET value relative to LIABILITY value also called the FUNDING RATIO.
    FUNDING RATIO measures the assets relative to liabilities. If it's equal to 100ù, it means that assets are SUFFICIENT TO COVER LIABILITIES:
        F(t) = A(t) / L(t).
    The difference between ASSET and LIABILITY values is called SURPLUS when the difference is positive and DEFICIT when the difference is negative:
        S(t) = A(t) - L(t)
             OR
        D(t) = A(t) - L(t)

    The present value of a set of liabilities L where each liability L(i) is due at time t(i) is given by : PV(L) = sum( B(ti) * L(i) ) where:
        B(ti): the discount factor at time t(i)
        If we assume the yield curve is flat and the annual rate of interest is r then B(t) is given by: B(t) = 1 / (1+r)**t

    Institutional investors main concern is an UNEXPECTED INCREASE IN THE PRESENT VALUE OF HEIR LIABILITIES.
    LIABILITY-HEDGING PORTFOLIOS (LHP) or GOAL-HEDGING PORTFOLIOS (GHP) are portfolios with payoffs matching the date and nominal amount of LIABILITY/GOAL payments.
    Note: we say LIABILITY in the context of institutional money management and GOAL in the context of individual money management.

    Often, this is done by investing in nominal bonds or real bonds. Sometimes, it is more convenient to use derivatives based investment solution (inflation swaps,
    inflation link liabilities, ...). More generally, The safe asset the asset that is safe to the investor and his specific goal.

    When CASH-FLOW MATCHING is not FEASIBLE/PRACTICAL, we can use FACTOR EXPOSURE MATCHING instead (also called DURATION MATCHING). This is done by holding a bond
    portfolio that has the exact same duration as the liability portfolio. In doing so, we ensure that if the interest rate goes up, both liability value and asset
    value will go up by the same amount because they have, by construction, the same duration.

    How do we choose how much to allocate to PSP and LHP for each given investor?
    In theory, the allocation to these two portfolios should be decided as a function of Risk Aversion parameter (gamma), meaning that more risk averse investors
    would invest more in liability hedging portfolios, and less risk averse investors would invest less in liability hedging portfolios, and more in performance
    seeking portfolios. However, the Risk Aversion parameter is not observable. That's just a mathematical quantity that we fit in the models but that has no actual
    real world counterpart.
    The best way is to think of Risk Aversion as a free parameter that represents the risk budget: the maximum loss that an investor can take in terms of funding ratio.


    :type
        assets: pd.Series
        liabilities: pd.Series
        interest_rate: float"""

    return assets / present_value(liabilities=liabilities, interest_rate=interest_rate)


def ins_to_ann(interest_rate):
    """Converts short rate to annualized rate
    1 + r(annual) = exp(r(instant))"""

    return np.expm1(interest_rate)


def ann_to_inst(interest_rate):
    """Convert annualized to short rate
    r(instant) = ln(1+ r(annual))"""

    return np.log1p(interest_rate)


def cir(n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """Implements the CIR model for interest rates: dr(t,r) = a(b - r(t))dt + sigma(r) * square(r(t)) * dW(t,r)"""

    if r_0 is None:
        r_0 = b

    r_0 = ann_to_inst(r_0)
    dt = 1 / steps_per_year

    num_steps = int(n_years * steps_per_year)
    dwt = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))

    rates = np.empty_like(dwt)
    rates[0] = r_0
    for step in range(1, num_steps):
        rt = rates[step - 1]  # previous rate
        drt = a * (b - rt) * dt + sigma * np.sqrt(rt) * dwt[step]
        rates[step] = rt + drt

    return pd.DataFrame(data=ins_to_ann(rates), index=range(num_steps))


def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns the series of cash flows generated by a bond,
    indexed by the payment/coupon number
    """
    n_coupons = round(maturity * coupons_per_year)
    coupon_amt = principal * coupon_rate / coupons_per_year
    coupon_times = np.arange(1, n_coupons + 1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows


def bond_price(maturity, principal=100, coupon_rate=.03, coupons_per_year=12, discount_rate=.03):
    """Price a bond based on parameters maturity, principal, coupon rate and number of coupons per year and the discount rate

    Note: as interest rate go up, the bond price falls."""

    cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
    pv = present_value(cash_flows, discount_rate / coupons_per_year)

    return pv


def macaulay_duration(flows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cash flows, given a per-period discount rate
    """
    discounted_flows = discount(flows.index, discount_rate) * pd.DataFrame(flows)
    weights = discounted_flows / discounted_flows.sum()
    return np.average(flows.index, weights=weights.iloc[:, 0])


def match_durations(cf_target_bond, cf_short_bond, cf_long_bond, discount_rate):
    """Returns the weight W, in the short bond cash flows, along with (1-W) in the long bond cash flows will have an effective duration that matched the target bond
    cash flows:
        Ws * ds + (1 - Ws) * dl = dt
        Ws = (dl - dt) / (dl - ds)"""

    d_target = macaulay_duration(cf_target_bond, discount_rate)
    d_short = macaulay_duration(cf_short_bond, discount_rate)
    d_long = macaulay_duration(cf_long_bond, discount_rate)

    weight = (d_long - d_target) / (d_long - d_short)

    return weight


def ldi():
    """LIABILITY DRIVEN INVESTMENT is trying to manage assets again by way of taking into account the presence of liabilities and doing it in a very efficient manner. 
    The objective is to generate the best performance through OPTIMAL EXPOSURE to REWARDED RISK FACTORS (increasing the funding ratio), but also HEDGE against UNEXPECTED 
    SHOCKS. These two conflicting objectives are best managed when managed separately: 
        - PERFORMANCE-SEEKING PORTFOLIO (PSP) which focuses on diversified, efficient access to risk premia.
        - LIABILITY-HEDGING PORTFOLIO (LHP) which focuses on hedging impact of risk factors in liabilities.

    max(w) E[u(At/Lt)] => w = [alpha(PSP) / gamma*sigma(PSP)] * w(PSP) + beta(LHP) * (1-1/gamma) * w(LHP) where:
        - alpha(PSP): is the PSP Sharpe ration; when alpha = 0, there is no investment in PSP.
        - beta(LHP): is the beta of liabilities; when beta = 0, there is no investment in LHP.
        sigma(PSP): is the PSP volatility; when sigma = infinity, there is no investment in PSP.
        gamma: is the risk-aversion (degree of freedom); when gamma = infinity, there is no investment in PSP."""

    pass
