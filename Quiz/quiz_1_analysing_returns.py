__author__ = 'nahla.errakik'

import edhec_risk_kit as erk

"""Read in the data in the file “Portfolios_Formed_on_ME_monthly_EW.csv” as we did in the lab sessions.
We performed a series of analysis on the ‘Lo 10’ and the ‘Hi 10’ columns which are the returns of the lowest and highest decile portfolios.
 For purposes of this assignment, we will use the lowest and highest quantile portfolios, which are labelled ‘Lo 20’ and ‘Hi 20’ respectively."""
returns = erk.get_data(path='./data/Portfolios_Formed_on_ME_monthly_EW.csv', date_format="%Y%m",
                       sub_column=['Lo 20', 'Hi 20'])
LO20 = returns['Lo 20']
HI20 = returns['Hi 20']

"""What was the Annualized Return of the Lo 20 portfolio over the entire period?"""
lo20_annualized_return = erk.annualized_return(returns=LO20, period_per_year=12)
print("Lo 20 annualized return: {} %".format(lo20_annualized_return * 100))

"""What was the Annualized Volatility of the Lo 20 portfolio over the entire period?"""
lo20_annualized_volatility = erk.annualized_volatility(returns=LO20, period_per_year=12)
print("Lo 20 annualized volatility: {} % \n".format(lo20_annualized_volatility * 100))

"""What was the Annualized Return of the Hi 20 portfolio over the entire period?"""
hi20_annualized_return = erk.annualized_return(returns=HI20, period_per_year=12)
print("Hi 20 annualized return: {} %".format(hi20_annualized_return * 100))

"""What was the Annualized Volatility of the Hi 20 portfolio over the entire period?"""
hi20_annualized_volatility = erk.annualized_volatility(returns=HI20, period_per_year=12)
print("Hi 20 annualized volatility: {} %\n".format(hi20_annualized_volatility * 100))

"""What was the Annualized Return of the Lo 20 portfolio over the period 1999 - 2015 (both inclusive)?"""
LO20_1999_2015 = returns['Lo 20']["1999":"2015"]
lo20_annualized_return_1999_2015 = erk.annualized_return(returns=LO20_1999_2015, period_per_year=12)
print("Lo 20 annualized return over the period of 1999-2015: {} %".format(lo20_annualized_return_1999_2015 * 100))

"""What was the Annualized Volatility of the Lo 20 portfolio over the period 1999 - 2015 (both inclusive)? """
lo20_annualized_volatility_1999_2015 = erk.annualized_volatility(returns=LO20_1999_2015, period_per_year=12)
print("Lo 20 annualized volatility over the period of 1999-2015: {} %\n".format(
    lo20_annualized_volatility_1999_2015 * 100))

"""What was the Annualized Return of the Hi 20 portfolio over the period  1999 - 2015 (both inclusive)?"""
HI20_1999_2015 = returns['Hi 20']["1999":"2015"]
hi20_annualized_return_1999_2015 = erk.annualized_return(returns=HI20_1999_2015, period_per_year=12)
print("Hi 20 annualized return over the period of 1999-2015: {} %".format(hi20_annualized_return_1999_2015 * 100))

"""What was the Annualized Volatility of the Hi 20 portfolio over the period 1999 - 2015 (both inclusive)?"""
hi20_annualized_volatility_1999_2015 = erk.annualized_volatility(returns=HI20_1999_2015, period_per_year=12)
print("Hi 20 annualized volatility over the period of 1999-2015: {} %\n".format(
    hi20_annualized_volatility_1999_2015 * 100))

"""What was the Max Drawdown (expressed as a positive number) experienced over the 1999-2015 period in the SmallCap (Lo 20) portfolio?"""
lo20_max_drawdown_1999_2015 = erk.get_max_drawdown(return_series=LO20_1999_2015)
print("Lo 20 max drawdown over the period of 1999-2015: {} %".format(lo20_max_drawdown_1999_2015 * 100))

"""At the end of which month over the period 1999-2015 did that maximum drawdown on the SmallCap (Lo 20) portfolio occur?"""
lo20_max_drawdown_period_1999_2015 = erk.get_max_drawdown_period(return_series=LO20_1999_2015)
print("Lo 20 max drawdown period over the period of 1999-2015: {}\n".format(lo20_max_drawdown_period_1999_2015))

"""What was the Max Drawdown (expressed as a positive number) experienced over the 1999-2015 period in the LargeCap (Hi 20) portfolio?"""
hi20_max_drawdown_1999_2015 = erk.get_max_drawdown(return_series=HI20_1999_2015)
print("Hi 20 max drawdown over the period of 1999-2015: {} %".format(hi20_max_drawdown_1999_2015 * 100))

"""Over the period 1999-2015, at the end of which month did that maximum drawdown of the LargeCap (Hi 20) portfolio occur?"""
hi20_max_drawdown_period_1999_2015 = erk.get_max_drawdown_period(return_series=HI20_1999_2015)
print("Hi 20 max drawdown period over the period of 1999-2015: {}\n".format(hi20_max_drawdown_period_1999_2015))

""""For the remaining questions, use the EDHEC Hedge Fund Indices data set that we used in the lab assignment and load them into Python."""
returns = erk.get_data(path="./data/edhec-hedgefundindices.csv", date_format="%d/%m/%Y")

"""Looking at the data since 2009 (including all of 2009) through 2018 which Hedge Fund Index has exhibited the highest semi-deviation?"""
highest_semi_deviation_hedge_fund = erk.semi_deviation(returns["2009":]).idxmax()
print("Hedge with the highest semi-deviation: {}\n".format(highest_semi_deviation_hedge_fund))

"""Looking at the data since 2009 (including all of 2009) which Hedge Fund Index has exhibited the lowest semideviation?"""
lowest_semi_deviation_hedge_fund = erk.semi_deviation(returns["2009":]).idxmin()
print("Hedge with the lowest semi-deviation: {}\n".format(lowest_semi_deviation_hedge_fund))

""""Looking at the data since 2009 (including all of 2009) which Hedge Fund Index has been most negatively skewed?"""
negative_skewness_since_2009 = erk.skewness_kurtosis(returns["2009":], skewness=True)[0].idxmin()
print("Hedge fund that is the most negatively skewed since 2009: {}".format(negative_skewness_since_2009))

"""Looking at the data since 2000 (including all of 2000) through 2018 which Hedge Fund Index has exhibited the highest kurtosis?"""
high_kurtosis_since_2000 = erk.skewness_kurtosis(returns["2000":], kurtosis=True)[1].idxmax()
print("Hedge fund that exhibited the highest kurtosis since 2000: {}\n".format(high_kurtosis_since_2000))
