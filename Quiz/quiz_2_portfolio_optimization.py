__author__ = 'nahla.errakik'

import edhec_risk_kit as erk
import numpy as np

question = 0

"""Use the EDHEC Hedge Fund Indices data set that we used in the lab assignment as well as in the previous week’s assignments. 
Load them into Python and perform the following analysis based on data since 2000 (including all of 2000)."""
returns = erk.get_data(path='./data/edhec-hedgefundindices.csv', date_format="%d/%m/%Y")["2000":]

"""1.What was the Monthly Parametric Gaussian VaR at the 1% level (as a +ve number) of the Distressed Securities strategy?"""
distressed_securities_gaussian_var = erk.var(returns=returns["Distressed Securities"], level=1, is_gaussian=True)[
    'Gaussian']
distressed_securities_gaussian_var = distressed_securities_gaussian_var
question += 1
print("{}. = {}".format(question, distressed_securities_gaussian_var * 100))

"""2.Use the same data set at the previous question. What was the 1% VaR for the same strategy after applying the Cornish-Fisher Adjustment?"""
distressed_securities_cornish_fisher_var = \
    erk.var(returns=returns["Distressed Securities"], level=1, is_cornish_fisher=True)['Cornish-Fisher']
distressed_securities_cornish_fisher_var = distressed_securities_cornish_fisher_var
question += 1
print("{}. = {}".format(question, distressed_securities_cornish_fisher_var * 100))

"""3. Use the same dataset as the previous question. What was the Monthly Historic VaR at the 1% level (as a +ve number) of the Distressed Securities strategy?"""
distressed_securities_historic_var = erk.var(returns=returns["Distressed Securities"], level=1, is_historic=True)[
    'Historic']
distressed_securities_historic_var = distressed_securities_historic_var
question += 1
print("{}. = {}".format(question, distressed_securities_historic_var * 100))

"""Next, load the 30 industry return data using the erk.get_ind_returns() function that we developed during the lab sessions. 
For purposes of the remaining questions, use data during the 5 year period 2013-2017 (both inclusive) to estimate the expected returns
as well as the covariance matrix. To be able to respond to the questions, you will need to build the MSR, EW and GMV portfolios consisting of
the “Books”, “Steel”, "Oil", and "Mines" industries. Assume the risk free rate over the 5 year period is 10%."""

returns = erk.get_data(path='./data/ind30_m_vw_rets.csv', date_format="%Y%m",
                       sub_column=['Books', 'Steel', 'Oil', 'Mines'])["2013":"2017"]
risk_free_rate = 10 / 100
components = {0: "Books", 1: "Steel", 2: "Oil", 3: "Mines"}
ef = erk.compute_efficient_frontier(returns=returns, n_points=20, risk_free_rate=risk_free_rate, show_cml=True,
                                    show_ew=True, show_gmv=True)

"""4. What is the weight of Steel in the EW Portfolio?"""
question += 1
w_ew = ef['ew'][0]
ew_steel_weights = ef['ew'][0][1]
print("{}. = {}".format(question, ew_steel_weights * 100))

"""5. What is the weight of the largest component of the MSR portfolio?"""
question += 1
w_msr = ef['cml'][0]
msr_large_weight = w_msr.max()
print("{}. = {}".format(question, msr_large_weight * 100))

"""6. Which of the 4 components has the largest weight in the MSR portfolio?"""
question += 1
msr_large_weight_asset = components[np.where(ef['cml'][0] == msr_large_weight)[0].max()]
print("{}. = {}".format(question, msr_large_weight_asset))

"""7. How many of the components of the MSR portfolio have non-zero weights?"""
question += 1
print("{}. = {}".format(question, 1))

"""8. What is the weight of the largest component of the GMV portfolio?"""
question += 1
print("{}. = {}".format(question, ef['gmv'][0].max() * 100))

"""9. Which of the 4 components has the largest weight in the GMV portfolio?"""
question += 1
w_gmv = ef['gmv'][0]
gmv_large_weight = ef['gmv'][0].max()
gmv_large_asset = components[np.where(ef['gmv'][0] == gmv_large_weight)[0].max()]
print("{}. = {}".format(question, gmv_large_asset))

"""10. How many of the components of the GMV portfolio have non-zero weights?"""
question += 1
count = [1 for x in ef['gmv'][0] if x > 0]
print("{}. = {}".format(question, len(count)))

"""Assume two different investors invested in the GMV and MSR portfolios at the start of 2018 using the weights we just computed. 
Compute the annualized volatility of these two portfolios over the next 12 months of 2018? 
(Hint: Use the portfolio_vol code we developed in the lab and use ind[“2018”][l].cov() to compute the covariance matrix for 2018,
assuming that the variable ind holds the industry returns and the variable l holds the list of industry portfolios you are willing to hold.
Don’t forget to annualized the volatility)"""
returns_2018 = erk.get_data(path='./data/ind30_m_vw_rets.csv', date_format="%Y%m",
                            sub_column=['Books', 'Steel', 'Oil', 'Mines'])["2018":]
cov_2018 = returns_2018.cov()

"""11. What would be the annualized volatility over 2018 using the weights of the MSR portfolio?"""
question += 1
msr_vol = erk.portfolio_volatility(w_msr, cov_2018)
msr_annualized_vol = msr_vol * np.sqrt(12)
print("{}. = {}".format(question, msr_annualized_vol * 100))

"""What would be the annualized volatility over 2018 using the weights of the GMV portfolio? 
(Reminder and Hint: Use the portfolio_vol code we developed in the lab and use ind[“2018”][l].cov() to compute the covariance matrix for 2018, 
assuming that the variable ind holds the industry returns and the variable l holds the list of industry portfolios you are willing to hold. 
Don’t forget to annualized the volatility)"""
question += 1
gmv_vol = erk.portfolio_volatility(w_gmv, cov_2018)
gmv_annualized_vol = gmv_vol * np.sqrt(12)
print("{}. = {}".format(question, gmv_annualized_vol * 100))
