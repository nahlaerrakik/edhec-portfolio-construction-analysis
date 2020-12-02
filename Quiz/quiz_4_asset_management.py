__author__ = 'nahla.errakik'

import edhec_risk_kit as erk
import pandas as pd

"""In the following questions, we will be working with three bonds:

B1 is a 15 Year Bond with a Face Value of $1000 that pays a 5% coupon semi-annually (2 times a year)
B2 is a 5 Year Bond with a Face value of $1000 that pays a 6% coupon quarterly (4 times a year)
B3 is a 10 Year Zero-Coupon Bond with a Face Value of $1000 (Hint: you can still use the erk.bond_cash_flows() and erk.bond_price() by setting the coupon amount to 0% and coupons_per_year to 1) Assume the yield curve is flat at 5%. Duration refers to Macaulay Duration
Hint: the macaulay_duration function gives as output the duration expressed in periods and not in years. If you want to get the yearly duration you need to divide the duration for coupons_per_year;

e.g.: duarion_B2 = erk.macaulay_duration(flows_B2, 0.05/4)/4"""


def question_1_2_3():
    b1 = erk.bond_price(15, 1000, 0.05, 2, 0.05)
    b2 = erk.bond_price(5, 1000, 0.06, 4, 0.05)
    b3 = 1000 / 1.05 ** 10

    bonds = {b1: 'b1', b2: 'b2', b3: 'b3'}

    max_bond = max([b1, b2, b3])
    min_bond = min([b1, b2, b3])

    print("Q1: Which of the three bonds is the most expensive? {}".format(bonds[max_bond]))
    print("Q2: Which of the three bonds is the least expensive? {}".format(bonds[min_bond]))
    print("Q3: What is the price of the 10 Year Zero Coupon Bond B3? {}".format(b3))


def question_4_5_6():
    d1 = erk.macaulay_duration(erk.bond_cash_flows(15, 1000, 0.05, 2), 0.05 / 2) / 2
    d2 = erk.macaulay_duration(erk.bond_cash_flows(5, 1000, 0.06, 4), 0.05 / 4) / 4
    d3 = erk.macaulay_duration(erk.bond_cash_flows(10, 1000, 0.00), 0.05)

    max_duration = max([d1, d2, d3])
    min_duration = min([d1, d2, d3])

    durations = {d1: 'd1', d2: 'd2', d3: 'd3'}

    print("Q4: Which of the three bonds has the highest (Macaulay) Duration? {}".format(durations[max_duration]))
    print("Q5: Which of the three bonds has the lowest  (Macaulay) Duration? {}".format(durations[min_duration]))
    print("Q6: What is the duration of the 5 year bond B2? {}".format(d3))


def question_7():
    liabilities = pd.Series(data=[100000, 200000, 300000], index=[3, 5, 10])
    res = erk.macaulay_duration(liabilities, .05)
    print(
        "Q7: Assume a sequence of 3 liabilities of $100,000, $200,000 and $300,000 that are 3, 5 and 10 years away, respectively. "
        "What is the Duration of the liabilities? {}".format(res))


def question_8():
    """Assuming the same set of liabilities as the previous question
    (i.e. a sequence of 3 liabilities of 100,000, 200,000 and $300,000 that are 3, 5 and 10 years away, respectively)
    build a Duration Matched Portfolio of B1 and B2 to match these liabilities. What is the weight of B2 in the portfolio?
    (Hint: the code we developed in class erk.match_durations() assumes that all the bonds have the same number of coupons per year.
    This is not the case here, so you will either need to enhance the code or compute the weight directly e.g. by entering the steps in a
    Jupyter Notebook Cell or at the Python Command Line)"""

    pass


def question_9():
    """Assume you can use any of the bonds B1, B2 and B3 to build a duration matched bond portfolio matched to the liabilities.
    Which combination of 2 bonds can you NOT use to build a duration matched bond portfolio?"""

    pass


def question_10():
    """Assuming the same liabilities as the previous questions (i.e. a sequence of 3 liabilities of 100,000, 200,000 and 300,000 that are 3, 5
    and 10 years away, respectively), build a Duration Matched Portfolio of B2 and B3 to match the liabilities.
    What is the weight of B2 in this portfolio?"""

    liabilities = pd.Series(data=[100000, 200000, 300000], index=[3, 5, 10])
    short_bond = erk.bond_cash_flows(5, 1000, .05, 4)
    long_bond = erk.bond_cash_flows(10, 1000, .05, 1)
    w_s = erk.match_durations(liabilities, short_bond, long_bond, 0.05)

    print("Q10: Assuming the same liabilities as the previous questions (i.e. a sequence of 3 liabilities of 100,000, 200,000 and 300,000 that are "
          "3, 5 and 10 years away, respectively), build a Duration Matched Portfolio of B2 and B3 to match the liabilities."
          "What is the weight of B2 in this portfolio? {}".format(w_s))


question_1_2_3()
# question_4_5_6()
# question_7()
# question_8()
# question_9()
question_10()
