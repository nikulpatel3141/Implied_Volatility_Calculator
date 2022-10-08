import csv
import numpy as np
import numpy as np
from scipy.stats import norm

def black_scholes_model(spot,risk_free_rate,expiry,strike,option_type,vol):
    r = risk_free_rate
    t = expiry

    #calculate relevant quantities for black scholes solution
    pv = strike*np.exp(-r*t)
    d_1 = (np.log(spot/strike) + (r + (vol**2)/2)*t)/(vol*np.sqrt(t))
    d_2 = d_1 - vol*np.sqrt(t)
    
    #use standard results to calculate price and vega according to black scholes
    if option_type == "c":
        price = norm.cdf(d_1)*spot - norm.cdf(d_2)*pv

    if option_type == "p":
        price = norm.cdf(-d_2)*pv - norm.cdf(-d_1)*spot

    vega = norm.pdf(d_1)*spot*np.sqrt(t)

    return price, vega


def interval_bisect(func,l_est,h_est,acc): #interval bisector to find root of func = 0
    print('bisect')
    upper = h_est
    lower = l_est
    err = upper - lower
    if func(upper)*func(lower) > 0:
        print('unsuitable initial guesses')
        return np.nan

    while err > acc:
        mid = (upper + lower)/2
        if func(lower)*func(mid) >= 0:
            lower = mid
        elif func(upper)*func(mid) >= 0:
            upper = mid
        else:
            return np.nan   #an error has occurred
        err = upper - lower
    return mid

def newton_raphson(func,est,acc):   #func is a vector function of price and vega
        diff, vega = func(est)
        h = diff/vega
        n = 0           #set iteration counter
        x = est
        while abs(h) > 1e-9:     #iterate until price corrections are below desired accuracy
            x = x - h
            n += 1
            if n > 10 or abs(h) > 100:
                return np.nan           #if no convergence then stop
        diff, vega = func(x)
        h = diff/vega

        return vol_est

class Vol_Calc:
    def __init__(self,spot,risk_free_rate,expiry,strike,option_type,market_price):
        self.spot = spot
        self.strike = strike
        self.r = risk_free_rate
        self.t = expiry/365         #convert to years
        self.option_type = option_type
        self.market_price = market_price
        self.vol_0 = 0.2       # set initial guess for vol
    
    def check_bounds(self): #check no arbitrage bounds
        if self.option_type == 'c':
            if self.market_price < max(0,self.spot - self.strike*np.exp(-self.r*self.t)) or self.market_price > self.spot:
                return False

        elif self.option_type == 'p':
            if self.market_price < max(0,self.strike*np.exp(-self.r*self.t) - self.market_price) or self.market_price > self.strike*np.exp(-self.r*self.t):
                return False

        else:
            return False

        return True

    def model_diff_vega(self,x): 
        price, vega =  black_scholes_model(self.spot,self.r,self.t,self.strike,self.option_type,x)
        return price - self.market_price , vega

    def vol_solver(self):           #first try N-R failing that interval bisect to find implied vol
        if not self.check_bounds():
            print('No arbitrage bounds not satisfied')
            return np.nan

        diff_vega = lambda x: self.model_diff_vega(x)
        imp_vol = newton_raphson(diff_vega,self.vol_0,1e-7)

        if np.isnan(imp_vol):
            fun = lambda x: diff_vega(x)[0]
            imp_vol = interval_bisect(fun,1e-5,100,1e-7)
            return imp_vol
        else:
            return imp_vol


def main_func():
    print('Hello! This is an implied volatility calculator for vanilla stock options. Please enter the following information:')
    try:
        market_price = float(input('Option market price: '))
        spot = float(input('Current spot price: '))
        strike = float(input('Strike price: '))
        r = float(input('Risk-free rate: '))
        t = float(input('Time to expiry, in days: '))
    
    except ValueError:
        print('Invalid input')
        return None
    
    option_type = input('Call (c) or Put (p) option? ')
    if option_type not in ['c','p'] or spot < 0 or strike < 0 or t < 0 or market_price < 0:
        print('Invalid Input')
        return None

    stock_instance = Vol_Calc(spot,r,t,strike,option_type,market_price)
    imp_vol = stock_instance.vol_solver()
    
    if np.isnan(imp_vol):
        print('Unable to find implied volatility')

    else:
        print('Implied volatility is: {0:4f}'.format(imp_vol)) 

if __name__ == '__main__':
    main_func()


