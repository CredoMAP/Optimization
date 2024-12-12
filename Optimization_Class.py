import numpy as np
import pandas as pd
import scipy.optimize as sco
import math

# Class for optimization for GMV ('gmv'), Max Sharpe ('max_sharpe'), Max Diversification ('max_div'),
# and Max Decorrelation ('max_dec'), Max Return ('max_return') and Tracking Error ('tracking_error').

# User can choose frequency for the data, max volatility, min and max weights for each asset and whether to use
# shrinkage or not on covariance matrix. The shrinkage estimates are calculated and implemented following Ledoit Wolf
# 2004 paper (https://github.com/oledoit/covShrinkage/blob/main/covCor.m).
# Class has two options: linear shrinkage towards constant-correlation matrix ('covCor')
# (the target preserves the diagonal of the sample covariance matrix and all correlation coefficients are the same) and
# nonlinear shrinkage derived under Stein’s loss, called linear-inverse shrinkage ('LIS').
# Second optional parameter k is possible under self._cov_cor(self.returns.to_numpy()) and self._LIS(self.returns),
# if k is absent - function demeans the data, if k = 0 the no demeaning takes place and if k = 1 then data Y
# has already been demeaned.

# Examples of usage:
# from Optimization_Class import PortfolioOptimizer
# optimizer = PortfolioOptimizer(to_test, freq='daily')
# weights_gmv = optimizer.optimize(method='max_sharpe', min_weights=[0.0]*len(to_test.columns), max_weights=[0.10, 0.5],
# max_volatility=0.15, robust=True, shrinkage_type='covCor')


class PortfolioOptimizer:
    def __init__(self, returns, freq='daily'):
        """
        Initialize the PortfolioOptimizer class with the given returns and frequency.

        Parameters:
        returns (pandas.DataFrame): A DataFrame containing the historical returns of assets.
        freq (str, optional): The frequency of the returns' data. Defaults to 'daily'.

        Raises:
        ValueError: If the specified frequency is not recognized.

        Attributes:
        returns (pandas.DataFrame): The historical returns of assets.
        mean_returns (pandas.Series): The mean returns of assets.
        cov_matrix (pandas.DataFrame): The covariance matrix of asset returns.
        num_assets (int): The number of assets in the portfolio.
        annual_factor (int): The annual factor based on the specified frequency.
        """
        self.returns = returns
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.num_assets = len(self.mean_returns)

        if freq == 'daily':
            self.annual_factor = 252
        elif freq == 'weekly':
            self.annual_factor = 52
        elif freq == 'monthly':
            self.annual_factor = 12
        else:
            raise ValueError("Frequency not recognized. Use 'daily', 'weekly', or 'monthly'.")

    def _portfolio_performance(self, weights, cov_matrix=None):
        if cov_matrix is None:
            cov_matrix = self.cov_matrix
        returns = np.dot(weights, self.mean_returns) * self.annual_factor
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(self.annual_factor)
        sharpe_ratio = returns / volatility
        return returns, volatility, sharpe_ratio

    def _neg_sharpe_ratio(self, weights, cov_matrix=None):
        return -self._portfolio_performance(weights, cov_matrix)[2]

    def _min_variance(self, weights, cov_matrix=None):
        return self._portfolio_performance(weights, cov_matrix)[1]

    def _diversification_ratio(self, weights, cov_matrix=None):
        if cov_matrix is None:
            cov_matrix = self.cov_matrix
        asset_vols = np.sqrt(np.diag(cov_matrix)) * np.sqrt(self.annual_factor)
        weighted_vols = np.dot(weights, asset_vols)
        portfolio_vol = self._portfolio_performance(weights, cov_matrix)[1]
        return -weighted_vols / portfolio_vol

    def _tracking_error(self, weights, index_returns):
        portfolio_returns = np.dot(weights, self.returns.T)
        tracking_error = np.std(portfolio_returns - index_returns) * np.sqrt(self.annual_factor)
        return tracking_error

    def _decorrelation(self, weights, cov_matrix=None):
        correlation_matrix = np.corrcoef(self.returns.T)
        weighted_corr = np.dot(weights.T, np.dot(correlation_matrix, weights))
        return weighted_corr

    def _constraint(self, weights):
        return np.sum(weights) - 1

    def _volatility_constraint(self, weights, max_volatility, cov_matrix=None):
        return max_volatility - self._portfolio_performance(weights, cov_matrix)[1]

    def _min_volatility_constraint(self, weights, min_volatility, cov_matrix=None):
        return self._portfolio_performance(weights, cov_matrix)[1] - min_volatility

    def _neg_returns(self, weights, cov_matrix=None):
        return -self._portfolio_performance(weights, cov_matrix)[0]

    def optimize(self, method='gmv', max_volatility=None, min_volatility=None, min_weights=None, max_weights=None,
                 max_single_weight=None, robust=False, shrinkage_type='covCor', index_returns=None):
        if robust:
            if shrinkage_type == 'covCor':
                cov_matrix = self._cov_cor(self.returns.to_numpy())
            elif shrinkage_type == 'LIS':
                cov_matrix = self._LIS(self.returns)
            else:
                raise ValueError("Unknown shrinkage type specified")
        else:
            cov_matrix = self.cov_matrix

        constraints = [{'type': 'eq', 'fun': self._constraint}]
        if max_volatility is not None:
            constraints.append(
                {'type': 'ineq', 'fun': self._volatility_constraint, 'args': (max_volatility, cov_matrix)})
        if min_volatility is not None:
            constraints.append(
                {'type': 'ineq', 'fun': self._min_volatility_constraint, 'args': (min_volatility, cov_matrix)})

        if min_weights is None:
            min_weights = np.zeros(self.num_assets)
        if max_weights is None:
            max_weights = np.ones(self.num_assets)

        if max_single_weight is not None:
            max_weights = np.full(self.num_assets, max_single_weight)

        bounds = tuple((min_weights[i], max_weights[i]) for i in range(self.num_assets))
        initial_guess = np.array(self.num_assets * [1. / self.num_assets])

        if method == 'max_sharpe':
            result = sco.minimize(self._neg_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds,
                                  constraints=constraints, args=(cov_matrix,))
        elif method == 'gmv':
            result = sco.minimize(self._min_variance, initial_guess, method='SLSQP', bounds=bounds,
                                  constraints=constraints, args=(cov_matrix,))
        elif method == 'max_div':
            result = sco.minimize(self._diversification_ratio, initial_guess, method='SLSQP', bounds=bounds,
                                  constraints=constraints, args=(cov_matrix,))
        elif method == 'max_dec':
            result = sco.minimize(self._decorrelation, initial_guess, method='SLSQP', bounds=bounds,
                                  constraints=constraints, args=(cov_matrix,))
        elif method == 'max_return':
            result = sco.minimize(self._neg_returns, initial_guess, method='SLSQP', bounds=bounds,
                                  constraints=constraints, args=(cov_matrix,))
        elif method == 'tracking_error':
            if index_returns is None:
                raise ValueError("index_returns must be provided for tracking error optimization.")
            result = sco.minimize(self._tracking_error, initial_guess, method='SLSQP', bounds=bounds,
                                  constraints=constraints, args=(index_returns,))
        else:
            raise ValueError("Unknown method specified")

        weights = np.round(result.x, 4)
        returns, volatility, sharpe_ratio = self._portfolio_performance(weights, cov_matrix)
        diversification_ratio = -self._diversification_ratio(weights, cov_matrix)

        return {
            'weights': weights,
            'returns': round(returns, 4),
            'volatility': round(volatility, 4),
            'sharpe_ratio': round(sharpe_ratio, 4),
            'diversification_ratio': round(diversification_ratio, 4),
            'tracking_error': round(self._tracking_error(weights, index_returns),
                                    4) if method == 'tracking_error' else None
        }

    def _cov_cor(self, Y, k=None):
        N, p = Y.shape
        if k is None or np.isnan(k) or k == '':
            Y = Y - np.mean(Y, axis=0)
            k = 1
        n = N - k
        sample = np.dot(Y.T, Y) / n
        samplevar = np.diag(sample)
        sqrtvar = np.sqrt(samplevar)
        rBar = (np.sum(sample / (sqrtvar[:, None] * sqrtvar[None, :])) - p) / (p * (p - 1))
        target = rBar * sqrtvar[:, None] * sqrtvar[None, :]
        np.fill_diagonal(target, samplevar)
        Y2 = Y ** 2
        sample2 = np.dot(Y2.T, Y2) / n
        piMat = sample2 - sample ** 2
        pihat = np.sum(piMat)
        gammahat = np.linalg.norm(sample - target, 'fro') ** 2
        rho_diag = np.sum(np.diag(piMat))
        term1 = np.dot(Y.T ** 3, Y) / n
        term2 = samplevar[:, None] * sample
        thetaMat = term1 - term2
        np.fill_diagonal(thetaMat, 0)
        rho_off = rBar * np.sum((1 / sqrtvar[:, None] * sqrtvar[None, :]) * thetaMat)
        rhohat = rho_diag + rho_off
        kappahat = (pihat - rhohat) / gammahat
        shrinkage = max(0, min(1, kappahat / n))
        sigmahat = shrinkage * target + (1 - shrinkage) * sample
        return sigmahat

    def _LIS(self, Y, k=None):
        N = Y.shape[0]
        p = Y.shape[1]

        if k is None or math.isnan(k):
            Y = Y.sub(Y.mean(axis=0), axis=1)
            k = 1

        n = N - k
        c = p / n
        sample = pd.DataFrame(np.matmul(Y.T.to_numpy(), Y.to_numpy())) / n
        sample = (sample + sample.T) / 2

        lambda1, u = np.linalg.eig(sample)
        lambda1 = lambda1.real
        u = u.real

        lambda1 = lambda1.real.clip(min=0)
        dfu = pd.DataFrame(u, columns=lambda1)
        dfu.sort_index(axis=1, inplace=True)
        lambda1 = dfu.columns

        h = (min(c**2, 1/c**2)**0.35) / p**0.35
        invlambda = 1 / lambda1[max(1, p-n+1)-1:p]
        dfl = pd.DataFrame()
        dfl['lambda'] = invlambda
        Lj = dfl[np.repeat(dfl.columns.values, min(p, n))]
        Lj = pd.DataFrame(Lj.to_numpy())
        Lj_i = Lj.subtract(Lj.T)

        theta = Lj.multiply(Lj_i).div(Lj_i.multiply(Lj_i).add(
            Lj.multiply(Lj) * h**2)).mean(axis=0)

        if p <= n:
            deltahat_1 = (1-c) * invlambda + 2 * c * invlambda * theta
        else:
            print("p must be <= n for Stein's loss")
            return -1

        temp = pd.DataFrame(deltahat_1)
        x = min(invlambda)
        temp.loc[temp[0] < x, 0] = x
        deltaLIS_1 = temp[0]

        temp1 = dfu.to_numpy()
        temp2 = np.diag(1/deltaLIS_1)
        temp3 = dfu.T.to_numpy().conjugate()
        sigmahat = pd.DataFrame(np.matmul(np.matmul(temp1, temp2), temp3))
        return sigmahat








