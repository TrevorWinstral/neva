"""High(er) level utilities to automate some Neva's common tasks."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import lambertw

from . import BankingSystem, ibeval


def shock_and_solve(b_sys, equity_delta, method, solve_assets=True, **kwargs):
    """Shock the equities of banks and compute the fixed point of equities.

    In order to keep balance sheets consistent, shocks in equity correspond to
    an equal shock in external assets. The fixed point of equities incorporates
    losses of all rounds. The (intermediate) equities for all rounds are also
    saved (in `b_sys.history`). If balance sheets imply that losses in the
    (fixed point) equities appear even without any shock, if is possible to
    adjust external assets and their volatility to remove such effect.

    Parameters:
        b_sys (BankingSystem): banking system to shock
        equity delta (sequence): equity shocks to banks
        method (str): how to compute the fixed point equities, the currently
                      supported options are: `exante_en_blackcox_gbm` for
                      ex-ante valuation with banks that can default before
                      maturity and external assets following a Geometric
                      Brownian Motion; `exante_en_merton_gbm`, as the previous
                      method with banks that can default only at maturity;
                      `exante_furfine_merton_gbm`, as the previous
                      method with a simplified endogenous recovery rate;
                      `eisenberg_noe` for the Eisenberg and Noe model;
                      `linear_dr` for the Linear DebtRank model.
        solve_assets (bool): if `True` external assets and volatilities are
                             adjusted such that without shocks the fixed point
                             equities are equal to the initial equities
        **kwargs (dict): additional method-specific parameters; e.g. some
                         methods allow the `recovery_rate` sequence to specify
                         the (possibly heterogenous) recovery rates of banks
    """

    # dispatching parameters
    if (
        method == "exante_en_blackcox_gbm"
        or method == "exante_en_merton_gbm"
        or method == "exante_furfine_merton_gbm"
    ):
        if "recovery_rate" in kwargs:
            recovery_rate = kwargs["recovery_rate"]
        else:
            recovery_rate = [0 for _ in b_sys]

    if method == "linear_dr":
        equity_init = [bnk.get_naiveequity() for bnk in b_sys]

    # solving for extarnal assets and their volatility
    if solve_assets:
        for idx, bnk in enumerate(b_sys):
            if method == "exante_en_blackcox_gbm":
                bnk.ibeval = lambda ae, bnk=bnk, rr=recovery_rate[
                    idx
                ]: ibeval.exante_en_blackcox_gbm(bnk.equity, ae, rr, bnk.sigma_asset)
            elif method == "exante_en_merton_gbm":
                bnk.ibeval = lambda ae, bnk=bnk, rr=recovery_rate[
                    idx
                ]: ibeval.exante_en_merton_gbm(
                    bnk.equity, ae, bnk.ibliabtot + bnk.extliab, rr, bnk.sigma_asset
                )
            # this should not have any effect, as the valuation function is
            # constant
            elif method == "eisenberg_noe":
                bnk.ibeval = lambda ae, bnk=bnk: ibeval.eisenberg_noe(
                    bnk.equity, bnk.ibliabtot + bnk.extliab
                )
            elif method == "linear_dr":
                bnk.ibeval = lambda ae, bnk=bnk: ibeval.lin_dr(
                    bnk.equity, equity_init[idx]
                )
            elif method == "exante_furfine_merton_gbm":
                bnk.ibeval = lambda ae, bnk=bnk, rr=recovery_rate[
                    idx
                ]: ibeval.exante_furfine_merton_gbm(bnk.equity, ae, rr, bnk.sigma_asset)
        b_sys.fixedpoint_extasset_sigmaasset()

    # shocking external assets of the same "pound" amount of the equity
    for idx, bnk in enumerate(b_sys):
        bnk.equity -= equity_delta[idx]
        # bnk.equity = max(bnk.equity, 0)
        bnk.extasset -= equity_delta[idx]
        # bnk.extasset = max(bnk.extasset, 0)

    # finding the equity
    b_sys.set_history(True)
    for idx, bnk in enumerate(b_sys):
        if method == "exante_en_blackcox_gbm":
            bnk.ibeval = lambda e, bnk=bnk, rr=recovery_rate[
                idx
            ]: ibeval.exante_en_blackcox_gbm(e, bnk.extasset, rr, bnk.sigma_asset)
        elif method == "exante_en_merton_gbm":
            bnk.ibeval = lambda e, bnk=bnk, rr=recovery_rate[
                idx
            ]: ibeval.exante_en_merton_gbm(
                e, bnk.extasset, bnk.ibliabtot + bnk.extliab, rr, bnk.sigma_asset
            )
        elif method == "eisenberg_noe":
            bnk.ibeval = lambda e, bnk=bnk: ibeval.eisenberg_noe(
                e, bnk.ibliabtot + bnk.extliab
            )
        elif method == "linear_dr":
            bnk.ibeval = lambda e, bnk=bnk: ibeval.lin_dr(e, equity_init[idx])
        elif method == "exante_furfine_merton_gbm":
            bnk.ibeval = lambda e, bnk=bnk, rr=recovery_rate[
                idx
            ]: ibeval.exante_furfine_merton_gbm(e, bnk.extasset, rr, bnk.sigma_asset)
    b_sys.fixedpoint_equity()


def fire_sale(b_sys: BankingSystem, params: list, fast=False, **kwargs):
    """
    Incur the losses incurred by banks fire-selling assets to achieve their
    target leverage.

    To maintain a target leverage ratio, banks sell the least amount of assets
    in order to achieve the closest possible leverage ratio without going
    bankrupt. Banks calculate this by assuming their are the only seller in
    the market and calculating the average sale price for an illiquid asset
    when selling with price. The liquidity of a banks' assets is known by
    that bank.

    Interbank assets are not sold to account for leverage, and leverage is
    calculated using the valuation of the interbanks assets.

    Parameters:
        b_sys (BankingSystem): banking system, must have `save_history=True`
        params (list): parameter list coming from the parse_csv method,
                       assumed to contain the values `target_leverage`, and
                       `liquidity` for each bank.
        fast (bool): boolean to determine is the approximate bound is used,
                     if `False` then a finer optimization is done, to find the
                     optimal amount of external assets to be sold. If `True`
                     first Delta_bar is calculated (Winstral, 2023) and the
                     optimization is performed on the domain [0, Delta_bar].
    """
    # Calculate the leverage achieved after selling D external assets
    def new_leverage(D: float, bnk, liquidity: float):
        """
        Calculate the new leverage which is achieved after selling D
        external assets. D cannot exceed the amount of external assets.

        Parameters
            D (float): amount of external assets sold, less than total
                external assets.
            bnk (Bank): the bank in question
        """
        if D > bnk.extasset:
            raise ValueError(
                (
                    f"The value for D ({D}) exceeds the amount of"
                    " external assets held ({bnk.extasset})."
                )
            )
        if D < 0:
            raise ValueError(f"The value D ({D}) cannot be negative.")
        new_assets = (bnk.extasset - D) * np.exp(
            -1 * liquidity * D
        ) + bnk.get_ibassettot()
        new_equity = bnk.eval_equity()

        if new_equity == 0:  # if you go break-even, your leverage is 1
            return 1
        return new_assets / new_equity

    vnew_leverage = np.vectorize(new_leverage)

    # calculate how much should be sold and store that in bnk._to_sell
    total_sale = 0
    for idx, bnk in enumerate(b_sys):
        Ltot = bnk.extliab + bnk.ibliabtot
        target_leverage = float(params[bnk.name]["target_leverage"])
        liquidity = float(
            params[bnk.name]["liquidity"]
        )  # currently all banks have the same liquidity
        leverage = new_leverage(0, bnk, liquidity)

        # Is the leverage already lower than the target leverage, continue
        if leverage <= target_leverage:
            bnk._to_sell = 0
            continue

        # Is the target leverage achievable
        D_range = np.linspace(
            0, bnk.extasset, max(int(bnk.extasset) + 1, 100)
        )  # possible amount of assets to be sold
        leverage_range = vnew_leverage(
            D_range, bnk, liquidity
        )  # possible leverages which can be achieved
        viable_leverage_range = leverage_range[
            leverage_range >= 0
        ]  # leverages achieved without going bankrupt
        # print(viable_leverage_range)

        # scale by 100 such that values which are within absolutely 0.01 dist^2 (<0.1 dist), are scaled to be within distance 1
        dist_to_target_leverage = 100 * (viable_leverage_range - target_leverage) ** 2
        # argmin returns first occurence of min, so the smallest amount is sold to achieve leverage
        to_sell = D_range[leverage_range >= 0][
            np.argmin(dist_to_target_leverage.astype(int))
        ]
        # print(bnk.extasset, to_sell)

        # setup to have sales incurred simultaneously
        bnk._to_sell = to_sell
        # print(
        #     f"liquidity: {liquidity}\t expected avg price: {(1 - np.exp(-1 * to_sell * liquidity)) / (to_sell * liquidity)}"
        # )
        total_sale += to_sell

    # average price which assets are sold at
    if total_sale > 0:
        average_price = (1 / (total_sale * liquidity)) * (
            1 - np.exp(-liquidity * total_sale)
        )
    else:
        average_price = 1
    # print(f"Actual Average price: {average_price}")
    for idx, bnk in enumerate(b_sys):
        # each bank sells the assets in bnk.to_sell at average_price and puts
        # the cash towards paying liabilities
        bnk.execute_sale(average_price, b_sys)
    for bnk in b_sys:
        bnk.equity = bnk.get_naiveequity()
    if b_sys.save_history:
        b_sys.history.append(b_sys.get_equity())
