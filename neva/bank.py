"""Bank represented through its balance sheet and valuation functions.

The balance sheet of the bank consists of assets and liabilities, both of
which can be either external or interbank. Interbank assets and liabilites are 
towards other banks:

    A_ij : money that bank i loaned to bank j, 

which is an asset for bank i. A corresponding liability L_ji = A_ij exists in 
the balance sheet of bank j, so that:

    L_i^{ib} = \sum_j L_ij: aggregate interbank liabilities of bank i.

External assets and liabilities are towards entities not belonging to the 
banking system:

    A_i^e : aggregate external assets of bank i,
    L_i^e : aggregate external liabilities of bank i.

Therefore, the face (or book) value of the equity of bank i is:

    E_i = A_i^e - L_i^e + \sum_j A_ij - L_i^{ib} .

Liabilities have the following priorities (from the most senior to the most 
junior): external, interbank, equity. The mark-to-market valuation of external 
assets of bank i is: 

    A_i^e * V_i^e(E_i) ,

where V_i^e (E_i) is the external valuation function for bank i and it takes 
values in the real interval [0, 1]. Similarly, the mark-to-market valuation of
interbank assets is:

    A_ij * V_ij(E_i, E_j) , 

where V_ij(E_i, E_j) is the full interbank valuation function for the couple 
of banks i and j and it also takes values in the real interval [0, 1]. Here 
the full interbank valuation function is factorised in following way: 

    V_ij(E_i, E_j) = ibeval_lender(E_i) * ibeval(E_j) ,

the lender interbank valuation function and the (proper) interbank valuation 
function. The former captures the dependence of the valuation of interbank 
assets on the equity of the lender, while the latter on the equity of the 
borrower.

The valuation of the equities is obtained as fixed point of the map: 

    E = \Phi(E) ,

where: 

    \Phi(E_i) =  A_i^e * V_i^e(E_i) - L_i^e +
                 + \sum_j A_ij * V_ij(E_i, E_j) - L_i^{ib} .

References:
    [1] P. Barucca, M. Bardoscia, F. Caccioli, M. D'Errico, G. Visentin, 
        G. Caldarelli, S. Battiston. Network Valuation in Financial Systems, 
        Mathematical Finance, https://doi.org/10.1111/mafi.12272
"""

from __future__ import division

import numbers


# pylint: disable=too-many-instance-attributes
class Bank(object):
    """Bank with balance sheet and valuation functions.

    Attributes:
        name (str): name of the bank
        extasset (float): external assets
        extliab (float): external liabilities
        exteval (fun(equity)): valuation function for external assets
        ibasset (sequence of tuples (`Bank`, float)): interbank assets
        ibliabtot (float): total interbank liabilities
        ibeval (fun(equity)): valuation function for interbank assets
        ibeval_lender (fun(equity)): lender valuation function for interbank
                                     assets
        equity (float): equity
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        extasset=0,
        extliab=0,
        exteval=(lambda e, ea, el: ea - el),
        ibliabtot=0,
        ibeval=(lambda e: 1),
        ibeval_lender=(lambda e: 1),
        equity=None,
        name=" ",
    ):
        """
        Create a bank.

        Parameters:
            extasset (float): external assets
            extliab (float): external liabilities
            exteval (fun(equity)): valuation function for external assets
            ibliabtot (float): total interbank liabilities
            ibeval (fun(equity)): valuation function for interbank assets
            ibeval_lender (fun(equity)): lender valuation function for
                                         interbank assets
            equity (float, optional): equity, if not provided it is set to its
                                      face value
            name (str): name of the bank
        """
        if not isinstance(extasset, numbers.Number):
            raise TypeError
        self.extasset = extasset
        if not isinstance(extliab, numbers.Number):
            raise TypeError
        self.extliab = extliab
        self.exteval = exteval
        self.ibasset = []
        if not isinstance(ibliabtot, numbers.Number):
            raise TypeError
        self.ibliabtot = ibliabtot
        self.ibeval = ibeval
        self.ibeval_lender = ibeval_lender
        if equity is None:
            self.equity = self.get_naiveequity()
        else:
            self.equity = equity
        self.name = name
        self._to_sell = None  # for synchronous fire sales

    @property
    def to_sell(self):
        return self._to_sell

    @to_sell.setter
    def set_to_sell(self, to_sell):
        if to_sell < 0:
            raise ValueError(f"The value to be sold ({to_sell}) must be non-negative.")
        if to_sell > self.extasset:
            raise ValueError(
                f"The value to be sold ({to_sell}) may not exceed the amount of external assets held."
            )
        if not isinstance(to_sell, (int, float)):
            raise TypeError(
                f"The type to be sold ({type(to_sell)}) must be an int or float."
            )
        self._to_sell = to_sell

    def execute_sale(self, price, b_sys):
        """Sell the amount of assets in the `to_sell variable`  at the price `price`.

        To sell the amount of assets, take the assets the bank currently posseses,
        sell the amount designated by `to_sell` at the price `price`. The cash
        from this sale is put towards first buying back external liabilities,
        then towards buying back interbank liabilities in a uniform fashion.
        The remaining assets are marked down by a factor 1/price, to account
        for the drop in price due to illiquidity.

        Parameters:
            price (float): the price to be sold at, given as a fraction of the
                face value, in [0,1].
            b_sys (BankingSystems): The banking system the given bank is in.
        """
        if not isinstance(price, (int, float)):
            raise TypeError(
                (
                    "The selling price must be of type int or float, "
                    f"but is of type {type(price)}."
                )
            )
        if (price > 1) or (price < 0):
            raise ValueError(
                (
                    "The selling price must be fraction of the"
                    "original price, i.e. in [0,1], but price"
                    f"is equal to {price}."
                )
            )

        # 1: reduce assets by amount sold, and multiply leftover by the
        # slippage factor (average price)
        new_assets = self.extasset - self._to_sell
        cash = self._to_sell * price
        new_asset_value = new_assets * price
        self.extasset = new_asset_value

        if (
            cash <= self.extliab
        ):  # if the returned cash is less than external liabilities
            self.extliab -= cash

        # 2: reduce liabilities by recuperated cash, first external liabilities,
        # then uniformly across all IB liabilities
        else:  # if we have to buyback interbank liabilities
            cash -= self.extliab  # cash after buying back external liabilities
            self.extliab = 0
            ibmatrix = b_sys.get_ibasset_matrix()
            percent_to_buy = min(cash / self.ibliabtot, 1)
            bnk_to_id = {b_sys[bnk_id]: bnk_id for bnk_id in b_sys.banksbyid}
            bnk_id = bnk_to_id[self]
            # go through the b_sys, if a bank j is a creditor to this bank
            # i.e. if ibmatrix[j][bnk_id] > 0, then reduce this by the percent
            # to buy
            for j, lender in enumerate(b_sys.banks):
                if hasattr(lender, "ibasset"):
                    for i in range(len(lender.ibasset)):
                        borrower, amount = lender.ibasset[i]
                        if bnk_to_id[borrower] == bnk_id:
                            lender.ibasset[i] = (
                                borrower,
                                amount * (1 - percent_to_buy),
                            )
                            lender.extasset += amount * percent_to_buy
            self.ibliabtot *= 1 - percent_to_buy

        pass

    def set_ibasset(self, ibasset):
        """Set interbank assets.

        Interbank assets are represented as a sequence of 2-tuples whose first
        element is a `Bank` object (the borrower) and whose second element is
        the face value of the interbank asset.

        Parameters:
            ibasset (sequence of tuples (`Bank`, float)): interbank assets.
        """
        if not isinstance(ibasset, (list, tuple)):
            raise TypeError
        for item in ibasset:
            if not isinstance(item, (list, tuple)):
                raise TypeError
            else:
                if (not isinstance(item[0], Bank)) or (
                    not isinstance(item[1], numbers.Number)
                ):
                    raise TypeError
        self.ibasset = ibasset
        self.equity = self.get_naiveequity()

    def get_ibassettot(self):
        """Return the total face value of interbank assets."""
        return sum((ibasset for borrower, ibasset in self.ibasset))

    def get_naiveequity(self):
        """Return the face value of equity.

        Such value corresponds to both interbank and external assets taken at
        their face value, i.e. without any devaluation occurring. Therefore,
        it is the largest possible value for the equity.
        """
        return self.extasset - self.extliab + self.get_ibassettot() - self.ibliabtot

    def get_leastequity(self):
        """Return the least possible value of equity.

        Such value correponds to a complete devaluation of all assets, i.e. to
        interbank and external assets having a null mark-to-marked valuation.
        """
        return -self.extliab - self.ibliabtot

    def eval_equity(self):
        """Return the one-step valuation of equity.

        The equity is computed here as a single step of the iterative map for
        the bank. The equity valution should be performed by iterating the map
        for all banks, until convergence.

        Note:
            The equity of the bank is returned and not stored in `Bank.equity`
            because the map is updated syncronously.
        """
        # print(self.name, self.ibasset)
        # print(self.name, self.extasset)
        # for borrower, ibasset in self.ibasset:
        #    print(self.name, borrower.name,
        #          borrower.ibeval(borrower.equity),
        #          borrower.extasset,
        #          borrower.equity,
        #          borrower.sigma_asset)

        return (
            self.exteval(self.equity, self.extasset, self.extliab)
            + self.ibeval_lender(self.equity)
            * sum(
                (
                    ibasset * borrower.ibeval(borrower.equity)
                    for borrower, ibasset in self.ibasset
                )
            )
            - self.ibliabtot
        )

    def __str__(self):
        """Print info about the bank's balance sheet."""
        template = (
            "bank name: {name}\n"
            "external assets: {extasset}\n"
            "external liabilities: {extliab}\n"
            "interbank assets: {ibasset}\n"
            "interbank liabilities: {ibliab}\n"
            "equity (naive): {equity} ({naive})"
        )
        return template.format(
            name=self.name,
            extasset=self.extasset,
            extliab=self.extliab,
            ibasset=self.get_ibassettot(),
            ibliab=self.ibliabtot,
            equity=self.equity,
            naive=self.get_naiveequity(),
        )
