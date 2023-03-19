import sys

print(sys.executable)
sys.path.append("/home/trevor/Documents/University/msc/thesis/code/neva/")


import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

try:
    import neva
except ModuleNotFoundError:
    import NEVA.neva as neva
# create perfect CP network with 20 nodes in center, 80 in periphery (corresponds to Netherlands/eMID papers)

# Now create balance sheets
# according to Upper and Worms 2004, use ibassets = 63, ibliabs = 71 for banks
def RAS(
    exposures: np.ndarray,
    target_assets: np.ndarray,
    target_liabs: np.ndarray,
    max_iter=1000,
):
    for i in range(max_iter):
        current_liabs = exposures.sum(axis=-1)
        adjustment_factor = target_liabs / current_liabs
        adjustment_factor[np.isinf(adjustment_factor)] = 1
        exposures = (adjustment_factor * exposures.T).T

        current_assets = exposures.sum(axis=0)
        adjustment_factor = target_assets / current_assets
        adjustment_factor[np.isinf(adjustment_factor)] = 1
        exposures = adjustment_factor * exposures
    return exposures


def add_edges_and_sim(
    illiquidity: float, leverage_target: float = 15, num_steps=40, seed=1
):
    np.random.seed(seed)
    core = 20  # 20
    periphery = 80  # 80
    adj = np.zeros((core + periphery, core + periphery))
    for i in range(core - 1):
        for j in range(i + 1, core):
            # choose a random direction, i -> j or j -> i and make a link
            if np.random.uniform(0, 1) < 0.5:
                adj[i, j] = 1
            else:
                adj[j, i] = 1

    # connect each in the core to 2 distinct random periphery nodes (lending and borrowing)
    for i in range(core):
        j1, j2 = np.random.choice(
            np.arange(core, core + periphery), size=2, replace=False
        )
        adj[i, j1] = 1
        adj[j2, i] = 1

    # if a bank in the periphery has no connections, add a random one to the core
    for j in range(core, core + periphery):
        if (adj[:, j].max() < 1) and (adj[j, :].max() < 1):
            if np.random.uniform(0, 1) < 0.5:
                adj[np.random.choice(np.arange(core)), j] = 1
            else:
                adj[j, np.random.choice(np.arange(core))] = 1
    old_adj = adj.copy()

    # Based on Systemically important exposures (Roncoroni), banks on average have 9.5 interbank assets/liabilities
    # and 400-9.5/361-9.5 external assets/liabilities (billions).
    target_ext_assets = 400 - 9.5
    target_ext_liabs = 361 - 9.5
    target_ib_assets = 9.5
    target_ib_liabs = 9.5
    adj = RAS(
        old_adj,
        np.ones(core + periphery) * target_ib_assets,
        np.ones(core + periphery) * target_ib_liabs,
        max_iter=500,
    )
    np.savetxt("data/ex_exposures.csv", adj, delimiter=",", fmt="%0.3f")

    header = "bank_name,external_asset,external_liabilities,sigma_equity,maturity,liquidity,target_leverage"
    data = []
    tot_ext_assets = np.dot(target_ext_assets, np.ones(core + periphery)).sum()
    liquidity = -1 * np.log(illiquidity) / tot_ext_assets
    for i in range(core + periphery):
        data.append(
            [
                i,
                target_ext_assets,
                target_ext_liabs,
                0.2,
                1.25,
                liquidity,
                leverage_target,
            ]
        )
    np.savetxt(
        "data/ex_balance_sheets.csv",
        np.array(data),
        fmt=["%i"] + ["%0.07f"] * 6,
        header=header,
        delimiter=",",
        comments="",
    )

    results = np.zeros((num_steps + 1, 4, core + periphery))
    results[0] = run_simulation(0)
    for step in range(1, num_steps + 1):
        possible_links = np.argwhere(old_adj == 0)
        # remove diagonal elements
        possible_links = possible_links[possible_links[:, 0] != possible_links[:, 1]]
        i, j = possible_links[np.random.choice(possible_links.shape[0])]
        old_adj[i, j] = 1

        adj = RAS(
            old_adj,
            np.ones(core + periphery) * target_ib_assets,
            np.ones(core + periphery) * target_ib_liabs,
            max_iter=500,
        )
        np.savetxt("data/ex_exposures.csv", adj, delimiter=",", fmt="%0.3f")
        results[step] = run_simulation(step)
    return results


def run_simulation(id):
    bsys, params = neva.parse_csv("data/ex_balance_sheets.csv", "data/ex_exposures.csv")

    # Geometric Browianian Motion on external assets, whose volatility is
    # estimated via the volatility of equities.
    sigma_equity = [float(params[bnk]["sigma_equity"]) for bnk in params]
    bsys = neva.BankingSystemGBMse.with_sigma_equity(bsys, sigma_equity)

    # storing initial equity
    equity_start = bsys.get_equity()
    # plt.plot([b.extasset/equity_start[i] for i,b in enumerate(bsys)])
    # plt.show()

    # shocks to initial equity: 50%
    equity_delta = equity_start[:]
    equity_delta = [e * 0.1 for e in equity_start]

    # running ex-ante Black and Cox, as in [2]
    # with recovery rate equal to 60%
    recovery_rate = [0.6 for _ in bsys]
    neva.shock_and_solve(
        bsys,
        equity_delta,
        "exante_en_blackcox_gbm",
        solve_assets=False,
        recovery_rate=recovery_rate,
    )

    # reading equities after one round and after all rounds
    equity_direct = bsys.history[1]
    equity_neva1 = bsys.history[-1]
    leverage_neva1 = [b.extasset / equity_neva1[i] for i, b in enumerate(bsys)]

    # fire sale
    neva.utils.fire_sale(bsys, params)
    equity_fs1 = bsys.history[-1]
    leverage_fs1 = [b.extasset / equity_fs1[i] for i, b in enumerate(bsys)]

    neva.shock_and_solve(
        bsys,
        [0 for e in equity_start],
        "exante_en_blackcox_gbm",
        solve_assets=False,
        recovery_rate=recovery_rate,
    )

    # reading equities after one round and after all rounds
    equity_neva2 = bsys.history[-1]
    leverage_neva2 = [b.extasset / equity_neva2[i] for i, b in enumerate(bsys)]

    plt.clf()
    plt.plot(equity_start, label="Initial")
    plt.plot(equity_neva1, label="NEVA 1")
    plt.plot(equity_fs1, label="Firesale")
    plt.plot(equity_neva2, label="NEVA 2")
    plt.legend()
    plt.title("Equity after each step")
    plt.savefig(f"figures/equity_profiles/{id}.png")
    plt.clf()
    return np.array([equity_start, equity_neva1, equity_fs1, equity_neva2])


a = run_simulation(0)
r = add_edges_and_sim(0.9, leverage_target=25, num_steps=1)
plt.plot(r[:, 3, :].sum(axis=-1))
plt.title("Final Equity")
plt.show()
