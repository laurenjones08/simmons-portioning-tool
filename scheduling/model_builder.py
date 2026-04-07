from pyomo.environ import *


def build_model(
    P_set, T_set, K_set, L_set, B_set,
    WIP, D_eff, Y, V, R, H,
    bucket_of_k,
    line_of_k,
    L_delay,
    line_throughput,
    gamma=1.0
):
    """
    P_set   : iterable of SKUs p
    T_set   : iterable of days t
    K_set   : iterable of line-specific decisions k
    L_set   : iterable of lines l
    B_set   : iterable of WIP buckets b

    WIP[b,t]            : lbs breast available in bucket b on day t
    D_eff[p,t]          : effective demand for sku p on day t
    Y[p,k]              : lbs of sku p produced by decision k per lb assigned
    V[k]                : value per lb of decision k
    R[k]                : lbs/hour for decision k
    H[l,t]              : hours available on line l, day t
    bucket_of_k[k]      : bucket assigned to decision k
    line_of_k[k]        : line assigned to decision k
    L_delay[p]          : max allowed delay for sku p
    line_throughput[l]  : total lbs/hour capacity of line l

    gamma               : penalty on |prod[p,t] - D_eff[p,t]|
    """

    m = ConcreteModel()

    # =========================
    # Sets
    # =========================
    m.P = Set(initialize=list(P_set), ordered=True)
    m.T = Set(initialize=list(T_set), ordered=True)
    m.K = Set(initialize=list(K_set), ordered=True)
    m.L = Set(initialize=list(L_set), ordered=True)
    m.B = Set(initialize=list(B_set), ordered=True)

    # =========================
    # Parameters
    # =========================
    m.WIP = Param(m.B, m.T, initialize=WIP, within=NonNegativeReals)
    m.D_eff = Param(m.P, m.T, initialize=D_eff, within=NonNegativeReals)
    m.Y = Param(m.P, m.K, initialize=Y, within=NonNegativeReals)
    m.V = Param(m.K, initialize=V, within=Reals)
    m.R = Param(m.K, initialize=R, within=PositiveReals)
    m.H = Param(m.L, m.T, initialize=H, within=NonNegativeReals)
    m.Lag = Param(m.P, initialize=L_delay, default=0, within=NonNegativeIntegers)
    m.bucket_of_k = Param(m.K, initialize=bucket_of_k, within=m.B)
    m.line_of_k = Param(m.K, initialize=line_of_k, within=m.L)
    m.line_throughput = Param(m.L, initialize=line_throughput, within=PositiveReals)

    # =========================
    # Decision Variables
    # =========================
    # x[k,t] = lbs assigned to line-specific decision k on day t
    m.x = Var(m.K, m.T, domain=NonNegativeReals)

    # prod[p,t] = lbs of sku p produced on day t
    m.prod = Var(m.P, m.T, domain=NonNegativeReals)

    # dev[p,t] = |prod[p,t] - D_eff[p,t]|
    m.dev = Var(m.P, m.T, domain=NonNegativeReals)

    # =========================
    # Constraints
    # =========================

    # 1) WIP availability by bucket and day
    # 1) WIP availability by bucket and day
    def bucket_wip_rule(m, b, t):
        relevant_k = [k for k in m.K if value(m.bucket_of_k[k]) == b]

        if not relevant_k:
            return Constraint.Skip

        return sum(m.x[k, t] for k in relevant_k) <= m.WIP[b, t]

    m.BucketWIPConstraint = Constraint(m.B, m.T, rule=bucket_wip_rule)

    # 2) Line-hour capacity by line and day
    def line_capacity_rule(m, l, t):
        return sum(
            m.x[k, t] / m.R[k]
            for k in m.K
            if value(m.line_of_k[k]) == l
        ) <= m.H[l, t]

    m.LineCapacityConstraint = Constraint(m.L, m.T, rule=line_capacity_rule)

    # 3) Line throughput capacity by line and day
    def line_throughput_rule(m, l, t):
        return sum(
            m.x[k, t]
            for k in m.K
            if value(m.line_of_k[k]) == l
        ) <= m.line_throughput[l] * m.H[l, t]

    m.LineThroughputConstraint = Constraint(m.L, m.T, rule=line_throughput_rule)

    # 4) Production definition
    def production_rule(m, p, t):
        return m.prod[p, t] == sum(
            m.Y[p, k] * m.x[k, t]
            for k in m.K
        )

    m.ProductionConstraint = Constraint(m.P, m.T, rule=production_rule)

    # 5) Delayed demand coverage
    T_list = list(m.T.data())

    def delayed_demand_rule(m, p, t):
        lag = int(value(m.Lag[p]))
        t_idx = T_list.index(t)

        if t_idx < lag:
            return Constraint.Skip

        lhs_days = T_list[:t_idx + 1]
        rhs_days = T_list[:t_idx + 1 - lag]

        return sum(m.prod[p, r] for r in lhs_days) >= sum(m.D_eff[p, r] for r in rhs_days)

    m.DelayedDemandConstraint = Constraint(m.P, m.T, rule=delayed_demand_rule)

    # 6) Absolute deviation linearization
    def dev_pos_rule(m, p, t):
        return m.dev[p, t] >= m.prod[p, t] - m.D_eff[p, t]

    m.DevPosConstraint = Constraint(m.P, m.T, rule=dev_pos_rule)

    def dev_neg_rule(m, p, t):
        return m.dev[p, t] >= m.D_eff[p, t] - m.prod[p, t]

    m.DevNegConstraint = Constraint(m.P, m.T, rule=dev_neg_rule)

    # =========================
    # Objective Function
    # =========================
    def objective_rule(m):
        production_value = sum(
            m.V[k] * m.x[k, t]
            for k in m.K
            for t in m.T
        )

        demand_variability_penalty = gamma * sum(
            m.dev[p, t]
            for p in m.P
            for t in m.T
        )

        return production_value - demand_variability_penalty

    m.Obj = Objective(rule=objective_rule, sense=maximize)

    return m