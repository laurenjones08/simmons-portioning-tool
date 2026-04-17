from numpy_compat import ensure_numpy_legacy_aliases

ensure_numpy_legacy_aliases()

from pyomo.environ import *


def build_model(
    P_set, T_set, K_set, L_set, B_set, M_set,
    WIP, D_week1, monthly_contract, Y, V, R, H,
    bucket_of_k,
    line_of_k,
    month_of_day,
    week1_dates,
    line_throughput,
    big_allowed,
    small_allowed,
    gamma=1.0
):
    """
    P_set              : iterable of SKUs p
    T_set              : iterable of production dates t
    K_set              : iterable of line-specific decisions k
    L_set              : iterable of lines l
    B_set              : iterable of WIP buckets b
    M_set              : iterable of months m

    WIP[b,t]           : lbs breast available in bucket b on date t
    D_week1[p,t]       : short-term daily demand for week 1 only
    monthly_contract[p,m] : monthly contractual demand for sku p in month m
    Y[p,k]             : lbs of sku p produced by decision k per lb assigned
    V[k]               : value per lb of decision k
    R[k]               : lbs/hour for decision k
    H[l,t]             : hours available on line l, date t
    bucket_of_k[k]     : bucket assigned to decision k
    line_of_k[k]       : line assigned to decision k
    month_of_day[t]    : month associated with date t
    week1_dates        : dates in the short-term week
    line_throughput[l] : total lbs/hour capacity of line l

    gamma              : penalty on week 1 unmet / over-production deviation
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
    m.M = Set(initialize=list(M_set), ordered=True)

    m.Week1Dates = Set(initialize=list(week1_dates), within=m.T, ordered=True)

    # =========================
    # Parameters
    # =========================
    m.WIP = Param(m.B, m.T, initialize=WIP, within=NonNegativeReals)
    m.D_week1 = Param(m.P, m.T, initialize=D_week1, default=0.0, within=NonNegativeReals)
    m.monthly_contract = Param(m.P, m.M, initialize=monthly_contract, default=0.0, within=NonNegativeReals)
    m.Y = Param(m.P, m.K, initialize=Y, within=NonNegativeReals)
    m.V = Param(m.K, initialize=V, within=Reals)
    m.R = Param(m.K, initialize=R, within=PositiveReals)
    m.H = Param(m.L, m.T, initialize=H, within=NonNegativeReals)
    m.bucket_of_k = Param(m.K, initialize=bucket_of_k, within=m.B)
    m.line_of_k = Param(m.K, initialize=line_of_k, within=m.L)
    m.line_throughput = Param(m.L, initialize=line_throughput, within=PositiveReals)

    m.big_allowed = Param(m.P, initialize=big_allowed, within=Binary)
    m.small_allowed = Param(m.P, initialize=small_allowed, within=Binary)

    # month_of_day is easier as a plain Python dict for filtering
    month_of_day_dict = dict(month_of_day)

    # =========================
    # Decision Variables
    # =========================
    # x[k,t] = lbs assigned to line-specific decision k on date t
    m.x = Var(m.K, m.T, domain=NonNegativeReals)

    # prod[p,t] = lbs of sku p produced on date t
    m.prod = Var(m.P, m.T, domain=NonNegativeReals)

    # dev[p,t] = |prod[p,t] - D_week1[p,t]| for week 1 only
    m.dev = Var(m.P, m.Week1Dates, domain=NonNegativeReals)

    # prod_big[p,t] = lbs of sku p produced from big bird on date t
    m.prod_big = Var(m.P, m.T, domain=NonNegativeReals)

    # prod_small[p,t] = lbs of sku p produced from small bird on date t
    m.prod_small = Var(m.P, m.T, domain=NonNegativeReals)

    # =========================
    # Constraints
    # =========================

    # 1) WIP availability by bucket and date
    def bucket_wip_rule(m, b, t):
        relevant_k = [k for k in m.K if value(m.bucket_of_k[k]) == b]

        if not relevant_k:
            return Constraint.Skip

        return sum(m.x[k, t] for k in relevant_k) <= m.WIP[b, t]

    m.BucketWIPConstraint = Constraint(m.B, m.T, rule=bucket_wip_rule)

    # 2) Line-hour capacity by line and date
    def line_capacity_rule(m, l, t):
        return sum(
            m.x[k, t] / m.R[k]
            for k in m.K
            if value(m.line_of_k[k]) == l
        ) <= m.H[l, t]

    m.LineCapacityConstraint = Constraint(m.L, m.T, rule=line_capacity_rule)

    # 3) Line throughput capacity by line and date
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

    # 5) Week 1 demand coverage
    # If you want hard fulfillment of short-term demand, keep this.
    def week1_demand_rule(m, p, t):
        return m.prod[p, t] >= m.D_week1[p, t]

    m.Week1DemandConstraint = Constraint(m.P, m.Week1Dates, rule=week1_demand_rule)

    # 6) Monthly contract fulfillment
    def monthly_contract_rule(m, p, month):
        days_in_month = [t for t in m.T if month_of_day_dict[t] == month]

        if not days_in_month:
            return Constraint.Skip

        return sum(m.prod[p, t] for t in days_in_month) >= m.monthly_contract[p, month]

    m.MonthlyContractConstraint = Constraint(m.P, m.M, rule=monthly_contract_rule)

    # 7) Absolute deviation linearization for week 1 only
    def dev_pos_rule(m, p, t):
        return m.dev[p, t] >= m.prod[p, t] - m.D_week1[p, t]

    m.DevPosConstraint = Constraint(m.P, m.Week1Dates, rule=dev_pos_rule)

    def dev_neg_rule(m, p, t):
        return m.dev[p, t] >= m.D_week1[p, t] - m.prod[p, t]

    m.DevNegConstraint = Constraint(m.P, m.Week1Dates, rule=dev_neg_rule)

    # 8) Big/Small Bird Split
    def production_split_rule(m, p, t):
        return m.prod[p, t] == m.prod_big[p, t] + m.prod_small[p, t]

    m.ProductionSplitConstraint = Constraint(m.P, m.T, rule=production_split_rule)

    def big_eligibility_rule(m, p, t):
        return m.prod_big[p, t] <= 1000000 * m.big_allowed[p]

    m.BigEligibilityConstraint = Constraint(m.P, m.T, rule=big_eligibility_rule)

    def small_eligibility_rule(m, p, t):
        return m.prod_small[p, t] <= 1000000 * m.small_allowed[p]

    m.SmallEligibilityConstraint = Constraint(m.P, m.T, rule=small_eligibility_rule)
    # =========================
    # Objective Function
    # =========================
    def objective_rule(m):
        production_value = sum(
            m.V[k] * m.x[k, t]
            for k in m.K
            for t in m.T
        )

        week1_deviation_penalty = gamma * sum(
            m.dev[p, t]
            for p in m.P
            for t in m.Week1Dates
        )

        return production_value - week1_deviation_penalty

    m.Obj = Objective(rule=objective_rule, sense=maximize)

    return m
