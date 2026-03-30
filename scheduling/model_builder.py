from pyomo.environ import *

def build_model(P_set, T_set, K_set, L_set,
                WIP, D, Y, V, R, H, A,
                L_delay,
                gamma=1.0, beta=1.0):
    """
    P_set   : iterable of SKUs p
    T_set   : iterable of days t
    K_set   : iterable of decisions k
    L_set   : iterable of lines l

    WIP[t]      : lbs breast available on day t
    D[p,t]      : demand for sku p on day t
    Y[p,k]      : lbs of sku p produced by decision k per lb assigned
    V[k]        : value per lb of decision k
    R[k]        : lbs/hour for decision k
    H[l,t]      : hours available on line l, day t
    A[k,l]      : 1 if line l is preferred for decision k, else 0
    L_delay[p]  : max allowed delay for sku p

    gamma       : penalty on |P_pt - D_pt|
    beta        : penalty on allowing non-preferred line use
    """

    m = ConcreteModel()

    # Sets
    m.P = Set(initialize=list(P_set), ordered=True)
    m.T = Set(initialize=list(T_set), ordered=True)
    m.K = Set(initialize=list(K_set), ordered=True)
    m.L = Set(initialize=list(L_set), ordered=True)


    # Parameters
    m.WIP = Param(m.T, initialize=WIP, within=NonNegativeReals)
    m.D   = Param(m.P, m.T, initialize=D, within=NonNegativeReals)
    m.Y   = Param(m.P, m.K, initialize=Y, within=NonNegativeReals)
    m.V   = Param(m.K, initialize=V, within=Reals)
    m.R   = Param(m.K, initialize=R, within=PositiveReals)
    m.H   = Param(m.L, m.T, initialize=H, within=NonNegativeReals)
    m.A   = Param(m.K, m.L, initialize=A, within=Binary)
    m.Lag = Param(m.P, initialize=L_delay, within=NonNegativeIntegers)


    # Decision Variables
    # X_klt = lbs assigned to decision k on line l on day t
    m.x = Var(m.K, m.L, m.T, domain=NonNegativeReals)

    # P_pt = lbs of sku p produced on day t
    m.prod = Var(m.P, m.T, domain=NonNegativeReals)

    # z_kt = 1 if decision k is allowed to use a non-preferred line on day t
    m.z = Var(m.K, m.T, domain=Binary)

    # dev_pt = |P_pt - D_pt| linearization variable
    m.dev = Var(m.P, m.T, domain=NonNegativeReals)


    # Constraints
    # 1) WIP availability by day
    def wip_rule(m, t):
        return sum(m.x[k, l, t] for k in m.K for l in m.L) <= m.WIP[t]
    m.WIPConstraint = Constraint(m.T, rule=wip_rule)

    # 2) Line-hour capacity by line and day
    def line_capacity_rule(m, l, t):
        return sum(m.x[k, l, t] / m.R[k] for k in m.K) <= m.H[l, t]
    m.LineCapacityConstraint = Constraint(m.L, m.T, rule=line_capacity_rule)

    # 3) Preferred-line logic
    # If A[k,l] = 1, line is preferred and always allowed
    # If A[k,l] = 0, line is non-preferred and only allowed if z[k,t] = 1
    def line_preference_rule(m, k, l, t):
        return m.x[k, l, t] <= m.R[k] * m.H[l, t] * (m.A[k, l] + (1 - m.A[k, l]) * m.z[k, t])
    m.LinePreferenceConstraint = Constraint(m.K, m.L, m.T, rule=line_preference_rule)

    # 4) Production definition
    def production_rule(m, p, t):
        return m.prod[p, t] == sum(m.Y[p, k] * m.x[k, l, t] for k in m.K for l in m.L)
    m.ProductionConstraint = Constraint(m.P, m.T, rule=production_rule)

    # 5) Delayed demand coverage
    # sum_{r=1}^t P_pr >= sum_{r=1}^{t-L_p} D_pr, for t > L_p
    T_list = list(m.T.data())

    def delayed_demand_rule(m, p, t):
        lag = int(value(m.Lag[p]))
        t_idx = T_list.index(t)

        if t_idx < lag:
            return Constraint.Skip

        lhs_days = T_list[:t_idx + 1]
        rhs_days = T_list[:t_idx + 1 - lag]

        return sum(m.prod[p, r] for r in lhs_days) >= sum(m.D[p, r] for r in rhs_days)

    m.DelayedDemandConstraint = Constraint(m.P, m.T, rule=delayed_demand_rule)

    # 6) Absolute deviation linearization for |P_pt - D_pt|
    def dev_pos_rule(m, p, t):
        return m.dev[p, t] >= m.prod[p, t] - m.D[p, t]
    m.DevPosConstraint = Constraint(m.P, m.T, rule=dev_pos_rule)

    def dev_neg_rule(m, p, t):
        return m.dev[p, t] >= m.D[p, t] - m.prod[p, t]
    m.DevNegConstraint = Constraint(m.P, m.T, rule=dev_neg_rule)


    # Objective Function
    def objective_rule(m):
        production_value = sum(m.V[k] * m.x[k, l, t] for k in m.K for l in m.L for t in m.T)
        demand_variability_penalty = gamma * sum(m.dev[p, t] for p in m.P for t in m.T)
        nonpreferred_line_penalty = beta * sum(m.z[k, t] for k in m.K for t in m.T)

        return production_value - demand_variability_penalty - nonpreferred_line_penalty

    m.Obj = Objective(rule=objective_rule, sense=maximize)

    return m