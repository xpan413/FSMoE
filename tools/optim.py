from scipy import optimize, linalg
import numpy as np
from scipy.optimize import leastsq, differential_evolution, NonlinearConstraint
import math


class Optimizer:
    def __init__(self, alpha, beta, a, l, others=0) -> None:
        ###a2a,ag,exp,rs,gar
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.tmp_state = False
        d0, t0 = self.optimize_degree(0)
        if (self.alpha[0] + self.beta[0] * self.a[0]) < 1:
            d0 = 1
        _ = self.get_overlap_part(d0)
        self.foward_state = False
        if self.exp_gap > (self.alpha[1] + self.beta[1] * self.a[1] / d0):
            self.foward_state = True
        self.d0 = d0
        self.t0 = t0
        self.alpha[2] *= 2
        self.beta[2] *= 2
        d1, t1 = self.optimize_degree(0)
        if d0 == 1:
            d1 = 1
        self.d1 = d1
        self.t1 = t1
        self.gap = self.get_overlap_part(d1)
        self.l = l
        self.others = others

    def gar_time(self, g):
        if g < 1e-5:
            return 0
        return self.alpha[4] + g * self.beta[4]

    def get_stored_grad(self, grads, t_olps):
        # grads : pre-layer grad, length is layer - 1
        # t_olps: overlaped time per layer,length equal with grads
        # supposed that the first one is dense layer and follows dense,moe, dense, moe ...
        l = len(grads)
        stack_v = []
        stack_idx = []

        stack_olp_v = []

        for i in range(l):
            # print(gar_time(grads[i])-t_olps[i])
            if self.gar_time(grads[i]) > t_olps[i]:
                tmp = self.gar_time(grads[i]) - t_olps[i]
                if t_olps[i] > self.alpha[4]:
                    stack_olp_v.append((t_olps[i] - self.alpha[4]) / self.beta[4])

                    stack_v.append((tmp) / self.beta[4])
                else:
                    stack_olp_v.append(0)
                    stack_v.append(grads[i])
                stack_idx.append(i)
            else:
                delta = t_olps[i] - self.gar_time(grads[i])
                while delta > 1e-5 and len(stack_v) > 0:
                    v_tmp = stack_v.pop()
                    if delta > self.gar_time(v_tmp):
                        delta -= self.gar_time(v_tmp)
                        stack_idx.pop()
                    else:
                        tmp = self.gar_time(v_tmp) - delta
                        if delta > self.alpha[4]:
                            stack_v.append((tmp) / self.beta[4])
                        else:
                            stack_v.append(v_tmp)
                        delta = 0
                tmp = t_olps[i] - delta
                if tmp > self.alpha[4]:
                    stack_olp_v.append((tmp - self.alpha[4]) / self.beta[4])
                else:
                    stack_olp_v.append(0)

        outs = []
        idxs = []
        tmpi = -1
        for item, idx in zip(stack_v, stack_idx):

            if idx % 2 == 1:
                if tmpi != -1:
                    outs[-1] += item
                    tmpi = -1
                else:
                    outs.append(item)
                    idxs.append(idx)
            else:
                outs.append(item)
                idxs.append(idx)
                tmpi = idx
        return outs, idxs, stack_olp_v

    def get_overlap_part(self, d):
        alpha = self.alpha
        beta = self.beta
        a = self.a
        if (d * alpha[1] + a[1] * beta[1]) > (d * alpha[0] + a[0] * beta[0]):

            def g2(d):
                return (
                    (d - 1) * alpha[1]
                    + (d - 1) / d * a[1] * beta[1]
                    + (d - 1) * alpha[3]
                    + (d - 1) / d * a[3] * beta[3]
                    - d * alpha[2]
                    - a[2] * beta[2]
                )

            if g2(d) > 0:
                self.exp_gap = 0
                return (
                    d * alpha[1]
                    + a[1] * beta[1]
                    + d * alpha[3]
                    + a[3] * beta[3]
                    - 2 * (d - 1) * alpha[0]
                    - 2 * (d - 1) / d * a[0] * beta[0]
                )
            else:
                self.exp_gap = -g2(d)
                return (
                    d * alpha[1]
                    + a[1] * beta[1]
                    + d * alpha[3]
                    + a[3] * beta[3]
                    - 2 * (d - 1) * alpha[0]
                    - 2 * (d - 1) / d * a[0] * beta[0]
                    - g2(d)
                )

        def g2(d):
            return (
                2 * (d - 1) * alpha[0]
                + 2 * (d - 1) / d * a[0] * beta[0]
                - d * alpha[2]
                - a[2] * beta[2]
            )

        if g2(d) > 0:
            self.exp_gap = 0
            return alpha[1] + a[1] / d * beta[1] + alpha[3] + a[3] / d * beta[3]
        else:
            self.exp_gap = -g2(d)
            return (
                d * alpha[2]
                + a[2] * beta[2]
                - 2 * (d - 1) * alpha[0]
                - 2 * (d - 1) / d * a[0] * beta[0]
                + alpha[1]
                + a[1] / d * beta[1]
                + alpha[3]
                + a[3] / d * beta[3]
            )

    def optimize_gar(self, gars):
        if len(gars) == 0:
            return [0] * (self.l - 1)
        if len(gars) == 1:
            return gars
        d_tmp, _ = self.optimize_degree(self.gap + self.beta[4] * (gars[0]))
        if d_tmp == 1:
            return gars
        bounds = []
        sumup = 0

        ys = []
        for item in gars:
            sumup += item
            bounds.append([0, sumup])
            ys.append(sumup)

        # 定义约束条件函数
        def constraint_func(xs):
            constraints = []
            for n in range(1, len(xs) + 1):
                # 检查 xs 的前 n 项和的每个元素是否都小于 ys 的对应元素
                constraint = ys[n - 1] - np.sum(xs[:n])
                constraints.append(constraint)
            return constraints

        Nonlinear_Constraint = NonlinearConstraint(constraint_func, 0, np.inf)

        def obj_func(xs):
            outv = 0
            sumx = 0
            for item in xs:
                sumx += item
                if self.gap > self.alpha[4]:
                    _, v = self.optimize_degree(self.gap + self.beta[4] * (item))
                else:
                    _, v = self.optimize_degree(self.gar_time * (item))
                outv += v
            outv += (sumup - sumx) * self.beta[4]
            return outv

        # xs=[ 5.985e+04,1.104e+05,5.039e+04,1.919e+04,4.139e+05,9.498e+04,2.881e+04,3.581e+04,3.196e+04,1.104e+04,3.279e+05]
        # print(np.array(xs).sum())
        # print(obj_func(xs))
        # print(constraint_func(xs))
        outs = differential_evolution(obj_func, bounds, constraints=Nonlinear_Constraint)
        # print(outs)
        y_tmp = obj_func(gars)

        if outs.fun > (y_tmp - 1e-5):
            return gars
        return outs.x

    def optimize_degree_case2(self, gar):
        # ag bigger than a2a
        alpha = self.alpha
        beta = self.beta
        a = self.a
        self.tmp_state = False

        def check(gs, f, d):

            d = d[0]
            if d < 0:
                return [d], 9999
            d1 = np.ceil(d).astype(np.int64)
            d2 = np.floor(d).astype(np.int64)
            upper = [1, 1, 2, 4, 4, 8, 8, 8, 8]
            downer = [1, 1, 2, 2, 4, 4, 4, 4, 8]
            d1 = d1 if d1 <= 8 else 8
            d2 = d2 if d2 <= 8 else 8
            d1 = upper[d1]
            d2 = downer[d2]
            f1 = True if d1 > 0 else False
            f2 = True if d2 > 0 else False
            for g in gs:
                if f1 and g([d1]) < -1e-5:
                    f1 = False
                if f2 and g([d2]) < -1e-5:
                    f2 = False
            v1 = f([d1]) if f1 else 9999
            v2 = f([d2]) if f2 else 9999
            if v1 < v2:
                return [d1], v1
            return [d2], v2

        def gg(d):
            d = d[0]
            return (d * alpha[1] + a[1] * beta[1]) - (d * alpha[0] + a[0] * beta[0])

        def f1(d):
            d = d[0]
            return (
                2 * alpha[0]
                + 2 / d * a[0] * beta[0]
                + d * alpha[1]
                + a[1] * beta[1]
                + d * alpha[3]
                + a[3] * beta[3]
            )

        def g0(d):
            d = d[0]
            return (
                d * alpha[1]
                + a[1] * beta[1]
                + d * alpha[3]
                + a[3] * beta[3]
                - 2 * (d - 1) * alpha[0]
                - 2 * (d - 1) / d * a[0] * beta[0]
                - gar
            )

        def g1(d):
            d = d[0]
            return d - 1

        def g2(d):
            d = d[0]
            return (
                (d - 1) * alpha[1]
                + (d - 1) / d * a[1] * beta[1]
                + (d - 1) * alpha[3]
                + (d - 1) / d * a[3] * beta[3]
                - d * alpha[2]
                - a[2] * beta[2]
            )

        constraints = [
            dict(type="ineq", fun=gg),
            dict(type="ineq", fun=g0),
            dict(type="ineq", fun=g1),
            dict(type="ineq", fun=g2),
        ]
        d1 = optimize.minimize(f1, (1), method="SLSQP", constraints=constraints).x
        d1, t1 = check([gg, g1, g2, g0], f1, d1)

        def f2(d):
            d = d[0]
            return (
                2 * alpha[0]
                + 2 / d * a[0] * beta[0]
                + alpha[1]
                + a[1] / d * beta[1]
                + alpha[3]
                + a[3] / d * beta[3]
                + d * alpha[2]
                + a[2] * beta[2]
            )

        def g3(d):
            return -g2(d)

        constraints = [
            dict(type="ineq", fun=gg),
            dict(type="ineq", fun=g0),
            dict(type="ineq", fun=g1),
            dict(type="ineq", fun=g3),
        ]
        d2 = optimize.minimize(f2, (2), method="SLSQP", constraints=constraints).x
        d2, t2 = check([gg, g1, g3, g0], f2, d2)

        def ig0(d):
            return -g0(d)

        def f3(d):
            d = d[0]
            return 2 * d * alpha[0] + 2 * a[0] * beta[0] + gar

        def g4(d):
            d = d[0]
            return (
                2 * (d - 1) * alpha[0]
                + 2 * (d - 1) / d * a[0] * beta[0]
                + gar
                - (d) * (alpha[1] + a[1] / d * beta[1] + alpha[3] + a[3] / d * beta[3])
            )

        constraints = [
            dict(type="ineq", fun=gg),
            dict(type="ineq", fun=ig0),
            dict(type="ineq", fun=g1),
            dict(type="ineq", fun=g4),
            dict(type="ineq", fun=g2),
        ]
        d3 = optimize.minimize(f3, (2), method="SLSQP", constraints=constraints).x
        d3, t3 = check([gg, ig0, g1, g4, g2], f3, d3)

        def g41(d):
            d = d[0]
            return (
                2 * (d - 1) * alpha[0]
                + 2 * (d - 1) / d * a[0] * beta[0]
                + gar
                - (
                    alpha[1]
                    + a[1] / d * beta[1]
                    + alpha[3]
                    + a[3] / d * beta[3]
                    + d * alpha[2]
                    + a[2] * beta[2]
                )
            )

        constraints = [
            dict(type="ineq", fun=gg),
            dict(type="ineq", fun=ig0),
            dict(type="ineq", fun=g1),
            dict(type="ineq", fun=g41),
            dict(type="ineq", fun=g3),
        ]
        d31 = optimize.minimize(f3, (2), method="SLSQP", constraints=constraints).x
        d31, t31 = check([gg, ig0, g1, g41, g3], f3, d31)

        def g5(d):
            return -g4(d)

        def g51(d):
            return -g41(d)

        constraints = [
            dict(type="ineq", fun=gg),
            dict(type="ineq", fun=ig0),
            dict(type="ineq", fun=g1),
            dict(type="ineq", fun=g5),
            dict(type="ineq", fun=g2),
        ]
        d4 = optimize.minimize(f2, (2), method="SLSQP", constraints=constraints).x
        d4, t4 = check([gg, ig0, g1, g5, g2], f2, d4)

        constraints = [
            dict(type="ineq", fun=gg),
            dict(type="ineq", fun=ig0),
            dict(type="ineq", fun=g1),
            dict(type="ineq", fun=g51),
            dict(type="ineq", fun=g3),
        ]
        d41 = optimize.minimize(f2, (8), method="SLSQP", constraints=constraints).x
        d41, t41 = check([gg, ig0, g1, g51, g3], f2, d41)

        tt = [t1, t2, t3, t4, t31, t41]

        dd = [d1[0], d2[0], d3[0], d4[0], d31[0], d41[0]]
        # if dist.get_rank()==0:
        #     print(f2([1]))
        #     print(gg([1]))
        #     print(g1([1]))
        #     print(g3([1]))
        #     print(g0([1]))
        #     print(tt)
        #     print(dd)
        idxi = np.array(tt).argmin()
        if idxi in [1, 3, 5]:
            self.tmp_state = True
        return dd[idxi], np.array(tt).min()

    def optimize_degree(self, gar):
        alpha = self.alpha
        beta = self.beta
        a = self.a
        self.tmp_state = False

        def check(gs, f, d):
            if d < 0:
                return d, 9999
            d = d[0]
            d1 = np.ceil(d).astype(np.int64)
            d2 = np.floor(d).astype(np.int64)
            upper = [1, 1, 2, 4, 4, 8, 8, 8, 8]
            downer = [1, 1, 2, 2, 4, 4, 4, 4, 8]
            d1 = d1 if d1 <= 8 else 8
            d2 = d2 if d2 <= 8 else 8

            d1 = upper[d1]
            d2 = downer[d2]
            f1 = True if d1 > 0 else False
            f2 = True if d2 > 0 else False
            for g in gs:
                if f1 and g([d1]) < -1e-5:
                    f1 = False
                if f2 and g([d2]) < -1e-5:
                    f2 = False
            v1 = f([d1]) if f1 else 9999
            v2 = f([d2]) if f2 else 9999
            if v1 < v2:
                return [d1], v1
            return [d2], v2

        def gg(d):
            d = d[0]
            return -(d * alpha[1] + a[1] * beta[1]) + (d * alpha[0] + a[0] * beta[0])

        def f1(d):
            d = d[0]
            return (
                2 * d * alpha[0]
                + 2 * a[0] * beta[0]
                + alpha[1]
                + a[1] / d * beta[1]
                + alpha[3]
                + a[3] / d * beta[3]
            )

        def g0(d):
            d = d[0]
            return alpha[1] + a[1] / d * beta[1] + alpha[3] + a[3] / d * beta[3] - gar

        def g1(d):
            d = d[0]
            return d - 1

        def g2(d):
            d = d[0]
            return (
                2 * (d - 1) * alpha[0]
                + 2 * (d - 1) / d * a[0] * beta[0]
                - d * alpha[2]
                - a[2] * beta[2]
            )

        constraints = [
            dict(type="ineq", fun=gg),
            dict(type="ineq", fun=g0),
            dict(type="ineq", fun=g1),
            dict(type="ineq", fun=g2),
        ]
        d1 = optimize.minimize(f1, (1), method="SLSQP", constraints=constraints).x
        d1, t1 = check([gg, g1, g2, g0], f1, d1)

        def f2(d):
            d = d[0]
            return (
                2 * alpha[0]
                + 2 / d * a[0] * beta[0]
                + alpha[1]
                + a[1] / d * beta[1]
                + alpha[3]
                + a[3] / d * beta[3]
                + d * alpha[2]
                + a[2] * beta[2]
            )

        def g3(d):
            return -g2(d)

        constraints = [
            dict(type="ineq", fun=gg),
            dict(type="ineq", fun=g0),
            dict(type="ineq", fun=g1),
            dict(type="ineq", fun=g3),
        ]
        d2 = optimize.minimize(f2, (4), method="SLSQP", constraints=constraints).x
        d2, t2 = check([gg, g1, g3, g0], f2, d2)

        def ig0(d):
            return -g0(d)

        def f3(d):
            d = d[0]
            return 2 * d * alpha[0] + 2 * a[0] * beta[0] + gar

        def g4(d):
            return g2(d) - g0(d)

        constraints = [
            dict(type="ineq", fun=gg),
            dict(type="ineq", fun=ig0),
            dict(type="ineq", fun=g1),
            dict(type="ineq", fun=g4),
        ]
        d3 = optimize.minimize(f3, (2), method="SLSQP", constraints=constraints).x
        d3, t3 = check([gg, ig0, g1, g4], f3, d3)

        def g5(d):
            return -g2(d) + g0(d)

        constraints = [
            dict(type="ineq", fun=gg),
            dict(type="ineq", fun=ig0),
            dict(type="ineq", fun=g1),
            dict(type="ineq", fun=g5),
        ]
        d4 = optimize.minimize(f2, (2), method="SLSQP", constraints=constraints).x
        d4, t4 = check([gg, ig0, g1, g5], f2, d4)

        d5, t5 = self.optimize_degree_case2(gar)

        tt = [t1, t2, t3, t4, t5]
        dd = [d1[0], d2[0], d3[0], d4[0], d5]
        idxi = np.array(tt).argmin()

        if idxi in [1, 3]:
            self.tmp_state = True
        return dd[np.array(tt).argmin()], np.array(tt).min()

    def measure_others(self, b, s, m, h, heads):
        if self.others is not None:
            return (
                self.others
                + self.alpha[1]
                + self.beta[1] * (b * s * m)
                + self.alpha[3]
                + self.beta[3] * (b * s * m)
            )

        alpha_dict = {}
        alpha_dict[512] = 0.7674542565
        alpha_dict[1024] = 6.4634952121e-1
        alpha_dict[256] = 9.061728879e-1
        beta_dict = {}
        beta_dict[512] = 1.7435315251e-10
        beta_dict[1024] = 2.5884462453e-10
        beta_dict[256] = 1.6690605555e-10
        alpha = alpha_dict[s]
        beta = beta_dict[s]
        a = (
            b * s * m * m // 4 * 3
            + b * heads // 4 * s * s * m // heads
            + b * heads // 4 * s * s * m // heads
            + b * s * m // 4 * m
        )
        return (
            (2 * alpha + 2 * a * beta)
            + self.alpha[1]
            + self.beta[1] * (b * s * m)
            + self.alpha[3]
            + self.beta[3] * (b * s * m)
        )

    def time_grad(self, time):
        if time > self.alpha[4]:
            return (time - self.alpha[4]) / self.beta[4]
        return 0

    def run(self, b, s, m, h, heads, mp_size=1, att_type=0):  # att_type==1->mixtral,==0:gpt

        if att_type == 1:
            grad = m * m // mp_size * 3 // 2 + m * m // mp_size
        else:
            grad = m * m // mp_size * 3 + m // mp_size * 3 + m * m // mp_size + m
        l = self.l
        others = self.measure_others(b, s, m, h, heads)

        gars, idxs, stack_olp_v = self.get_stored_grad(
            [grad, 0] * (l - 1), [others, self.gap] * (l - 1)
        )
        self.gflag = False
        self.agflag = False

        if len(gars) == 0:
            if self.exp_gap < self.others + 5:
                exp_gap = 0
            else:
                exp_gap = 0.7 * self.exp_gap

            t_grad = self.gar_time(grad)
            g1 = min(t_grad, exp_gap)
            g2 = min(max(0, t_grad - exp_gap), others)
            g3 = max(0, t_grad - g1 - g2)

            if exp_gap > 1.8 * t_grad:
                self.gflag = True
            grad_moe = self.time_grad(g1 + g3)
            grad_other = self.time_grad(g2)
            stack_olp_v = [grad_other, grad_moe] * (l - 1)
            if g2 < self.others:
                self.agflag = True
        self.gar = gars

        xs = self.optimize_gar(gars)

        def myround(x):
            return math.ceil(x) + math.ceil(x) % 2

        xs = np.array(xs)
        xs_moe_reverse = xs + myround(stack_olp_v[1])
        final_xs_moe = []
        for i in range(l - 1):
            final_xs_moe.append(int(xs_moe_reverse[l - 2 - i]))

        tmp = myround(stack_olp_v[0])
        final_xs_att = [int(grad * (l - 1) - np.array(final_xs_moe).sum() - tmp * (l - 2))] + [
            tmp
        ] * (l - 2)

        degrees = []
        for item in final_xs_moe:
            d, _ = self.optimize_degree(self.gar_time(item))
            _ = self.get_overlap_part(d)
            self.backward_state = False
            if self.exp_gap > (self.alpha[1] + self.beta[1] * self.a[1] / d):
                self.backward_state = True
            self.backward_state = self.tmp_state
            degrees.append(d)

        return final_xs_moe, final_xs_att, degrees


# gpt-12,m=768,h=3072,seq=2048,batch=8
# 22.89
##################################
######b: batch size
######s: sequence length
######m: model dimension
######h: hidden dimension
######heads: attention heads
######l: number of layers
######time_others: the time cost of the gating and attention part during bp
######alpha_set: [all-to-all, all-gather, GEMM, reduce-scatter, all-reduce] follow t = alpha + workload*beta
######beta_set ...
######moe_factor: topk * capacity_factor
######number_gemm: the number of gemm in a expert


def get_optimal_degree(
    b,
    s,
    m,
    h,
    heads,
    l,
    time_others,
    es_size,
    mp_size,
    alpha_set,
    beta_set,
    moe_factor,
    number_gemm=2,
):
    factor = moe_factor
    obj = Optimizer(
        others=time_others,
        l=l,
        alpha=alpha_set,
        beta=beta_set,
        a=[
            factor * b * s * m / es_size,
            factor * b * s * m,
            factor * b * s * m * h / es_size * number_gemm,
            factor * b * s * m,
        ],
    )
    final_xs_moe, final_xs_att, degrees = obj.run(b, s, m, h, heads, mp_size=8, att_type=0)
    print("forward degree:" + str(obj.d0))
    print("back degrees per layer:" + str(degrees + [obj.d1]))


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--L", type=int, default=1024)
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--H", type=int, default=14336)
    parser.add_argument("--number_gemm", type=int, default=3)
    parser.add_argument("--att_type", type=int, default=1)  # 1:mixtra,0:gpt
    parser.add_argument("--f", type=float, default=2.4)  # topk * capacity_factor
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--es_size", type=int, default=8)
    parser.add_argument("--mp_size", type=int, default=8)
    parser.add_argument("--layer", type=int, default=2)  # number of layers
    parser.add_argument(
        "--t_others", type=float, default=1.3
    )  # time cost of gating and attention in bp
    args = parser.parse_args()

    ### need to set
    alpha_set = [
        2.3515350484e-01,
        0.0549,
        5.9194353200e-01,
        0.0688,
        3.4099276392e-01,
    ]  # [all-to-all, all-gather, GEMM, reduce-scatter, all-reduce] follow t = alpha + workload*beta
    beta_set = [1.8543323833e-06, 6.72388e-7, 5.6847387697e-11, 6.70433e-7, 3.4950701465e-06]

    get_optimal_degree(
        args.B,
        args.L,
        args.M,
        args.H,
        args.heads,
        args.layer,
        args.t_others,
        args.es_size,
        args.mp_size,
        alpha_set,
        beta_set,
        args.f,
        args.number_gemm,
    )
