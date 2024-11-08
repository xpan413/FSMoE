#
# Created on Wed Sep 06 2023
#
# Copyright (c) 2023 HPML
#
import argparse
from typing import Tuple, Any
from schemoe import * 
import torch
from torch import nn,Tensor
import torch.distributed as dist
from torch.nn import functional as F
import math
from schemoe.dispatch_combine import reduce_scatter,gather,_BakAllReduce,_AllReduce

from scipy import optimize,linalg
import numpy as np
from scipy.optimize import leastsq,differential_evolution,NonlinearConstraint

class Optimizer:
    def __init__(self,alpha,beta,a,l,others=0) -> None:
        ###a2a,ag,exp,rs,gar
        self.alpha=alpha
        self.beta=beta
        self.a=a
        self.tmp_state=False
        d0,t0=self.optimize_degree(0)
        if (self.alpha[0]+self.beta[0]*self.a[0]) <1:
            d0=1
        _=self.get_overlap_part(d0)
        # if dist.get_rank()==0:
        #     for i in range(len(self.a)):
        #         print(self.beta[i]*self.a[i])
        #         print(self.alpha[i]+self.beta[i]*self.a[i])
        self.foward_state=False
        if self.exp_gap>(self.alpha[1]+self.beta[1]*self.a[1]/d0):
            self.foward_state=True
        self.d0=d0
        self.t0=t0
        global exp_ba
        global exp_bb
        self.alpha[2]=exp_ba
        self.beta[2]=exp_bb
        d1,t1=self.optimize_degree(0)
        if d0==1:
            d1=1
        self.d1=d1
        self.t1=t1
        self.gap=self.get_overlap_part(d1)
        self.l=l
        self.others=others
        

    def gar_time(self,g):
        if g<1e-5:
            return 0
        return self.alpha[4]+g*self.beta[4]
    def get_stored_grad(self,grads,t_olps):
        #grads : pre-layer grad, length is layer - 1 
        #t_olps: overlaped time per layer,length equal with grads
        #supposed that the first one is dense layer and follows dense,moe, dense, moe ...
        l=len(grads)
        stack_v=[]
        stack_idx=[]

        stack_olp_v=[]
       
        
        for i in range(l):
            # print(gar_time(grads[i])-t_olps[i])
            if self.gar_time(grads[i])>t_olps[i]:
                tmp=self.gar_time(grads[i])-t_olps[i]
                if t_olps[i]>self.alpha[4]:
                    stack_olp_v.append((t_olps[i]-self.alpha[4])/self.beta[4])
                    
                    stack_v.append((tmp)/self.beta[4])
                else:
                    stack_olp_v.append(0)
                    stack_v.append(grads[i])
                stack_idx.append(i)
            else:
                delta=t_olps[i]-self.gar_time(grads[i])
                while delta>1e-5 and len(stack_v)>0:
                    v_tmp=stack_v.pop()
                    if delta>self.gar_time(v_tmp):
                        delta-=self.gar_time(v_tmp)
                        stack_idx.pop()
                    else:
                        tmp=self.gar_time(v_tmp)-delta
                        if delta>self.alpha[4]:
                            stack_v.append((tmp)/self.beta[4])
                        else:
                            stack_v.append(v_tmp)
                        delta=0
                tmp=t_olps[i]-delta
                if tmp>self.alpha[4]:
                    stack_olp_v.append((tmp-self.alpha[4])/self.beta[4])
                else:
                    stack_olp_v.append(0)
        
        outs=[]
        idxs=[]
        tmpi=-1
        for item,idx in zip(stack_v,stack_idx):
           
            
            if idx%2==1:
                if tmpi!=-1:
                    outs[-1]+=item
                    tmpi=-1
                else:
                    outs.append(item)
                    idxs.append(idx)
            else:
                outs.append(item)
                idxs.append(idx)
                tmpi=idx
        return outs,idxs,stack_olp_v
                
    def get_overlap_part(self,d):
        alpha=self.alpha
        beta=self.beta
        a=self.a
        if (d*alpha[1]+a[1]*beta[1])>(d*alpha[0]+a[0]*beta[0]):
            
            def g2(d):
                return (d-1)*alpha[1]+(d-1)/d*a[1]*beta[1]+(d-1)*alpha[3]+(d-1)/d*a[3]*beta[3]-d*alpha[2]-a[2]*beta[2]
            if g2(d)>0:
                self.exp_gap=0
                return d*alpha[1]+a[1]*beta[1]+d*alpha[3]+a[3]*beta[3]-2*(d-1)*alpha[0]-2*(d-1)/d*a[0]*beta[0]
            else:
                self.exp_gap=-g2(d)
                return d*alpha[1]+a[1]*beta[1]+d*alpha[3]+a[3]*beta[3]-2*(d-1)*alpha[0]-2*(d-1)/d*a[0]*beta[0]-g2(d)
        def g2(d):
            return 2*(d-1)*alpha[0]+2*(d-1)/d*a[0]*beta[0]-d*alpha[2]-a[2]*beta[2]
        if g2(d)>0:
            self.exp_gap=0
            return alpha[1]+a[1]/d*beta[1]+alpha[3]+a[3]/d*beta[3]
        else:
            self.exp_gap=-g2(d)
            return d*alpha[2]+a[2]*beta[2]-2*(d-1)*alpha[0]-2*(d-1)/d*a[0]*beta[0]+alpha[1]+a[1]/d*beta[1]+alpha[3]+a[3]/d*beta[3]
    def optimize_gar(self,gars):
        if len(gars)==0:
            return [0]*(self.l-1)
        if len(gars)==1:
            return gars
        d_tmp,_=self.optimize_degree(self.gap+self.beta[4]*(gars[0]))
        if d_tmp==1:
            return gars
        bounds=[]
        sumup=0
        
        ys=[]
        for item in gars:
            sumup+=item
            bounds.append([0,sumup])
            ys.append(sumup)
       
        # 定义约束条件函数
        def constraint_func(xs):
            constraints = []
            for n in range(1, len(xs) + 1):
                # 检查 xs 的前 n 项和的每个元素是否都小于 ys 的对应元素
                constraint = ys[n-1] - np.sum(xs[:n])
                constraints.append(constraint)
            return constraints
        Nonlinear_Constraint = NonlinearConstraint(constraint_func,0,np.inf)
        def obj_func(xs):
            outv=0
            sumx=0
            for item in xs:
                sumx+=item
                if self.gap>self.alpha[4]:
                    _,v=self.optimize_degree(self.gap+self.beta[4]*(item))
                else:
                    _,v=self.optimize_degree(self.gar_time*(item))
                outv+=v
            outv+=(sumup-sumx)*self.beta[4]
            return outv

        
        # xs=[ 5.985e+04,1.104e+05,5.039e+04,1.919e+04,4.139e+05,9.498e+04,2.881e+04,3.581e+04,3.196e+04,1.104e+04,3.279e+05]
        # print(np.array(xs).sum())
        # print(obj_func(xs))
        # print(constraint_func(xs))
        outs=differential_evolution(obj_func,bounds,constraints=Nonlinear_Constraint)
        # print(outs)
        y_tmp=obj_func(gars)
        
        if outs.fun>(y_tmp-1e-5):
            return gars
        return outs.x
    def optimize_degree_case2(self,gar):
        #ag bigger than a2a
        alpha=self.alpha
        beta=self.beta
        a=self.a
        self.tmp_state=False
        def check(gs,f,d):
            
            d=d[0]
            if d<0:
                return [d],9999
            d1=np.ceil(d).astype(np.int64)
            d2=np.floor(d).astype(np.int64)
            upper=[1,1,2,4,4,8,8,8,8]
            downer=[1,1,2,2,4,4,4,4,8]
            d1=d1 if d1<=8 else 8
            d2=d2 if d2<=8 else 8
            d1=upper[d1]
            d2=downer[d2]
            f1=True if d1>0 else False
            f2=True if d2>0 else False
            for g in gs:
                if f1 and g([d1])<-1e-5:
                    f1=False
                if f2 and g([d2])<-1e-5:
                    f2=False
            v1=f([d1]) if f1 else 9999
            v2=f([d2]) if f2 else 9999
            if v1<v2:
                return [d1],v1
            return [d2],v2

        def gg(d):
            d=d[0]
            return (d*alpha[1]+a[1]*beta[1])-(d*alpha[0]+a[0]*beta[0])
        def f1(d):
            d=d[0]
            return 2*alpha[0]+2/d*a[0]*beta[0]+d*alpha[1]+a[1]*beta[1]+d*alpha[3]+a[3]*beta[3]

        def g0(d):
            d=d[0]
            return d*alpha[1]+a[1]*beta[1]+d*alpha[3]+a[3]*beta[3]-2*(d-1)*alpha[0]-2*(d-1)/d*a[0]*beta[0]-gar

        def g1(d):
            d=d[0]
            return d-1

        def g2(d):
            d=d[0]
            return (d-1)*alpha[1]+(d-1)/d*a[1]*beta[1]+(d-1)*alpha[3]+(d-1)/d*a[3]*beta[3]-d*alpha[2]-a[2]*beta[2]

        constraints = [dict(type='ineq', fun=gg),dict(type='ineq', fun=g0),dict(type='ineq', fun=g1),dict(type='ineq', fun=g2)]
        d1 = optimize.minimize(f1,(1),method='SLSQP',constraints=constraints).x
        d1,t1=check([gg,g1,g2,g0],f1,d1)
       
        
        def f2(d):
            d=d[0]
            return 2*alpha[0]+2/d*a[0]*beta[0]+alpha[1]+a[1]/d*beta[1]+alpha[3]+a[3]/d*beta[3]+d*alpha[2]+a[2]*beta[2]
        def g3(d):
            return -g2(d)
        constraints = [dict(type='ineq', fun=gg),dict(type='ineq', fun=g0),dict(type='ineq', fun=g1),dict(type='ineq', fun=g3)]
        d2 = optimize.minimize(f2,(2),method='SLSQP',constraints=constraints).x
        d2,t2=check([gg,g1,g3,g0],f2,d2)
        
        
        def ig0(d):
            return -g0(d)

        def f3(d):
            d=d[0]
            return 2*d*alpha[0]+2*a[0]*beta[0]+gar
        
        def g4(d):
            d=d[0]
            return 2*(d-1)*alpha[0]+2*(d-1)/d*a[0]*beta[0]+gar-(d)*(alpha[1]+a[1]/d*beta[1]+alpha[3]+a[3]/d*beta[3])
        constraints = [dict(type='ineq', fun=gg),dict(type='ineq', fun=ig0),dict(type='ineq', fun=g1),dict(type='ineq', fun=g4),dict(type='ineq', fun=g2)]
        d3 = optimize.minimize(f3,(2),method='SLSQP',constraints=constraints).x
        d3,t3=check([gg,ig0,g1,g4,g2],f3,d3)

        def g41(d):
            d=d[0]
            return 2*(d-1)*alpha[0]+2*(d-1)/d*a[0]*beta[0]+gar-(alpha[1]+a[1]/d*beta[1]+alpha[3]+a[3]/d*beta[3]+d*alpha[2]+a[2]*beta[2])
        constraints = [dict(type='ineq', fun=gg),dict(type='ineq', fun=ig0),dict(type='ineq', fun=g1),dict(type='ineq', fun=g41),dict(type='ineq', fun=g3)]
        d31 = optimize.minimize(f3,(2),method='SLSQP',constraints=constraints).x
        d31,t31=check([gg,ig0,g1,g41,g3],f3,d31)
        
        def g5(d):
            return -g4(d)
        def g51(d):
            return -g41(d)

        constraints = [dict(type='ineq', fun=gg),dict(type='ineq', fun=ig0),dict(type='ineq', fun=g1),dict(type='ineq', fun=g5),dict(type='ineq', fun=g2)]
        d4 = optimize.minimize(f2,(2),method='SLSQP',constraints=constraints).x
        d4,t4=check([gg,ig0,g1,g5,g2],f2,d4)

        constraints = [dict(type='ineq', fun=gg),dict(type='ineq', fun=ig0),dict(type='ineq', fun=g1),dict(type='ineq', fun=g51),dict(type='ineq', fun=g3)]
        d41 = optimize.minimize(f2,(8),method='SLSQP',constraints=constraints).x
        d41,t41=check([gg,ig0,g1,g51,g3],f2,d41)

        tt=[t1,t2,t3,t4,t31,t41]
       
        dd=[d1[0],d2[0],d3[0],d4[0],d31[0],d41[0]]
        # if dist.get_rank()==0:
        #     print(f2([1]))
        #     print(gg([1]))
        #     print(g1([1]))
        #     print(g3([1]))
        #     print(g0([1]))
        #     print(tt)
        #     print(dd)
        idxi=np.array(tt).argmin()
        if idxi in [1,3,5]:
            self.tmp_state=True
        return dd[idxi],np.array(tt).min()



    def optimize_degree(self,gar):
        alpha=self.alpha
        beta=self.beta
        a=self.a
        self.tmp_state=False
        def check(gs,f,d):
            if d<0:
                return d,9999
            d=d[0]
            d1=np.ceil(d).astype(np.int64)
            d2=np.floor(d).astype(np.int64)
            upper=[1,1,2,4,4,8,8,8,8]
            downer=[1,1,2,2,4,4,4,4,8]
            d1=d1 if d1<=8 else 8
            d2=d2 if d2<=8 else 8
           
            d1=upper[d1]
            d2=downer[d2]
            f1=True if d1>0 else False
            f2=True if d2>0 else False
            for g in gs:
                if f1 and g([d1])<-1e-5:
                    f1=False
                if f2 and g([d2])<-1e-5:
                    f2=False
            v1=f([d1]) if f1 else 9999
            v2=f([d2]) if f2 else 9999
            if v1<v2:
                return [d1],v1
            return [d2],v2
        
        def gg(d):
            d=d[0]
            return -(d*alpha[1]+a[1]*beta[1])+(d*alpha[0]+a[0]*beta[0])
        
        def f1(d):
            d=d[0]
            return 2*d*alpha[0]+2*a[0]*beta[0]+alpha[1]+a[1]/d*beta[1]+alpha[3]+a[3]/d*beta[3]

        def g0(d):
            d=d[0]
            return alpha[1]+a[1]/d*beta[1]+alpha[3]+a[3]/d*beta[3]-gar

        def g1(d):
            d=d[0]
            return d-1

        def g2(d):
            d=d[0]
            return 2*(d-1)*alpha[0]+2*(d-1)/d*a[0]*beta[0]-d*alpha[2]-a[2]*beta[2]

        constraints = [dict(type='ineq', fun=gg),dict(type='ineq', fun=g0),dict(type='ineq', fun=g1),dict(type='ineq', fun=g2)]
        d1 = optimize.minimize(f1,(1),method='SLSQP',constraints=constraints).x
        d1,t1=check([gg,g1,g2,g0],f1,d1)
       
        
        def f2(d):
            d=d[0]
            return 2*alpha[0]+2/d*a[0]*beta[0]+alpha[1]+a[1]/d*beta[1]+alpha[3]+a[3]/d*beta[3]+d*alpha[2]+a[2]*beta[2]
        def g3(d):
            return -g2(d)
        constraints = [dict(type='ineq', fun=gg),dict(type='ineq', fun=g0),dict(type='ineq', fun=g1),dict(type='ineq', fun=g3)]
        d2 = optimize.minimize(f2,(4),method='SLSQP',constraints=constraints).x
        d2,t2=check([gg,g1,g3,g0],f2,d2)
        
        
        def ig0(d):
            return -g0(d)

        def f3(d):
            d=d[0]
            return 2*d*alpha[0]+2*a[0]*beta[0]+gar
        
        def g4(d):
            return g2(d)-g0(d)
        constraints = [dict(type='ineq', fun=gg),dict(type='ineq', fun=ig0),dict(type='ineq', fun=g1),dict(type='ineq', fun=g4)]
        d3 = optimize.minimize(f3,(2),method='SLSQP',constraints=constraints).x
        d3,t3=check([gg,ig0,g1,g4],f3,d3)

        
        def g5(d):
            return -g2(d)+g0(d)
        constraints = [dict(type='ineq', fun=gg),dict(type='ineq', fun=ig0),dict(type='ineq', fun=g1),dict(type='ineq', fun=g5)]
        d4 = optimize.minimize(f2,(2),method='SLSQP',constraints=constraints).x
        d4,t4=check([gg,ig0,g1,g5],f2,d4)

        d5,t5=self.optimize_degree_case2(gar)


        tt=[t1,t2,t3,t4,t5]
        dd=[d1[0],d2[0],d3[0],d4[0],d5]
        idxi=np.array(tt).argmin()
       
        if idxi in [1,3]:
            self.tmp_state=True
        return dd[np.array(tt).argmin()],np.array(tt).min()
    def measure_others(self,b,s,m,h,heads):
        if self.others is not None:
            return self.others+self.alpha[1]+self.beta[1]*(b*s*m)+self.alpha[3]+self.beta[3]*(b*s*m)

        alpha_dict={}
        alpha_dict[512]=0.7674542565
        alpha_dict[1024]=6.4634952121E-1
        alpha_dict[256]=9.061728879E-1
        beta_dict={}
        beta_dict[512]=1.7435315251E-10
        beta_dict[1024]=2.5884462453E-10
        beta_dict[256]=1.6690605555E-10
        alpha=alpha_dict[s]
        beta=beta_dict[s]
        a=(b*s*m*m//4*3+b*heads//4*s*s*m//heads+b*heads//4*s*s*m//heads+b*s*m//4*m)
        return (2*alpha+2*a*beta)+self.alpha[1]+self.beta[1]*(b*s*m)+self.alpha[3]+self.beta[3]*(b*s*m)
    def time_grad(self,time):
        if time>self.alpha[4]:
            return (time-self.alpha[4])/self.beta[4]
        return 0
    def run(self,b,s,m,h,heads,args):
        

        if args.att_type==1:
            grad= (int)(args.M*args.M//args.mp_size*(1+2*args.heads/32.)+args.M*args.M//args.mp_size)
            grad=m*m//args.mp_size*3//2+m*m//args.mp_size
        else:
            grad=m*m//args.mp_size*3+m//args.mp_size*3+m*m//args.mp_size+m
        l=self.l
        others=self.measure_others(b,s,m,h,heads)
        
        
        gars,idxs,stack_olp_v=self.get_stored_grad([grad,0]*(l-1),[others,self.gap]*(l-1))
        self.gflag=False
        self.agflag=False
        if len(gars)==0 and (args.type not in [4,6]):
            if self.exp_gap<self.others+5:
                exp_gap=0
            else:
                exp_gap=0.7*self.exp_gap
           
            t_grad=self.gar_time(grad)
            g1=min(t_grad,exp_gap)
            g2=min(max(0,t_grad-exp_gap),others)
            g3=max(0,t_grad-g1-g2)
            
            if exp_gap>1.8*t_grad:
                self.gflag=True
            grad_moe=self.time_grad(g1+g3)
            grad_other=self.time_grad(g2)
            stack_olp_v=[grad_other,grad_moe]*(l-1)
            if g2<self.others:
                self.agflag=True
        self.gar=gars
        if dist.get_rank()==0:
            xs=self.optimize_gar(gars)
            
            xs=torch.tensor(xs,dtype=torch.float32).cuda()
        else:
            xs=torch.zeros(self.l-1,dtype=torch.float32).cuda()
   
        dist.all_reduce(xs)
        xs=xs.detach().to('cpu').numpy()
        def myround(x):
            return math.ceil(x)+math.ceil(x)%2
        if dist.get_rank()==0:
            print(grad)
            print(self.gar_time(grad))
            print(self.d0)
            print(self.d1)
            print(others)
            print(gars)
            print(xs)
            print(stack_olp_v)
            print(self.exp_gap)
            print(self.gap)
        xs_moe_reverse=xs+myround(stack_olp_v[1])
        final_xs_moe=[]
        for i in range(l-1):
            final_xs_moe.append(int(xs_moe_reverse[l-2-i]))

        tmp=myround(stack_olp_v[0])
        final_xs_att=[int(grad*(l-1)-np.array(final_xs_moe).sum()-tmp*(l-2))]+[tmp]*(l-2)

        degrees=[]
        for item in final_xs_moe:
            # if dist.get_rank()==0:
            #     print(self.gar_time(item))
            d,_=self.optimize_degree(self.gar_time(item))
            _=self.get_overlap_part(d)
            self.backward_state=False
            if self.exp_gap>(self.alpha[1]+self.beta[1]*self.a[1]/d):
                self.backward_state=True
            self.backward_state=self.tmp_state
            degrees.append(d)
        
        return final_xs_moe,final_xs_att,degrees
def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)
def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator
def split_tensor_along_last_dim(tensor, num_partitions,
                                contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
    except ImportError:
        flash_attn_unpadded_func = None
try:
    from einops import rearrange
except ImportError:
    rearrange = None
import warnings
from typing import List, Optional, Tuple, Union
import math
class MixtralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MixtralAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config,mp_size=8, layer_idx= None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.mp_size=mp_size

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads//mp_size
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads//mp_size
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads      
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size//mp_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = MixtralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size//self.mp_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output


class MistralConfig:
    def __init__(self,hidden_size,num_attention_heads,attention_dropout,num_key_value_heads,max_position_embeddings,rope_theta) -> None:
        self.hidden_size=hidden_size
        self.num_attention_heads=num_attention_heads
        self.attention_dropout=attention_dropout
        self.num_key_value_heads=num_key_value_heads
        self.max_position_embeddings=max_position_embeddings
        self.rope_theta=rope_theta
class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        assert flash_attn_unpadded_func is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        assert all((i.is_cuda for i in (q,k,v)))

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]

        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)

        if self.training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q

            is_causal = self.causal
            cu_seqlens_k = cu_seqlens_q
            dropout_p = self.dropout_p
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                        device=q.device)
            dropout_p = 0

        output = flash_attn_unpadded_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
            dropout_p,
            softmax_scale=self.softmax_scale, causal=is_causal
        )

        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output
class Attention(nn.Module):
    def __init__(self, args,mp_group) -> None:
        super().__init__()

        self.layer_norm=torch.nn.LayerNorm(args.M,eps=1e-5,elementwise_affine=True)
        self.hidden_size=args.M
        qkv_size_per_partition = (args.M// args.mp_size) * 3
        out_size_per_partition = args.M // args.mp_size

        if args.att_type==1:
            self.local_attn = FlashSelfAttention(causal=True)
        else:
            self.local_attn=None
     
        self.attn_qkvw = nn.Parameter(0.001*torch.randn(qkv_size_per_partition,args.M).cuda(),
                                    requires_grad=True)
    
        self.attn_qkvb = nn.Parameter(torch.zeros(qkv_size_per_partition).cuda(),
                                    requires_grad=True)
        
        self.attn_ow = nn.Parameter(0.001*torch.randn(args.M,
                                                out_size_per_partition ).cuda(),
                                    requires_grad=True)

        self.attn_ob = nn.Parameter(torch.zeros(args.M).cuda(),
                                    requires_grad=True)
        self.num_attention_heads_per_partition = args.heads // args.mp_size
        self.hidden_size_per_partition = args.M // args.mp_size
        self.hidden_size_per_attention_head = args.M // args.heads

        self.mp_group = mp_group
        self.norm_factor=math.sqrt(args.M// args.heads)
        self.attention_dropout = torch.nn.Dropout(0.2)
    def forward(self,input):
        seq=input.shape[1]
        
        norm_input=self.layer_norm(input)
        
        qkv_out=F.linear(norm_input,self.attn_qkvw,self.attn_qkvb)
     
        new_tensor_shape = qkv_out.size()[:-1] + \
                (3,self.num_attention_heads_per_partition,self.hidden_size_per_attention_head)
        mixed_x_layer = qkv_out.view(*new_tensor_shape)
        mixed_x_layer=mixed_x_layer.permute(1,0,3,2,4).contiguous().view(new_tensor_shape[1],new_tensor_shape[0],new_tensor_shape[3],-1)
        
        (query_layer,
        key_layer,
        value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
        if self.local_attn is not None:
            query_layer, key_layer, value_layer = [rearrange(x, 's b ... -> b s ...').contiguous()
                                for x in (query_layer, key_layer, value_layer)]
            context_layer = self.local_attn(query_layer, key_layer, value_layer)
            context_layer = rearrange(context_layer, 'b s h d -> b s (h d)').contiguous()
        else:
            output_size = (query_layer.size(1),
                        query_layer.size(2),
                        query_layer.size(0),
                        key_layer.size(0))
            
            # [sq, b, np, hn] -> [sq, b * np, hn]
            query_layer = query_layer.view(output_size[2],
                                        output_size[0] * output_size[1], -1)
            # [sk, b, np, hn] -> [sk, b * np, hn]
            key_layer = key_layer.view(output_size[3],
                                    output_size[0] * output_size[1], -1)
            
            # preallocting result tensor: [b * np, sq, sk]
            matmul_result = torch.empty(
                output_size[0]*output_size[1],
                output_size[2],
                output_size[3],
                dtype=query_layer.dtype,
                device=qkv_out.device)
            

            norm_factor=self.norm_factor
            # Raw attention scores. [b * np, sq, sk]
        
            matmul_result = torch.baddbmm(
                matmul_result,
                query_layer.transpose(0, 1),   # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0, alpha=(1.0/norm_factor))
        
            # change view to [b, np, sq, sk]
            attention_scores = matmul_result.view(*output_size)
            attention_scores = attention_scores * (norm_factor)
            if not hasattr(self, 'attention_mask'):
                self.attention_mask = torch.tril(torch.ones(
        (1, seq, seq), device=attention_scores.device)).view(
                1, 1, seq, seq)
                self.attention_mask = (self.attention_mask < 0.5)
            attention_scores.masked_fill_(self.attention_mask, -10000.0)
        
            attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
            attention_probs=attention_probs.type(input.dtype)
            # attention_probs = self.attention_dropout(attention_probs)
            # =========================
            # Context layer. [sq, b, hp]
            # =========================

            # value_layer -> context layer.
            # [sk, b, np, hn] --> [b, np, sq, hn]

            # context layer shape: [b, np, sq, hn]
            output_size = (value_layer.size(1),
                        value_layer.size(2),
                        query_layer.size(0),
                        value_layer.size(3))

            # change view [sk, b * np, hn]
            value_layer = value_layer.view(value_layer.size(0),
                                        output_size[0] * output_size[1], -1)
            
            # change view [b * np, sq, sk]
            attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                                output_size[2], -1)

            # matmul: [b * np, sq, hn]
        
            context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

            # change view [b, np, sq, hn]
            context_layer = context_layer.view(*output_size)

            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_layer_shape = context_layer.size()[:-2] + \
                (self.hidden_size_per_partition,)
            context_layer = context_layer.view(*new_context_layer_shape)
            
            
            context_layer=context_layer.permute(1,0,2)
        
        output=F.linear(context_layer,self.attn_ow,self.attn_ob)
        
        return output
class CudaEventTimer(object):
    def __init__(self, start_event: torch.cuda.Event, end_event: torch.cuda.Event):
        self.start_event = start_event
        self.end_event = end_event

    def get_elapsed_msec(self):
        torch.cuda.current_stream().wait_event(self.end_event)
        self.end_event.synchronize()
        return self.start_event.elapsed_time(self.end_event)

class SynchronizedWallClockTimer:
    """Group of timers. Borrowed from Nvidia Megatron code"""
    class Timer:
        """Timer."""
        def __init__(self, name):
            self.name_ = name
            self.started_ = False
            self.event_timers = []
            self.start_event = None
            self.elapsed_records = None

        def start(self):
            """Start the timer."""
            assert not self.started_, f"{self.name_} timer has already been started"
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
            self.started_ = True

        def stop(self, reset=False, record=False):
            """Stop the timer."""
            assert self.started_, "timer is not started"
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self.event_timers.append(CudaEventTimer(self.start_event, end_event))
            self.start_event = None
            self.started_ = False

        def _get_elapsed_msec(self):
            self.elapsed_records = [et.get_elapsed_msec() for et in self.event_timers]
            self.event_timers.clear()
            return sum(self.elapsed_records)

        def reset(self):
            """Reset timer."""
            self.started_ = False
            self.start_event = None
            self.elapsed_records = None
            self.event_timers.clear()

        def elapsed(self, reset=True):
            """Calculate the elapsed time."""
            started_ = self.started_
            # If the timing in progress, end it first.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            elapsed_ = self._get_elapsed_msec()
            # Reset the elapsed time
            if reset:
                self.reset()
            # If timing was in progress, set it back.
            if started_:
                self.start()
            return elapsed_

        def mean(self,trim_percent=0.1):
            self.elapsed(reset=False)
            m,s=trim_mean(self.elapsed_records, trim_percent)
            return m,s

    def __init__(self):
        self.timers = {}

    def get_timers(self):
        return self.timers

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    @staticmethod
    def memory_usage():
        alloc = "mem_allocated: {:.4f} GB".format(torch.cuda.memory_allocated() /
                                                  (1024 * 1024 * 1024))
        max_alloc = "max_mem_allocated: {:.4f} GB".format(
            torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024))
        cache = "cache_allocated: {:.4f} GB".format(torch.cuda.memory_cached() /
                                                    (1024 * 1024 * 1024))
        max_cache = "max_cache_allocated: {:.4f} GB".format(
            torch.cuda.max_memory_cached() / (1024 * 1024 * 1024))
        return " | {} | {} | {} | {}".format(alloc, max_alloc, cache, max_cache)



    def get_mean(self, names, normalizer=1.0, reset=True):
        """Get the mean of a group of timers."""
        assert normalizer > 0.0
        means = {}
        for name in names:
            if name in self.timers:
                elapsed_time = (self.timers[name].mean() * 1000.0 / normalizer)
                means[name] = elapsed_time
        return means
import numpy as np
def trim_mean(data, trim_percent):
    """Compute the trimmed mean of a list of numbers.

    Args:
        data (list): List of numbers.
        trim_percent (float): Percentage of data to trim.

    Returns:
        float: Trimmed mean.
    """
    assert trim_percent >= 0.0 and trim_percent <= 1.0
    n = len(data)
    # Account for edge case of empty list
    if len(data) == 0:
        return 0,0
    data.sort()
    k = int(round(n * (trim_percent)))
    m=np.mean(data[k:n - k])
    s=np.std(data[k:n - k])
    return m,s
test_timer=SynchronizedWallClockTimer()

class MOELayer(MOELayerBase):
    def __init__(self, gate,order: OrderBase = None,experts=None,dispatch_combine=None,  callbacks= [],ep_group=None,es_group=None,parm_group=None):
        super().__init__(gate,callbacks)
        self.order=order
        self.experts=experts
        self.dispatch_combine=dispatch_combine
        self.ep_group=ep_group
        self.es_group=es_group
        self.parm_group=parm_group
    def do_order(self, batch: dict) -> None:
        self.order.do_order(self,batch)
    def do_iorder(self, batch: dict) -> None:
        self.order.do_iorder(self,batch)
    def do_experts(self, batch: dict) -> None:
        self.experts.do_experts(self,batch)
    def do_dispatch(self, batch: dict) -> None:
        self.dispatch_combine.do_dispatch(self,batch)
    def do_combine(self, batch: dict) -> None:
        self.dispatch_combine.do_combine(self,batch)
    def forward(self, input):
        batch = {"data": input}
        self.before_moe_start_hook(batch)
        # if dist.get_rank()==0:
        #     print("gate")
        self.do_gate(batch)
        # if dist.get_rank()==0:
        #     print("order")
        self.do_order(batch)
        self.before_dispatch_hook(batch)
        # if dist.get_rank()==0:
        #     print("a2a")
        self.do_dispatch(batch)
        self.after_dispatch_hook(batch)
        self.do_experts(batch)
        self.before_combine_hook(batch)
        self.do_combine(batch)
        self.after_combine_hook(batch)
        self.do_iorder(batch)
        self.before_moe_end_hook(batch)
        return batch
def moe_init(args,pg_options=None):
    # Create a comm prependicular to the pipeline group as gate group
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    for i in range(0, args.es_size):
        ranks = range(i, world_size, args.es_size)
        group = torch.distributed.new_group(ranks,pg_options=pg_options)
        group1=torch.distributed.new_group(ranks,pg_options=pg_options)
        if rank in ranks:
            ep_group = group
            extra_group=group1
    for i in range(0, world_size, args.es_size):
        ranks = range(i, i + args.es_size)
        group = torch.distributed.new_group(ranks,pg_options=pg_options)
        if rank in ranks:
            es_group = group
    return ep_group,es_group,extra_group
class _ExtraOpt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,gar,s,ep_group):
        ctx.s=s
        ctx.gar=gar
        ctx.ep_group=ep_group
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.gar is not None:
            ctx.s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(ctx.s):
                dist.all_reduce(ctx.gar, op=dist.ReduceOp.SUM,group=ctx.ep_group)
        return grad_output,None,None,None
class MOETransformerLayer(nn.Module):
    def __init__(self,args,gar_moe=None,gar_att=None,mp_group=None,ep_group=None,bak_degree=4,degree=4,extra_group=None,extra_forward_state=False,extra_backward_state=False,obj=None):
        super().__init__()
        self.mp_group=mp_group
        self.ep_group=ep_group
        self.extra_group=extra_group
        expert_num=dist.get_world_size()//args.es_size
        model_dim=args.M
        hidden_dim=args.H
        seq=args.L
        batch_size=args.B
        if args.type==3:
            degree=1
            bak_degree=1
        elif args.type==4:
            degree=1
            bak_degree=1
       
        self.s_inter = torch.cuda.Stream(priority=-1)
        # if args.type==5:
        
        if args.type in [0,1,6]:
            s_intra=self.s_inter
        else:
            s_intra = torch.cuda.Stream(priority=-1)
        self.obj=obj
        if obj is not None and dist.get_rank()==0:
            print(obj.gflag)
            print("**")
        if (obj is not None and obj.gflag and args.type!=5)or args.type in [1,6]:
            self.s_ar=s_intra
        else:
            self.s_ar=torch.cuda.Stream(priority=0)
        self.s_ar=self.s_inter
        events_list=[[torch.cuda.Event(enable_timing=True) for _ in range(max(degree,bak_degree))]for _ in range(4)]
        gar_tensor=gar_moe
        self.gar_tensor_att=gar_att
        
        # gar_tensor=torch.empty([1641967+2040991]).cuda()
        # self.gar_tensor_att=torch.empty([4705649+1]).cuda()
        # gar_tensor=None
        # self.gar_tensor_att=torch.empty([8388608]).cuda()
        # self.gar_tensor_att=None
        self.layer_norm=torch.nn.LayerNorm(args.M,eps=1e-5,elementwise_affine=True)
        if args.att_type==0:
            self.att=Attention(args,mp_group)
        else:
            config=MistralConfig(args.M,32,0.0,args.heads,32768,1000000.0)
            self.att=MixtralAttention(config,args.es_size)
        #fsmoe
        self.moe = MOELayer(
            gate=LinearGate(model_dim, expert_num),
            order=TutelOrder(drop_tokens=True,tutel_v=True,capacity_factor=args.C,degree=degree,bak_degree=bak_degree),
            dispatch_combine=OverlapDispatchCombine(s_intra,self.s_inter,self.s_ar,events_list,degree=degree,bak_degree=bak_degree,gar_tensor=gar_tensor,extra_forward_state=extra_forward_state,extra_backward_state=extra_backward_state),
            experts=OverlapFFNExpert(model_dim,hidden_dim//args.es_size,events_list,degree=degree,ep_size=dist.get_world_size(ep_group),ffn_type=args.ffn_type,num_local_experts=1),
            ep_group=ep_group,
            es_group=es_group,
            callbacks=[
                CALLBACK_REGISTRY("FlatCallback")(),
            ],
        )
        #ds
        # self.moe = MOELayer(
        #     gate=LinearGate(model_dim, expert_num),
        #     order=TutelOrder(drop_tokens=True,tutel_v=True,capacity_factor=1.2,topk=2),
        #     dispatch_combine=DSDispatchCombine(),
        #     experts=FFNExpert(model_dim,hidden_dim//args.es_size,ep_size=dist.get_world_size(ep_group)//2,ffn_type=args.ffn_type,num_local_experts=2),
        #     ep_group=ep_group,
        #     es_group=mp_group,
        #     callbacks=[
        #         CALLBACK_REGISTRY("FlatCallback")(),
        #     ],
        # )
        #parm
        # if gar_moe is not None and gar_att is not None: 
        #     self.gar_tensor_att = torch.cat([gar_moe,gar_att]) 
        # elif gar_moe is not None:
        #     self.gar_tensor_att=gar_moe
        # else:
        #     self.gar_tensor_att=gar_att
        # self.moe = MOELayer(
        #     gate=LinearGate(model_dim, expert_num),
        #     order=TutelOrder(drop_tokens=True,tutel_v=True,capacity_factor=args.C),
        #     dispatch_combine=ParmDispatchCombine(),
        #     experts=FFNExpert(model_dim,hidden_dim//args.es_size,ep_size=dist.get_world_size(ep_group),ffn_type=args.ffn_type,num_local_experts=1),
        #     ep_group=ep_group,
        #     es_group=mp_group,
        #     callbacks=[
        #         CALLBACK_REGISTRY("FlatCallback")(),
        #     ],
        # )
    #parm
    def forward_p(self,inp):
        
       
        y=_BakAllReduce.apply(self.mp_group,inp)
        
        y=self.att(y)
        y=_AllReduce.apply(self.mp_group,y)
        y=y+inp
        y=self.layer_norm(y)
        res_2=y
        shape_tmp = y.shape
        y=y.view(-1,shape_tmp[-1])
        y=y.split(y.shape[0]//dist.get_world_size(self.mp_group),dim=0)[dist.get_rank(self.mp_group)]
        batch=self.moe(y)
        y=batch['data']
        y=gather(self.mp_group,y,0)
        y=y.view(shape_tmp)
        y=y+res_2
       
        dist.all_reduce(self.gar_tensor_att, op=dist.ReduceOp.SUM,group=self.ep_group)

        return y
    def forward_ds(self,inp):
        
        y=_BakAllReduce.apply(self.mp_group,inp)
        
        y=self.att(y)
        y=_AllReduce.apply(self.mp_group,y)
       

        y=y+inp
        y=self.layer_norm(y)
        res_2=y
        batch=self.moe(y)
        y=batch['data']
        y=y+res_2
       
       
        dist.all_reduce(self.gar_tensor_att, op=dist.ReduceOp.SUM,group=self.ep_group)
        return y        
    def forward(self,inp):
        
        y=gather(self.mp_group,inp,1)
        
        y=self.att(y)
        if args.type!=5 and args.type!=3 and args.type!=0:
            y=reduce_scatter(self.mp_group,y,1,gar=self.gar_tensor_att,s=self.s_ar,ep_group=self.ep_group,flag=self.obj.agflag)
        else:
            y=reduce_scatter(self.mp_group,y,1)
       
        y=y+inp
        y=self.layer_norm(y)
        res_2=y
        batch=self.moe(y)
        y=batch['data']
        y=y+res_2
        # if dist.get_rank()==0:
        #     print("ar")
        
        if args.type==5:
            y=_ExtraOpt.apply(y,self.gar_tensor_att,self.s_ar,self.extra_group)
        if args.type in [3,0]:
            dist.all_reduce(self.gar_tensor_att, op=dist.ReduceOp.SUM,group=self.ep_group)
        return y
class TransformerModel(nn.Module):
    def __init__(self,l,args,es_group=None,ep_group=None,extra_group=None):
        super().__init__()
        others=0
        file_name="timelog.txt"
        if args.fp16:
            file_name="timelog.txt" if args.att_type==1 else "timelog.txt"
        if dist.get_rank()==0:
            print(file_name)
        with open(file_name)as f:
            lines=f.readlines()
        for line in lines:
            
            tmp=line.split(',')
            time,b,s,m,heads=float(tmp[0]),int(tmp[1]),int(tmp[2]),int(tmp[3]),int(tmp[4])
          
            if s==args.L and m==args.M and heads==args.heads and b==args.B :
                others=time
                break
        b=args.B
        s=args.L
        m=args.M
        h=args.H
        heads=args.heads

        global exp_a
        global exp_b
        #16: 1.3057496907E-06,16:2.9156973646E-06
        #32: 1.8543323833E-06,32:3.4950701465E-06
        #48: 2.3201445211E-06,48:4.9511764052E-06
        betafp32=[2.3201445211E-06, 2.3477633843E-07, exp_b, 2.2171110136E-07, 4.9511764052E-06]
        betafp16=betafp32
        beta=betafp16 if args.fp16 else betafp32

        factor=1 if args.C>4 else args.C
        if args.type in [0,1,6]:
            beta[0]+=2*(beta[1]+beta[3])
            beta[1]=0
            beta[3]=0
            obj=Optimizer(others=others,l=l+1,alpha=[  2.8745790958E-01+(3.3789345076E-01+3.9538829900E-01 )/2,0,exp_a,0, 5.1136532051E-01 ],beta=beta,a=[factor*b*s*m/args.es_size,0,factor*b*s*m*h/args.es_size*2,0])
        else:
            obj=Optimizer(others=others,l=l+1,alpha=[  2.8745790958E-01 ,3.3789345076E-01,exp_a,3.9538829900E-01, 5.1136532051E-01 ],beta=beta,a=[factor*b*s*m/args.es_size,factor*b*s*m,factor*b*s*m*h/args.es_size*2,factor*b*s*m])
        
        d_forward=obj.d0
        
        final_xs_moe,final_xs_att,degrees=obj.run(b,s,m,h,heads,args=args)
        
        if args.type in [2,5]:
            extra_forward_state=obj.foward_state
            extra_backward_state=obj.backward_state
        else:
            extra_forward_state=False
            extra_backward_state=False
        if dist.get_rank()==0:
            print(others)
            print(final_xs_moe)
            print(final_xs_att)
            print(degrees)
        if args.att_type==1:
            grad=args.M*args.M//args.mp_size*3//2+args.M*args.M//args.mp_size
            grad= (int)(args.M*args.M//args.mp_size*(1+2*args.heads/32.)+args.M*args.M//args.mp_size)
        else:
            grad=args.M*args.M//args.mp_size*3+args.M//args.mp_size*3+args.M*args.M//args.mp_size+args.M
        bak_degrees=degrees
        bak_d=obj.d1
        
        global gar_xs_moe
        gar_xs_moe=final_xs_moe[0]
        
        if args.type==6:
            betafp32=[2.3201445211E-06,2.3477633843E-07,exp_b , 2.2171110136E-07,4.9511764052E-06]
            betafp16=betafp32
            beta=betafp16 if args.fp16 else betafp32
            obj=Optimizer(others=others,l=l+1,alpha=[ 2.8745790958E-01 ,3.3789345076E-01,exp_a, 3.9538829900E-01, 5.1136532051E-01 ],beta=beta,a=[factor*b*s*m/args.es_size,factor*b*s*m,factor*b*s*m*h/args.es_size*2,factor*b*s*m])
            final_xs_moe,final_xs_att,_=obj.run(b,s,m,h,heads,args)
            if dist.get_rank()==0:
                print(final_xs_moe)
                print(final_xs_att)
            gar_xs_moe=final_xs_moe[0]
      
        gar_moes=final_xs_moe
        gar_atts=final_xs_att
        if (args.type in [4] and d_forward==1 and bak_degrees[0]==1):
            if dist.get_rank() == 0:
                out_log(args,0)
            exit(0)
        if args.type not in [2,4,6]:
            gar_moes=[None]*l
            gar_atts=[grad]*(l)#+[None]
            
            bak_degrees=[bak_d]*(l)#+[4]
        if args.type in [6,5]:
            bak_degrees=[bak_d]*(l)
        
       
        
        self.layers=torch.nn.ModuleList()
        for gar_moe,gar_att,bak_degree in zip(gar_moes,gar_atts,bak_degrees):
            if gar_moe is not None:
                tensor_gar_moe=torch.empty([gar_moe]).cuda()
                if args.fp16:
                    tensor_gar_moe=tensor_gar_moe.half()
            else:
                tensor_gar_moe=None
            if gar_att is not None:
                tensor_gar_att=torch.empty([gar_att]).cuda()
                if args.fp16:
                    tensor_gar_att=tensor_gar_att.half()
            else:
                tensor_gar_att=None
            self.layers.append(MOETransformerLayer(args,tensor_gar_moe,tensor_gar_att,es_group,ep_group,bak_degree=bak_degree,degree=d_forward,extra_group=extra_group,extra_forward_state=extra_forward_state,extra_backward_state=extra_backward_state,obj=obj))
       
    def forward(self,inp):
        for layer in self.layers:
            inp=layer(inp)
        return inp
def out_log(args,outtimes):
    if dist.get_rank() == 0:
        with open("outlog.txt", "a+") as f:
            f.write(
                str(outtimes)
                + ","
                + str(args.B)
                + ","
                + str(args.L)
                + ","
                + str(args.M)
                + ","
                + str(args.H)
                + ","
                + str(args.heads)
                + ","
                +str(args.C)
                + ","
                +str(args.type)
                + ","
                +str(args.ffn_type)
                + ","
                +str(args.layer)
                + "\n"
            )

def set_seed(seed=1):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
                        
from torch.distributed.distributed_c10d import ProcessGroupNCCL
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--B", type=int,default=4)
    parser.add_argument("--L", type=int,default=512)
    parser.add_argument("--M", type=int,default=8192)
    parser.add_argument("--H", type=int,default=8192*4)
    parser.add_argument("--ffn_type", type=int,default=0)
    parser.add_argument("--att_type", type=int,default=1)
    parser.add_argument("--C", type=float,default=1.2)
    parser.add_argument("--heads", type=int,default=32)
    parser.add_argument("--es_size", type=int, default=8)
    parser.add_argument("--mp_size", type=int, default=8)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--type", type=int, default=0)
    parser.add_argument("--layer", type=int, default=1)
    
    #type 0:pipmoe + grad + others
    #1 inter inter grad-others  /pipemoe baseline
    #2 intra inter slice-grad   /ours 
    #3:no overlap 
    #4:intra inter degree=1 slice-grad /compare with case 2 to valid the degree optimization 
    #5:intra inter priority    /lina compare with case 2 to valid the slice-grad
    #6:inter inter slice-grad / compare with case 2 to valid the schedule
    args = parser.parse_args()
    global gar_xs_moe
    gar_xs_moe=10
    # if ((args.M//args.heads) <=64) and(args.heads>=32)and args.fp16:
    #     args.att_type=1
    global exp_a
    global exp_b
    global exp_ba
    global exp_bb
    list_b= [ 1.5314242213E-10/2, 3.1718550205E-10/2]
    list_a=[1.0865108931E+00/2*1.3 ,1.4718947807E+00 /2*1.3 ]
    
    
    list_bb=[1.5314242213E-10, 3.1718550205E-10 ]
    list_ba= [1.0865108931E+00*1.3, 1.4718947807E+00*1.3 ]


    exp_a=list_a[args.ffn_type]
    exp_b=list_b[args.ffn_type]
    exp_ba=list_ba[args.ffn_type]
    exp_bb=list_bb[args.ffn_type]
    torch.cuda.set_device(args.local_rank)
    pg_options =ProcessGroupNCCL.Options()
    pg_options.is_high_priority_stream=True
    if args.type==3:
        pg_options=None
    dist.init_process_group(backend="nccl",pg_options=pg_options)
    ep_group,es_group,extra_group=moe_init(args,pg_options)
    
    expert_num=dist.get_world_size()//args.es_size
    model_dim=args.M
    seq=args.L
    batch_size=args.B
  
    set_seed(1) 
    def decorate_trace_handler(rank):
        def trace_handler(prof):
            if rank in [0]:
                prof.export_chrome_trace("test"+str(rank)+".json")
        return trace_handler

    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
        schedule=torch.profiler.schedule(
            wait=10,
            warmup=10,
            active=5),
        on_trace_ready=decorate_trace_handler(torch.distributed.get_rank())
    ) 
    
    model=TransformerModel(args.layer,args,es_group,ep_group,extra_group=extra_group).cuda()

    if (args.type in [6] and gar_xs_moe==0):
        if dist.get_rank() == 0:
            out_log(args,0)
        exit(0)
    if args.type in [3]:
        inp =  torch.randn((batch_size, seq, model_dim)).cuda()
    else:
        inp = torch.randn((batch_size, seq//args.es_size, model_dim)).cuda()
    inp.requires_grad=True
    if args.fp16:
        model=model.half()
        inp=inp.half()
    
    torch.distributed.barrier()
    torch.cuda.synchronize()
    with prof:
        for i in range(50):
            # if args.C==4.8:
            #     inp = torch.randn((batch_size, seq//args.es_size, model_dim)).cuda()
            torch.distributed.barrier()

            if dist.get_rank() == 0 and i>9:
                test_timer('forward').start()
            
            out=model(inp)
            l=out.mean()
            # if dist.get_rank()==0:
            #     print("bak")
            l.backward()
            
            torch.cuda.synchronize()
            # if dist.get_rank()==0:
            #     print("**")
            if dist.get_rank() == 0 and i>9:
                test_timer('forward').stop()
            
            if dist.get_rank()==0:
                # if i%10==0:
                #     print(i)
                # print(i)
                # prof.step()
                pass
            
    if dist.get_rank() == 0:
        tmout=test_timer('forward').mean(0.2)
        out_log(args,tmout[0])
        print(tmout)
    # print("Hello World")
# python -m torch.distributed.launch --nproc_per_node=8   --nnodes=1  --master_addr=localhost  --master_port=1234 --node_rank=0 a2a_test.py 
