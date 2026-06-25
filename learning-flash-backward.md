# Understanding: FlashAttention-2 Backward (recomputation)

Scratch doc for the /learn-from-session walkthrough. Delete when done (or ask me to move it into docs/).

Forward recap (what we saved): `Q, K, V, O` (size O(Nd)) and `L = logsumexp(S)` (size O(N)).
Goal of backward: given `dO`, compute `dQ, dK, dV`. Never materialize the O(N^2) `P` or `S`.

## 1. The problem

- [x] What backward must produce (dQ, dK, dV) and from what inputs (dO + saved Q,K,V,O,L)
- [x] Why the naive backward is memory-expensive (P/S are (N_q,N_k) = O(N^2), ~256x O for N=16k; x B\*H)
- [x] Why O+L is enough: P_ij = exp(S_ij - L_i); L=logsumexp bakes in max+sum -> 1-pass, stable, normalized
      (note: torch.compile does NOT auto-fuse this into flash; QK^T is still materialized)

## 2. The solution (eqs 13-19)

- [ ] (13) recompute S = QK^T / sqrt(d)
- [ ] (14) recover P_ij = exp(S_ij - L_i) -- why this IS softmax
- [ ] D = rowsum(O ∘ dO) -- why it equals rowsum(P ∘ dP)
- [x] (15) dV = P^T dO / (16) dP = dO V^T -- standard matmul backward from O = PV
- [x] (17) dS_ij = P_ij (dP_ij - D_i) -- same as (diag(P_i)-P_iP_i^T)dP_i expanded elementwise
- [x] (18) dQ = dS K / sqrt(d) / (19) dK = dS^T Q / sqrt(d) -- standard matmul backward from S = QK^T/sqrt(d)
- [ ] tiling: which outer loop accumulates which gradient

## 3. Broader context

- [ ] Why the backward keeps an O(N) activation-memory profile
- [ ] Blast radius: where this matters (long context, training memory)
