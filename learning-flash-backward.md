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

- [x] (13) recompute S = QK^T / sqrt(d) -- not stored in fwd; rebuilt from saved Q,K
- [x] (14) recover P_ij = exp(S_ij - L_i) -- L=logsumexp >= rowmax, so this is exactly softmax, 1 pass
- [x] D = rowsum(O ∘ dO) == rowsum(P ∘ dP)
      proof: rowsum(A∘B)=diag(AB^T); P dP^T = (PV) dO^T = O dO^T => diagonals equal
      (each diagonal entry (i,i) is the FULL within-row sum; off-diagonals are cross-row terms we drop)
      why O∘dO: P,dP are O(N^2) and would need materializing; O,dO are O(Nd) and already saved -> cheap
- [x] (15) dV = P^T dO / (16) dP = dO V^T -- standard matmul backward from O = PV
- [x] (17) dS_ij = P_ij (dP_ij - D_i) -- == (diag(P_i)-P_iP_i^T)dP_i expanded elementwise (softmax VJP)
      D_i is the per-row "g.p" centering scalar (same for all keys j in row i)
- [x] (18) dQ = dS K / sqrt(d) / (19) dK = dS^T Q / sqrt(d) -- standard matmul backward from S = QK^T/sqrt(d)
- [x] impl: assignment allows plain PyTorch + torch.compile (no online tricks needed since L is known).
      Module-level @torch.compile helper (stable identity -> cached). Triton bwd reuses it; must reshape
      d_o to the batch-FLATTENED layout of saved tensors and reshape grads back to caller dims.
      autograd: backward returns 4 values (dq, dk, dv, None) -- None for is_causal.

## 3. Broader context

- [x] "no online tricks" != "memory efficient": the compiled bwd still materializes O(N^2) S/P/dP/dS
- [x] Why fwd MUST be O(N) but bwd O(N^2) is tolerated: fwd activations persist across ALL layers until
      backward (O(N^2 \* L) -> fatal); bwd N^2 temps are transient, one layer at a time, freed immediately
- [x] Blast radius: long-context training memory; production flash also tiles the backward to stay O(N)
