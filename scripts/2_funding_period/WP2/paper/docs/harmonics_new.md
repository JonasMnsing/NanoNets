## Time-Dependent Two-Nanoparticle Network

### 1. Overview
We study two metallic nanoparticles (NP1, NP2) in series between a time-dependent source electrode (S) and a drain electrode (D), all above a gated substrate (G). No tunneling occurs directly to the substrate.

- **Source**: $U_S(t)=U_0\cos(\omega_0t)$  
- **NP1**: potential $\phi_1(t)$, excess charge $Q_1(t)$  
- **NP2**: potential $\phi_2(t)$, excess charge $Q_2(t)$  
- **Drain**: $U_D(t)=0$  
- **Substrate**: $U_G$ (static)

Each node connects only to nearest neighbors via a capacitor and a tunnel junction (modeled as a risistor $R_{ij}$ with rate $\Gamma_{ij}$).

---

### 2. Electrostatics and Capacitance Matrix
Define self- and mutual-capacitances:

- Self-capacitance of NP $i$ to gate: $C_i = K_sr_i$
- Mutual capacitance between node $i$ and node $j$: $C_{ij}\approx K_m\frac{r_i r_j}{r_i + r_j + d}$

with $K_s$ and $K_m$ as known constants, and $r_i$ as the radius of NP $i$ and $d$ as the shell-to-shell spacing between two nodes.  

Charge–potential relation:  
$$
\begin{pmatrix}Q_1\\Q_2\end{pmatrix}
= \mathbf C\,
\begin{pmatrix}\phi_1\\\phi_2\end{pmatrix},
\qquad
\mathbf C=\begin{pmatrix}C_{11}&-C_{12}\\-C_{12}&C_{22}\end{pmatrix},
$$  
with diagonal elements as the sums over nearest couplings:  
$$
C_{11}=C_{S1}+C_{12}+C_{1},
\quad
C_{22}=C_{2D}+C_{12}+C_{2}.
$$  
The inverse matrix is  
$$
\mathbf C^{-1}=\frac1\Delta
\begin{pmatrix}C_{22}&C_{12}\\C_{12}&C_{11}\end{pmatrix},
\quad
\Delta=C_{11}C_{22}-C_{12}^2.
$$
The capacitance matrix relates discrete charges to node potentials and sets the scale of charging energies.

---

### 3. Free-Energy Changes for Tunneling
Under orthodox Coulomb-blockade theory, an electron tunneling from node $i$ to $j$ changes the system free energy by
$$
\Delta F_{i\to j}
= e\bigl[\phi_i - \phi_j\bigr]
+\frac{e^2}{2}\bigl(C^{-1}_{ii}+C^{-1}_{jj}-2C^{-1}_{ij}\bigr).
$$
For tunneling between an electrode $E$ and NP $i$:
$$
\Delta F_{i\to E}
= e\bigl[\phi_i - U_E\bigr]
+\frac{e^2}{2}C^{-1}_{ii},
$$
The free energy encompasses work against potential differences and charging-energy cost. As an electrode serves as a charge reservoir we have reduced charging energy for those junctions. The magnitude of $\Delta F$ relative to the thermal energy $k_B T$ and bias energy $e U_0$ determines tunneling probabilites. Accordingly, tunneling is suppressed when $\Delta F \gg k_BT$ (Coulomg blockade) or when $\Delta F \lesssim eU_0$.

Explicit expressions for the six junctions follow by substituting the appropriate capacitance-matrix elements.
$$
\begin{aligned}
\Delta F_{1\to2}=e(\phi_1-\phi_2)+\frac{e^2}{2}\frac{C_{11}+C_{22}-2C_{12}}{\Delta}&,\quad
\Delta F_{2\to1}=e(\phi_2-\phi_1)+\frac{e^2}{2}\frac{C_{11}+C_{22}-2C_{12}}{\Delta}\\
\Delta F_{1\to S}=e(\phi_1-U_S)+\frac{e^2}{2}\frac{C_{22}}{\Delta}&,\quad
\Delta F_{S\to1}=e(U_S-\phi_1)+\frac{e^2}{2}\frac{C_{22}}{\Delta}\\
\Delta F_{2\to D}=e(\phi_2-U_D)+\frac{e^2}{2}\frac{C_{11}}{\Delta}&,\quad
\Delta F_{D\to2}=e(U_D-\phi_2)+\frac{e^2}{2}\frac{C_{11}}{\Delta}
\end{aligned}
$$

### 3.1 The $r_2 \gg r_1$ Limit

For a much larger second NP, the total capacitance for this NP is just given by its self-capacitance. Accordingly all remaining mutual capacitance values are negligable and the inverse approaches zero.  
$$
C_{22}\approx C_2\gg C_{12},\,C_{2D},
\quad C^{-1}_{22}\approx0,
\quad C^{-1}_{12}\approx0,
\quad C^{-1}_{11}\approx 1/C_{11}.
$$
In this limit NP2 acts as an equipotential reservoir. The free-energy changes reduce to a single-island form with the second island having a hardly changing potential upon charging:
$$
\begin{aligned}
\Delta F_{1\to2}=e(\phi_1-\phi_2)+\frac{e^2}{2C_{11}},\quad
\Delta F_{1\to S}=e(\phi_1-U_S)+\frac{e^2}{2C_{11}},\quad
\Delta F_{2\to D}=e(\phi_2-U_D).
\end{aligned}
$$

---

### 4. Exact Probabilistic Description: Master Equation

Define the charge state $\vec{n}$ and its probability $P(\vec{n},t)$. The master equation governs the time evolution:
$$
\frac{dP(\vec{n},t)}{dt} = \sum_{\vec{m}\neq\vec{n}}[\Gamma_{\vec{m}\rightarrow\vec{n}}P(\vec{m},t)-\Gamma_{\vec{n}\rightarrow\vec{m}}P(\vec{n},t)]
$$
The time-dependent tunneling rates are given by
$$
\Gamma_{ij}(t)
=-\frac{\Delta F_{ij}(t)}{e^2R_{ij}}
\;\frac{1}{1 - \exp\!\bigl[-\Delta F_{ij}(t)/(k_BT)\bigr]},
$$
Each tunneling event changes $(n_i,n_j)\rightarrow(n_i\pm1,n_j\mp1)$, causing instantaneous jumps in the node potentials via $\phi=C^{-1}Q$. Between events, $\phi_i$ adn $Q_i$ remain constant.

The master equation is exact for discrete-electron dynamics and crucial when tunneling events are rare or drive amplitudes are near the blockade threshold.

---

### 5. Macroscopic Current Conservation: Kirchhoff’s Law

By averaging over many stochastic trajectories, displacement currents and tunneling currents become smooth functions. KCL at node $i$ reads:
$$
\sum_j [C_{ij}\frac{d}{dt}(\phi_i - \phi_j) + I_{T,ij}] = 0
$$
where the ensemble-averaged tunneling current is
$$
I_{T,ij}(t) = e \sum_{\vec{n}}[\Gamma_{i \rightarrow j}(\vec{n},t)P(\vec{n},t)-\Gamma_{j \rightarrow i}(\vec{n},t)P(\vec{n},t)] 
$$
- Displacement currents $C_{ij}dV/dt$ represent field-induced charge shifts, not net electron flow. Still as only charge tunneling may change potentials, they are a direct consequence of tunneling events.
- Tunneling currents $I_{T,ij}$ capture net electron transfer across junctions.

KCL emerges as a mean-field description, valid when many tunneling events per cycle ensure smooth currents (large amplitudes). 

---

### 6. Closed-Form Solutions

In general we get for KCL at the first NP
$$
C_{S1}\frac{d}{dt}(U_S(t)-\phi_1)+C_{12}\frac{d}{dt}(\phi_2-\phi_1)+C_{1}\frac{d}{dt}(0-\phi_1)+I_{T,S1}+I_{T,12} = 0
$$
For small bias voltages of $eU_0<e^2/2C$ we reach Coulomb blockade regime and $I_{T,ij}\approx0$ (blocked tunneling). This allows us to integrate the remaining displacement currents in KCL to get:
$$
C_{S1}(U_S-\phi_1)+C_{12}(\phi_2-\phi_1)+C_{1}(0-\phi_1)=0,
$$
For both nodes we get the matrix form
$$
\begin{pmatrix}C_{S1}U_0\cos\omega_0t\\0\end{pmatrix}
=\begin{pmatrix}C_{11}&-C_{12}\\-C_{12}&C_{22}\end{pmatrix}
\begin{pmatrix}\phi_1\\\phi_2\end{pmatrix}
$$
with pure cosine solutions for each potential
$$
\phi_i(t)=\alpha_i\,U_0\cos\omega_0t,\quad
\alpha_1=\frac{C_{S1}\,C_{22}}{\Delta},\quad
\alpha_2=\frac{C_{S1}\,C_{12}}{\Delta}.
$$
For larger bias voltages $eU_0 \gtrsim e^2/2C$ or frequencies of $\omega_0 \sim 1/RC$ tunneling currents are obviously non-negligible and we have to solve the coupled master and KCL equation numerically.

---

### 7. AC-Driven Tunneling-Rate Harmonics

When we substitute the pure cosine solution into the free energy changes, we can write
$$
\Delta F_{i\to j}(t) = A_{ij}+B_{ij}\cos\omega_0t
$$
in a periodic form, with $A_{ij}$ containing the individual charging energies and $B_{ij}=e(\alpha_i-\alpha_j)U_0$ for the potential difference contribution.

Starting from the time‐dependent rate for junction $i\to j$:
$$
\Gamma_{ij}(t)
=-\frac{\Delta F_{ij}(t)}{e^2R_{ij}}
\;\frac{1}{1 - \exp\!\bigl[-\Delta F_{ij}(t)/(k_BT)\bigr]}
$$
1. **Geometric series**  
   $$
   \frac{1}{1 - e^{-x}}
   = \sum_{\ell=0}^{\infty} e^{-\ell x},
   \quad
   x=\frac{\Delta F_{ij}(t)}{k_BT}.
   $$
   Hence get
   $$
   \Gamma_{ij}(t)
   = -\frac{A_{ij}+B_{ij}\cos(\omega_0t)}{e^2R_{ij}}
     \sum_{\ell=0}^{\infty}
     \exp\!\Bigl[-\ell\frac{A_{ij}+B_{ij}\cos(\omega_0t)}{k_BT}\Bigr].
   $$

2. **Bessel expansion of the cosine‐exponential**  
   For each term in $\ell$,
   $$
   \exp\!\Bigl[-\ell\,\frac{B_{ij}}{k_BT}\cos(\omega_0t)\Bigr]
   = \sum_{m=-\infty}^{\infty}
     (-1)^m\,I_m\!\Bigl(\tfrac{\ell\,B_{ij}}{k_BT}\Bigr)
     \,e^{\,i\,m\,\omega_0t},
   $$
   where $I_m$ is the modified Bessel function and $I_m(-z)=(-1)^mI_m(z)$. The Bessel function is defined as
   $$
   I_m(x)=\frac{1}{\pi}\int_0^\pi\cos(m\theta)e^{xcos\theta}d\theta
   $$
   serving as the Fourier coefficient of the exponential function.

3. **Combine and reorganize**  
   $$
   \Gamma_{ij}(t)
   = -\frac{1}{e^2R_{ij}}
     \sum_{\ell=0}^{\infty}e^{-\ell A_{ij}/(k_BT)}
     \bigl[A_{ij}+B_{ij}\cos(\omega_0t)\bigr]
     \sum_{m=-\infty}^{\infty}
       (-1)^m\,I_m\!\Bigl(\tfrac{\ell B_{ij}}{k_BT}\Bigr)
       e^{\,i\,m\,\omega_0t}.
   $$
   Multiply out the $[A_{ij}+B_{ij}\cos]$ factor:
   - The $A_{ij}$ term contributes $A_{ij}(-1)^mI_m\,e^{i m\omega_0t}$.  
   - The $B_{ij}\cos\omega_0t$ term shifts indices via
     $\cos\omega_0t=\tfrac12(e^{i\omega_0t}+e^{-i\omega_0t})$.

4. **Fourier coefficient extraction**  
   Expand
   $$
   \Gamma_{ij}(t)=\sum_{n-\infty}^\infty \Gamma_{ij}^{(n)}e^{in\omega_0t}
   $$
   By definition,
   $$
   \Gamma_{ij}^{(n)}
   = \frac{\omega_0}{2\pi}\int_0^{2\pi/\omega_0}
     \Gamma_{ij}(t)\,e^{-i n\omega_0t}\,dt.
   $$
   The integral picks out:
   - $m=n$ from the $A_{ij}$ term (via $\int e^{i(m-n)t}dt$).  
   - $m=n\pm1$ from the $B_{ij}\cos$ term.
   
   Collecting all surviving contributions yields
   $$
   \boxed{
   \Gamma_{ij}^{(n)}
   = -\frac{1}{e^2R_{ij}}
     \sum_{\ell=0}^{\infty}e^{-\ell A_{ij}/k_BT}
     \Bigl[
       A_{ij}(-1)^n I_n\!\bigl(\tfrac{\ell B_{ij}}{k_BT}\bigr)
       + \tfrac{B_{ij}}{2}(-1)^{n-1}I_{n-1}\!\bigl(\tfrac{\ell B_{ij}}{k_BT}\bigr)
       + \tfrac{B_{ij}}{2}(-1)^{n+1}I_{n+1}\!\bigl(\tfrac{\ell B_{ij}}{k_BT}\bigr)
     \Bigr].
   }
   $$

### 8.1 Physical Implications
- **Drive-assisted ("photon-assisted") tunneling**: $\ell$ counts how many "quanta" are absorbed/emitted, weighted by $e^{-\ell A/k_BT}$. $I_n$ is the amplitude for converting $\ell$ quanta into $n$-th harmonic.  
- **Temperature regimes**: Two competing temperature effects with $\Gamma^{(n)} \propto \sum_l e^{-lA/k_BT}I_n(\frac{lB}{k_BT})$ where $A \propto e^2/2C$ is the charging energy scale and $B \propto e U_0$ is the drive amplitude.
  - $k_BT\ll A$: Exp-function strongly suppress all $\ell \ge 1$ terms, virtually no tunneling at all, blockade dominates. Even tough higher Besser orders are favored, there are simply no events to dress with harmonics $\rightarrow$ Spectrum collapses.  
  - $k_BT\gg A$: As we get $e^{-lA/k_BT} \approx 1$ many $\ell$ are thermally allowed. Still the argument of $I_n$ becomes small and as $I_n(x) \propto x^{n}/n!$ decays fast for higher harmonics, thermal smearing kills nonlinearity and we only get low-order harmonics.  
  - **Sweet spot**: $A/k_BT\sim1$, $eU_0/k_BT\sim1$ → rich harmonic spectrum. Accoringly the nonlinear tunneling-rate spectrum is fully captured by those two dimensionless ratios.
- **Symmetry**: Notice $AI_n + \frac{B}{2}(I_{n-1}-I_{n+1})$ and $(-1)^n$. Without DC charging offset $A$ and symmetrical drive $B\cos(\omega_0t)$ each coefficient of even $n$ vanishes, leaving only odd harmonics. Any asymmetry introduces even harmonics.  

### 8.2 Additional Implications in the $r_2\gg r_1$ Limit
- **NP2 as a perfect reservoir**  
  In this limit $C_{22}\gg C_{12},C_{2D}$, so $C_{22}^{-1}\approx0$ and $C_{12}^{-1}\approx0$.  Hence for the 2–D junction
  $$
    B_{2\to D}=e(\alpha_2-0)U_0\to0
    \;\Longrightarrow\;
    \Gamma_{2\to D}(t)\text{ has only the DC term }\Gamma^{(0)}_{2\to D}.
  $$
  Accordingly, this junction isn't adding any extra frequency mixing and no harmonics are additionally generated. The already rich current entering NP2 simply passes straight trough the 2-D junction into the drain.

- **Effective two-junction single-island device**  
  Only the S–1 and 1–2 junctions carry nonzero $B_{ij}$, so all non-zero $\Gamma^{(n\neq0)}$ arise from those two.  The device maps onto a single-electron transistor with NP1 as the island, junctions S–1 and 1–2 as source and drain.

- **Relative strength of harmonics**  
  The two active $B$-coefficients become
  $$
    B_{S\to1}
      =e\frac{C_{12}+C_{1s}}{C_{11}}\,U_0,
    \quad
    B_{1\to2}
      =e\frac{C_{S1}}{C_{11}}\,U_0,
  $$
  so a rich mixing (higher harmonics) depends on the ratios $\tfrac{C_{12}+C_{1s}}{C_{11}}$ vs. $\tfrac{C_{S1}}{C_{11}}$.

---
### 9. Outside of the Blockade limit
As we cannot negltect tunneling currents for larger bias voltages or larger frequencies, KCL at the first NP
$$
C_{S1}\frac{d}{dt}(U_S(t)-\phi_1)+C_{12}\frac{d}{dt}(\phi_2-\phi_1)+C_{1}\frac{d}{dt}(0-\phi_1)+I_{T,S1}+I_{T,12} = 0
$$
can only be handled by Floquet's theorem, where any linear system with time-periodic coefficients admits solutions that can be written as a Fourier series in harmonics of the drive frequency:

1. For our occupation probabilities we assume in steady state
$$
P_{\vec{n}}(t) = \sum_{m=-\infty}^{+\infty}P_{\vec{n}}^{(m)}e^{im\omega_0t}
$$
with complex $P^{(m)}$ capture the amplitude and phase of the $m$-th harmonic in the occupation of that charge state.

2. Likewise, rather than forcing $\phi_i(t)$ to be a pure cosine as in the small $\omega_0$ limit now we write
$$
\phi_i(t)=\sum_{m=-\infty}^{+\infty}\phi_i^{(m)}e^{im\omega_0t}
$$
with $\phi_i^{(0)}$ as the time-average (DC-shift), $\phi_i^{(\pm1)}$ the fundamental and $\phi_i^{(\pm2,\pm3,...)}$ the higher harmonics that emerge when the dynamics feed back nonlinearity.

3. Any time periodic object can itself be expanded
$$
\Gamma_{i \rightarrow j}(t) = \sum_{k=-\infty}^{+\infty}\Gamma_{i \rightarrow j}^{(k)}e^{ik\omega_0t}
\quad
I_{T,ij}(t) = \sum_{m=-\infty}^{+\infty}I_{T,ij}^{(m)}e^{im\omega_0t}
$$
For each $m$ we then get the upper KCL as
$$
im\omega_0[C_{S1}(V_{S1}^{(m)})+C_{12}(V_{12}^{(m)})+C_{1}(V_{1}^{(m)})]+I_{T,S1}^{(m)}+I_{T,12}^{(m)} = 0
$$
where $V_{S1}^{(m)}=U_0\delta_{|m|,1}-\phi_1^{(m)}$, $V_{12}^{(m)}=\phi_2^{(m)}-\phi_1^{(m)}$ and $V_{1}^{(m)}=-\phi_1^{(m)}$.

#### Physical Implications
1. In the small drive regime, $\phi_i(t)$ was locked to a single cosine at $\omega_0$. But now as capacitive and tunnel currents both matter, each $\phi_i(t)$ develops its own spectrum of harmonics $\phi_i^{m}$. Physically, the island no longer charges and discharges symmetrically but gets "pulled" or "pushed" in a frequency-dependent way that creates those higher-$m$ components.

2. When we sum up $im\omega_0[...]+I_{T,S1}^{(m)}+I_{T,12}^{(m)}$ we get frequency-mixing terms as the capacitance terms carries a phase shift relative to $\phi_1^{(m)}$ and the tunnel currents have their own amplitude and phase dictated by the nonlinear $\Gamma[\Delta F]$. Some harmonics can then cancel, others might reinforce, leading to new sidebands and potentially asymmetric spectra.

3. As $im\omega_0C$ and $1/R$ set two competing "impedances", at particular $\omega_0$ we might get resonant balances that boost certain $m$ over others.  For example, if the phase lag of the capacitive current at $m=2$ lines up perfectly with the nonlinearity of $\Gamma$ we might see a peak in second harmonic, even if it was weak in the quasi‐static limit.

4. Dynamic feedback of $\phi_i^{(m)}$ can break symmetry with nonzero DC component $\phi_i^{(0)}$ shifting the operating point allowing also even harmonics or phase lags having the waveform to no longer just be an odd function of time.

5. Each harmonic $\phi_i^{m}$ will have both an amplitude and a phase. As we crank up $U_0$ or sweep $T$, phases can rotate, causing interference between different harmonics, producing beats, envelope modulations or sub-harmonics (if there is enough memory in the system via slow tunneling)

