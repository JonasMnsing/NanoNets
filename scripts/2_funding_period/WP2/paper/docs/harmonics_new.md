## Time-Dependent Two-Nanoparticle Network

### 1. Overview
We consider a system of two metallic nanoparticles (NP1, NP2) arranged in series between a source (S) and drain (D) electrode. The electrodes and nanoparticles lie above a grounded substrate.  

- **Source**: time-dependent potential $U_S(t)$  
- **NP1**: potential $\phi_1(t)$, excess charge $Q_1(t)$  
- **NP2**: potential $\phi_2(t)$, excess charge $Q_2(t)$  
- **Drain**: potential $U_D(t)$ (often set to 0)  
- **Substrate**: ground (0 V)

Each nanoparticle has a self-capacitance to substrate and mutual capacitances to its nearest neighbours only (source–NP1, NP1–NP2, NP2–drain).  

---

### 2. Electrostatics and Capacitance Matrix
Each nanoparticle $i=1,2$ has self-capacitance to substrate $C_i = K_sr_i$ with $K_s$ a known constant and $r_i$ the NP radius.   Nearest-neighbour mutual capacitance $C_{ij}\approx K_m\frac{r_i r_j}{r_i + r_j + d}$ with $d$ the shell-to-shell spacing and $K_m$ a known constant.  

Define the charge–potential relation for NP1 and NP2:  
$$
\begin{pmatrix}Q_1\\Q_2\end{pmatrix}
= \mathbf C\,
\begin{pmatrix}\phi_1\\\phi_2\end{pmatrix},
\qquad
\mathbf C=\begin{pmatrix}C_{11}&-C_{12}\\-C_{12}&C_{22}\end{pmatrix},
$$  
where the diagonal elements are sums over nearest couplings:  
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

---

### 3. Free-Energy Change for Tunneling
Under orthodox Coulomb-blockade theory, when one electron ($-e$) tunnels from node $i$ to $j$, the change in free energy is:

- **NP–NP junction** $(i,j=1,2)$:
$$
\Delta F_{i\to j}
= e\bigl[\phi_i - \phi_j\bigr]
+\frac{e^2}{2}\bigl(C^{-1}_{ii}+C^{-1}_{jj}-2C^{-1}_{ij}\bigr).
$$

- **Electrode–NP junction** $(E=S,D)$:
$$
\Delta F_{i\to E}
= e\bigl[\phi_i - U_E\bigr]
+\frac{e^2}{2}C^{-1}_{ii},
$$

encompassing work against potential differences and charging-energy cost. As an electrode serves as a charge reservoir we have reduced charging energy for those junctions. 

Writing all six cases explicitly:
$$
\begin{aligned}
\Delta F_{1\to2}&=e(\phi_1-\phi_2)+\frac{e^2}{2}\frac{C_{11}+C_{22}-2C_{12}}{\Delta},\\
\Delta F_{2\to1}&=e(\phi_2-\phi_1)+\frac{e^2}{2}\frac{C_{11}+C_{22}-2C_{12}}{\Delta},\\
\Delta F_{1\to S}&=e(\phi_1-U_S)+\frac{e^2}{2}\frac{C_{22}}{\Delta},\\
\Delta F_{S\to1}&=e(U_S-\phi_1)+\frac{e^2}{2}\frac{C_{22}}{\Delta},\\
\Delta F_{2\to D}&=e(\phi_2-U_D)+\frac{e^2}{2}\frac{C_{11}}{\Delta},\\
\Delta F_{D\to2}&=e(U_D-\phi_2)+\frac{e^2}{2}\frac{C_{11}}{\Delta}.
\end{aligned}
$$

---

### 4. Limit $r_2\gg r_1$
If NP2 is much larger than NP1, then:
$$
C_{22}\approx C_2\gg C_{12},\,C_{2D},
\quad C^{-1}_{22}\approx0,
\quad C^{-1}_{12}\approx0,
\quad C^{-1}_{11}\approx 1/C_{11}.
$$
Thus NP2 acts as an equipotential reservoir. The free-energy changes reduce to a single-island form with the second island having a hardly changing potential upon charging:
$$
\begin{aligned}
\Delta F_{1\to2}&=e(\phi_1-\phi_2)+\frac{e^2}{2C_{11}},\\
\Delta F_{1\to S}&=e(\phi_1-U_S)+\frac{e^2}{2C_{11}},\\
\Delta F_{2\to D}&=e(\phi_2-U_D).
\end{aligned}
$$

---

### 5. Charge conservation and Kirchhoff's current law
First we approach our NP device from a continuous view where tunneling is assumed to be "smooth" or that we are working with an ensemble averages over many events. In this case we can define KCL for node $i$ to be
$$
\sum_j [C_{ij}\frac{d}{dt}(\phi_i - \phi_j) + I_{T,ij}] = 0
$$
with $j$ as the adjacent nodes. Firstly we have the displacement currents $C\frac{dV}{dt}$ for each capacitor where the capacitor accumalates or loses charges. This current does not represent a net movement of electrons across junctions but just the local changes in the electric field or induced surface charge. Hence those displacement currents does not contribute to the measurable current where only tunel events count instead it merely ensures that voltage variations still satisfy local KCL even in Coulomb blockade or high impedance regimes. A different way to put it is when thinking of our potentials in KMC to be piecewise constant. Voltages jump instantaneously when a tunneling event happens. Between events we have a fixed $\vec{Q}(t)$ and constant $\vec{\phi}(t)$ and thus $d\phi/dt=0$. Hence capacitor currents vanish between events. The only current is instantaneous and delta-function-like at the moment of tunneling. So in a single MC trajectory, there are no capacitors currents except delta-functions at tunneling events. In a macroscopic / averaged picture we can use continuous potentials and derive smooth displacement currents. Similarly we need the ensemble-averaged tunneling current across the resitor in between nodes $i$ and $j$ here, which is
$$
I_{T,ij}(t) = e \sum_{n_1,n_2}[\Gamma_{i \rightarrow j}(n_1,n_2;t)P_{n_1,n_2}-\Gamma_{j \rightarrow i}(n_1,n_2;t)P_{n_1,n_2}] 
$$

---

### 5. Quasi-Static AC Drive ($\omega_0\ll1/RC$)
In general we get for Kirchhoff's law at NP1
$$
C_{S1}\frac{d}{dt}(U_S(t)-\phi_1)+C_{12}\frac{d}{dt}(\phi_2-\phi_1)+C_{1}\frac{d}{dt}(0-\phi_1)+I_{T,S1}+I_{T,12} = 0
$$
Assuming now a pure AC drive around zero $U_S(t)=U_0\cos(\omega_0t),\;U_D=0$, with $\omega_0$ very small, the tunnel currents $I_T$ are driven to zero at each instant by Coulomb blockade equilibrium. The island charges re-arrange much faster so that there is literally no net resistive current flowing once they have caught up to the instantaneous bias. This allows us to integrate the remaining displacement currents to get:
$$
C_{S1}(U_S-\phi_1)+C_{12}(\phi_2-\phi_1)+C_{1}(0-\phi_1)=0,
$$
and similarly at NP2.  In matrix form:
$$
\begin{pmatrix}C_{S1}U_0\cos\omega_0t\\0\end{pmatrix}
=\begin{pmatrix}C_{11}&-C_{12}\\-C_{12}&C_{22}\end{pmatrix}
\begin{pmatrix}\phi_1\\\phi_2\end{pmatrix}.
$$
Solving gives pure cosine potentials:
$$
\phi_i(t)=\alpha_i\,U_0\cos\omega_0t,
$$
$$
\alpha_1=\frac{C_{S1}\,C_{22}}{\Delta},\quad
\alpha_2=\frac{C_{S1}\,C_{12}}{\Delta}.
$$

Substitute into $\Delta F_{i\to j}(t)$ to write each as
$A_{ij}+B_{ij}\cos\omega_0t$, with
$$B_{ij}=e(\alpha_i-\alpha_j)U_0.$$

---

### 6. Fourier Expansion of Tunneling Rates

We start from the time‐dependent rate for junction $i\to j$:
$$
\Gamma_{ij}(t)
=-\frac{\Delta F_{ij}(t)}{e^2R_{ij}}
\;\frac{1}{1 - \exp\!\bigl[-\Delta F_{ij}(t)/(k_BT)\bigr]},
$$
with
$\Delta F_{ij}(t)=A_{ij}+B_{ij}\cos(\omega_0t)$.

1. **Geometric series for the Bose factor**  
   $$
   \frac{1}{1 - e^{-x}}
   = \sum_{\ell=0}^{\infty} e^{-\ell x},
   \quad
   x=\frac{\Delta F_{ij}(t)}{k_BT}.
   $$
   Hence
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

---

### 7. Physical Implications
- **Drive-assisted ("photon-assisted") tunneling**: $\ell$ counts how many "quanta" are absorbed/emitted, weighted by $e^{-\ell A/k_BT}$. $I_n$ is the amplitude for converting $\ell$ quanta into $n$-th harmonic.  
- **Temperature regimes**: Two competing temperature effects with $\Gamma^{(n)} \propto \sum_l e^{-lA/k_BT}I_n(\frac{lB}{k_BT})$ where $A \propto e^2/2C$ is the charging energy scale and $B \propto e U_0$ is the drive amplitude.
  - $k_BT\ll A$: Exp-function strongly suppress all $\ell \ge 1$ terms, virtually no tunneling at all, blockade dominates. Even tough higher Besser orders are favored, there are simply no events to dress with harmonics $\rightarrow$ Spectrum collapses.  
  - $k_BT\gg A$: As we get $e^{-lA/k_BT} \approx 1$ many $\ell$ are thermally allowed. Still the argument of $I_n$ becomes small and as $I_n(x) \propto x^{n}/n!$ decays fast for higher harmonics, thermal smearing kills nonlinearity and we only get low-order harmonics.  
  - **Sweet spot**: $A/k_BT\sim1$, $eU_0/k_BT\sim1$ → rich harmonic spectrum. Accoringly the nonlinear tunneling-rate spectrum is fully captured by those two dimensionless ratios.
- **Symmetry**: Notice $AI_n + \frac{B}{2}(I_{n-1}-I_{n+1})$ and $(-1)^n$. Without DC charging offset $A$ and symmetrical drive $B\cos(\omega_0t)$ each coefficient of even $n$ vanishes, leaving only odd harmonics. Any asymmetry introduces even harmonics.  

#### 7.1 Additional Implications in the $r_2\gg r_1$ Limit
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
### 8. Dynamic AC Drive ($\omega_0\approx1/RC$)
At NP1, write
$$
C_{S1}\frac{d}{dt}(U_S(t)-\phi_1)+C_{12}\frac{d}{dt}(\phi_2-\phi_1)+C_{1}\frac{d}{dt}(0-\phi_1)+I_{T,S1}+I_{T,12} = 0
$$
with
$$
I_{T,ij}(t) = e \sum_{n_1,n_2}[\Gamma_{i \rightarrow j}(n_1,n_2;t)P_{n_1,n_2}-\Gamma_{j \rightarrow i}(n_1,n_2;t)P_{n_1,n_2}]
$$
By Floquet's theorem, any linear system with time-periodic coefficients admits solutions that can be written as a Fourier series in harmonics of the drive frequency:

1. For our occupation probabilities we assume in steady state
$$
P_{n_1,n_2}(t) = \sum_{m=-\infty}^{+\infty}P_{n_1,n_2}^{(m)}e^{im\omega_0t}
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
1. In the slow drive regime, $\phi_i(t)$ was locked to a single cosine at $\omega_0$. But now as capacitive and tunnel currents both matter, each $\phi_i(t)$ develops its own spectrum of harmonics $\phi_i^{m}$. Physically, the island no longer charges and discharges symmetrically but gets "pulled" or "pushed" in a frequency-dependent way that creates those higher-$m$ components.

2. When we sum up $im\omega_0[...]+I_{T,S1}^{(m)}+I_{T,12}^{(m)}$ we get frequency-mixing terms as the capacitance terms carries a phase shift relative to $\phi_1^{(m)}$ and the tunnel currents have their own amplitude and phase dictated by the nonlinear $\Gamma[\Delta F]$. Some harmonics can then cancel, others might reinforce, leading to new sidebands and potentially asymmetric spectra.

3. As $im\omega_0C$ and $1/R$ set two competing "impedances", at particular $\omega_0$ we might get resonant balances that boost certain $m$ over others.  For example, if the phase lag of the capacitive current at $m=2$ lines up perfectly with the nonlinearity of $\Gamma$ we might see a peak in second harmonic, even if it was weak in the quasi‐static limit.

4. Dynamic feedback of $\phi_i^{(m)}$ can break symmetry with nonzero DC component $\phi_i^{(0)}$ shifting the operating point allowing also even harmonics or phase lags having the waveform to no longer just be an odd function of time.

5. Each harmonic $\phi_i^{m}$ will have both an amplitude and a phase. As we crank up $U_0$ or sweep $T$, phases can rotate, causing interference between different harmonics, producing beats, envelope modulations or sub-harmonics (if there is enough memory in the system via slow tunneling)

