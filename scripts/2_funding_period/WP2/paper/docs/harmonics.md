## Time dependent Nanoparticle Networks

### Electrostatics:

We consider a $2$ x $1$ NP network with two electrodes (Source and Drain). The source electrode at voltage $U_S(t)$ is connected to NP1 defined by its potential $\phi_1(t)$ and number of excess charges $Q_1(t)$. NP1 is connected to NP2 defined by $(\phi_2(t),Q_2(t))$. NP2 is connected to the drain electrode at voltage $U_D(t)$. The whole network is put on an insulating enverionment above a substrate which is grounded. Accordingly a NP has a self capacitance relative to ground $C_i = K_s \cdot r_i$ with $K_s$ as a known constant and $r_i$ as the radius of NP $i$. As junctions are insulating, we also have a mutual capacitance between nodes defined as $C_{ij} \approx K_m \frac{r_i r_j}{r_i + r_j + d}$ with $d$ as the shell-to-shell spacing between two nodes and $K_m$ as a known constant. Electrostatics are defined via
$$
\begin{pmatrix}
  Q_1 \\
  Q_2
\end{pmatrix} = 
\bf{C}
\begin{pmatrix}
  \phi_1 \\
  \phi_2
\end{pmatrix} =
\begin{pmatrix}
  C_{11} & -C_{12} \\
  -C_{21} & C_{22}
\end{pmatrix}
\begin{pmatrix}
  \phi_1 \\
  \phi_2
\end{pmatrix}
$$

with **C** as the capacitance matrix and $C_{ii}$ as the sum of capacitance for NP $i$. We get 
$$
C_{11} = C_{S1} + C_{12} + C_{1}
$$
and 
$$
C_{22} = C_{2D} + C_{12} + C_{2}
$$ 
as our two sums for both NPs. Accordingly we get
$$
\vec{\phi}(t) = \bf{C^{-1}}\vec{q}(t) 
$$
for our the potential landscape given the current charge distribution. The inverse of the capacitance matrix is defined as
$$
\bf{C^{-1}}= \frac{1}{\Delta}
\begin{pmatrix}
  C_{22} & C_{12} \\
  C_{21} & C_{11}
\end{pmatrix}
$$
with $\Delta = C_{11}C_{22}-C_{12}^2$.

### Change in free energy

When a charge is tunneling inside the system, we have a change in free energy. The change in free energy for a NP-NP junction is defined as
$$
\Delta F_{i \rightarrow j} = e(\phi_j - \phi_i) + \frac{e^2}{2}(\bf{C_{ii}^{-1}} + \bf{C_{jj}^{-1}} - 2\bf{C_{ij}^{-1}})
$$
and for an Elecrtrode-NP junction
$$
\Delta F_{i \rightarrow E} = e(U_E - \phi_i) + \frac{e^2}{2}\bf{C_{ii}^{-1}}
$$
Both include a work term defining the cost or gain we get of taking one electron against the difference in island potentials. The second part consists of a charging term which defines the change in $\frac{1}{2}Q^{-1}C^{-1}Q$ when an electron is shifted. As an electrode serves as a charge reservoir we lose the penalty in charging energy in the second term and only inlcude the charging energy of the contributing NP. Below all junctions are summarized:
$$
\Delta F_{1 \rightarrow 2} = e(\phi_2 - \phi_1) + \frac{e^2}{2}\frac{C_{11}+C_{22}-2C_{12}}{\Delta}
$$
$$
\Delta F_{2 \rightarrow 1} = e(\phi_1 - \phi_2) + \frac{e^2}{2}\frac{C_{11}+C_{22}-2C_{12}}{\Delta}
$$
$$
\Delta F_{1 \rightarrow S} = e(U_S - \phi_1) + \frac{e^2}{2}\frac{C_{22}}{\Delta}
$$
$$
\Delta F_{S \rightarrow 1} = e(\phi_1 - U_S) + \frac{e^2}{2}\frac{C_{22}}{\Delta}
$$
$$
\Delta F_{2 \rightarrow D} = e(U_D - \phi_2) + \frac{e^2}{2}\frac{C_{11}}{\Delta}
$$
$$
\Delta F_{D \rightarrow 2} = e(\phi_2 - U_D) + \frac{e^2}{2}\frac{C_{11}}{\Delta}
$$

### Limit of $r_2 \gg r_1$

Assuming the drain attached NP to be much larger than the source attached NP, i.e. $r_2 \gg r_1$ we get $C_{12}\approx K_m \cdot r_1$ and $C_{11}^{-1}\approx \frac{1}{C_{11}}$, $C_{22}^{-1}\approx 0$, $C_{12}^{-1}\approx 0$.
For our set of free energy differences we get:
$$
\Delta F_{1 \rightarrow 2} = e(\phi_2 - \phi_1) + \frac{e^2}{2C_{11}}
$$
$$
\Delta F_{2 \rightarrow 1} = e(\phi_1 - \phi_2) + \frac{e^2}{2C_{11}}
$$
$$
\Delta F_{1 \rightarrow S} = e(U_S - \phi_1) + \frac{e^2}{2C_{11}}
$$
$$
\Delta F_{S \rightarrow 1} = e(\phi_1 - U_S) + \frac{e^2}{2C_{11}}
$$
$$
\Delta F_{2 \rightarrow D} = e(U_D - \phi_2)
$$
$$
\Delta F_{D \rightarrow 2} = e(\phi_2 - U_D)
$$
In this limit the large NP behaves like an equipotential "reservoir" which reduces all charging energy related costs to those of island 1 alone $\frac{e^2}{2C_{11}}$. Effectively our 2-island system reduces to a 1-island system and the large NP becomes clamped by its own self-capacitance having a hardly changing potential upon charging.

### Time dependent Input Signal:

For $U_D(t)=0$ and $U_S(t) = U_{0}\cos(\omega_0t)$ we get the Kirchhoff node equations
$$
C_{S1}(U_S(t)-\phi_1)+C_{12}(\phi_2-\phi_1)+C_{1}(0-\phi_1)=0
$$
and
$$
C_{12}(\phi_1-\phi_2)+C_{2D}(0-\phi_2)+C_{2}(0-\phi_2)=0
$$
In matrix form as above we define the system using the capacitance matrix:
$$
\begin{pmatrix}
  C_{S1}U_{0}\cos(\omega_0t) \\
  0
\end{pmatrix} =
\begin{pmatrix}
  C_{S1} + C_{12} + C_1 & -C_{12} \\
  -C_{12} & C_{12} + C_{2D} + C_{2}
\end{pmatrix}
\begin{pmatrix}
  \phi_1 \\
  \phi_2
\end{pmatrix}
$$
For our potentials we get
$$
\phi_1(t) = \frac{(C_{12}+C_{2D}+C_{2})C_{S1}}{\Delta}U_0\cos(\omega_0 t)=\alpha U_0\cos(\omega_0 t)
$$
and
$$
\phi_2(t) = \frac{C_{12}C_{S1}}{\Delta}U_0\cos(\omega_0 t)=\beta U_0\cos(\omega_0 t)
$$
We substitute these into our six $\Delta F$ values and end up with:
$$
\Delta F_{1 \rightarrow 2} = \frac{e^2}{2}\frac{C_{11}+C_{22}-2C_{12}}{\Delta}+e(\alpha - \beta)U_0 \cos(\omega_0 t) = A_{12} + e(\alpha - \beta)U_0 \cos(\omega_0 t)
$$
$$
\Delta F_{2 \rightarrow 1} = \frac{e^2}{2}\frac{C_{11}+C_{22}-2C_{12}}{\Delta}-e(\alpha - \beta)U_0 \cos(\omega_0 t) = A_{12} - e(\alpha - \beta)U_0 \cos(\omega_0 t)
$$
$$
\Delta F_{1 \rightarrow S} = \frac{e^2}{2}\frac{C_{22}}{\Delta} + e(\alpha - 1)U_0 \cos(\omega_0 t) = A_{S1} + e(\alpha - 1)U_0 \cos(\omega_0 t)
$$
$$
\Delta F_{S \rightarrow 1} = \frac{e^2}{2}\frac{C_{22}}{\Delta} - e(\alpha - 1)U_0 \cos(\omega_0 t) = A_{S1} - e(\alpha - 1)U_0 \cos(\omega_0 t)
$$
$$
\Delta F_{2 \rightarrow D} = \frac{e^2}{2}\frac{C_{11}}{\Delta} + e\beta U_0 \cos(\omega_0 t) = A_{2D} + e\beta U_0 \cos(\omega_0 t)
$$
$$
\Delta F_{D \rightarrow 2} = \frac{e^2}{2}\frac{C_{11}}{\Delta} - e\beta U_0 \cos(\omega_0 t) = A_{2D} - e\beta U_0 \cos(\omega_0 t)
$$
So at the end we get for each difference in free energy a form $\Delta F_{i \rightarrow j}(t) = A_{ij} + B_{ij}\cos(\omega_0 t)$ which depends only on constants and the applied boundary condition. This form should be independet of the actual network structure as we should be able to solve the linear system of equation for any type of topology should be achievalbe eventually achieving equations dependent on the external drive. This means that the upcoming results are general and independent of the network structure.

### Expanding the tunnelling rates

Once as we have our free energy differences in the form $\Delta F_{i \rightarrow j}(t) = A_{ij} + B_{ij}\cos(\omega_0 t)$ we can expand the corresponding rate
$$
\Gamma_{i \rightarrow j}(t)=-\frac{\Delta F_{i \rightarrow j}(t)}{e^2 R_{i \rightarrow j}} \frac{1}{1-\exp(-\Delta F_{i \rightarrow j}(t)/(k_B T))}
$$
in a Fourier series $\Gamma_{i \rightarrow j}(t) = \sum_m \Gamma_{i \rightarrow j}^{(m)}\exp(im\omega_0t)$.
Use
$$
\frac{1}{1-e^{-x}}=\sum_{l=0}^{\infty}e^{-lx}
$$
to rewrite
$$
\Gamma_{i \rightarrow j}(t)=-\frac{A_{ij} + B_{ij}\cos(\omega_0 t)}{e^2 R_{i \rightarrow j}} \sum_{l=0}^{\infty} \exp\bigg(-l \frac{A_{ij} + B_{ij}\cos(\omega_0 t)}{k_B T}\bigg)
$$
Now we can expand the exponential of a cosine as
$$
\exp\bigg(-\frac{lB_{ij}}{k_B T}\cos(\omega_0 t)\bigg) = \sum_{m=-\infty}^{\infty}(-1)^m I_m\bigg(\frac{lB_{ij}}{k_B T}\bigg)\exp(im\omega_0t)
$$
with $I_m$ as the modified Bessel function. In summary we get
$$
\Gamma_{i \rightarrow j}(t)=-\frac{A_{ij} + B_{ij}\cos(\omega_0 t)}{e^2 R_{i \rightarrow j}} \sum_{l=0}^{\infty} e^{-\frac{lA_{ij}}{k_B T}}\sum_{m=-\infty}^{\infty}(-1)^m I_m\bigg(\frac{lB_{ij}}{k_B T}\bigg)e^{im\omega_0t}
$$
Now as we can write the Fourier coefficient in general as
$$
\Gamma^{(n)}=\frac{\omega_0}{2 \pi}\int_0^{2\pi/\omega_0}\Gamma(t)e^{-in\omega_0t}dt
$$
plugging in our previous equation and solve the constant part of the integral as
$$
\frac{\omega_0}{2 \pi}\int_0^{2\pi/\omega_0}Ae^{i(m-n)\omega_0t}dt = A\delta_{m,n}
$$
and the time dependent part of the integral with the help of $\cos(\omega_0 t) = \frac{1}{2}(e^{i\omega_0t}+e^{-i\omega_0t})$ as
$$
\frac{\omega_0}{2 \pi}\int_0^{2\pi/\omega_0}B\cos(\omega_0t)e^{i(m-n)\omega_0t}dt = \frac{B}{2}(\delta_{m-n,+1} + \delta_{m-n,-1})
$$
we eventuall end up with the Fourier coefficient at harmonic $n$ as
$$
\Gamma_{i \rightarrow j}^{(n)}(t)=-\frac{1}{e^2 R_{i \rightarrow j}} \sum_{l=0}^{\infty} e^{-\frac{lA_{ij}}{k_B T}}\bigg[A_{ij}(-1)^nI_n\bigg(\frac{lB_{ij}}{k_B T}\bigg) + \frac{B_{ij}}{2}(-1)^{n-1}I_{n-1}\bigg(\frac{lB_{ij}}{k_B T}\bigg) + \frac{B_{ij}}{2}(-1)^{n+1}I_{n+1}\bigg(\frac{lB_{ij}}{k_B T}\bigg)\bigg]
$$
or
$$
\Gamma_{i \rightarrow j}^{(n)}(t)=-\frac{(-1)^n}{e^2 R_{i \rightarrow j}} \sum_{l=0}^{\infty} e^{-\frac{lA_{ij}}{k_B T}}\bigg[A_{ij}I_n\bigg(\frac{lB_{ij}}{k_B T}\bigg) - \frac{B_{ij}}{2}\bigg(I_{n-1}\bigg(\frac{lB_{ij}}{k_B T}\bigg) + I_{n+1}\bigg(\frac{lB_{ij}}{k_B T}\bigg)\bigg)\bigg]
$$
### Some physical consequences
- We have a "Photon-assisted" (= drive-assisted) tunneling process where higher harmonics $n\omega_0$ arise from multi-"photon" processes of order $l*n$
  - $l$ counts how many "quanta" of $B\cos(\omega_0t)$ are effectively absorbed or emitted before tunneling occurs (weighted by $\exp(-lA/k_BT)$)
  - $I_n(lB/k_BT)$ is the amplitude for converting $l$ quanta into an $n$-th harmonics in the rate.
- There are two competing temperature effects in our formula $\Gamma^{(n)} \propto \sum_l e^{-lA/k_BT}I_n(\frac{lB}{k_BT})$ where $A \propto e^2/2C$ is the charging energy scale and $B \propto e U_0$ is the drive amplitude.
  - At $k_BT \ll A$ the exp-factor strongly suppresses all $l \ge 1$ terms $\rightarrow$ virtually no tunneling at all (deep Coulomb blockade). Even tough $I_n$ higher Bessel orders are favored, there simply aren't any events to dress with harmonics. **Outcome:** Spectrum collapses.
  - At $k_BT \gg A$ we get $e^{-lA/k_BT} \approx 1$ for many $l$ which means that lots of processes are thermally allowed. But as the argument of $I_n$ becomes small and $I_n(x) \propto x^{n}/n!$ decays rapidly for higher harmonics. **Outcome:** Thermal smearing kills nonlinearity, harmonics beyond $n=1,2$ are negligible.
  - In the intermediate range of $k_BT \gtrsim A$ and $k_BT \lesssim B \approx eU_0$ large amount of $l$ aren't frozen and $I_n$ isn't tiny. In practice we want $\frac{A}{k_BT} \approx O(1)$ and $\frac{B}{k_BT} \approx O(1)$. **Outcome:** Rich Harmonic spectrum, enough thermal energy for tunneling, but at stron nonlinearity for higher order Bessel peaks. **The shape of our nonlinear tunneling-rate spectrum and hence the output current is fully captured by the dimensionless ratios** $$\frac{A}{k_B T} \text{ and } \frac{B}{k_B T}=\frac{eU_0}{k_B T}$$
- Notice $AI_n + \frac{B}{2}(I_{n-1}-I_{n+1})$ and the overall $(-1)^n$. Without a DC charging offset $A$ and symmetrical drive $B\cos(\omega_0t)$ each coefficient of even $n$ vanishes, leaving only odd harmonics. **Asymmetry will contribute to odd harmonics!**