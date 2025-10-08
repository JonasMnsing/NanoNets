## Dynamic Response / Input Frequency and Amplitude Dependence

### The Core Physical Mechanism:

The network's behavior is governed by the Coulomb Blockade in a 2D lattice. We have demonstrated that its primary mode of signal transfer is not a simple propagation of charge. Instead, it operates via a long-range "action-at-a-distance" electrostatic gating mechanism. An oscillating charge cluster, localized near the input, creates a time-varying potential field that extends across the entire network, controlling the current at the distant output.

**Black-Box Behavior**: As a signal processing element, the network functions as a nonlinear low-pass filter.

- At low frequencies (e.g., 0.005 MHz), it responds adiabatically, exhibiting a classic "Coulomb staircase" I-V characteristic.

- At high frequencies (e.g., 2.4 GHz), the response time of the network is too slow, and the signal is completely attenuated.

- At intermediate frequencies (e.g., 28 MHz), the network enters a complex dynamic regime, producing a distorted output rich in odd-numbered harmonics, a direct consequence of the network's physical symmetry.

**The Core Physical Mechanism**: We have proven that the complex behavior at 28 MHz is not caused by a simple "charge wave" propagating across the network. Instead, the mechanism is a more subtle, long-range electrostatic control:

- The AC input creates a strongly oscillating, but spatially localized, charge cluster on the nanoparticles immediately adjacent to the input electrode.

- This localized charge cluster acts as a dynamic electrostatic gate.

- Through the long-range capacitive coupling ($\textbf{C}^{-1}$ matrix), this gate generates a time-varying potential field that extends across the entire lattice.

- It is this non-local potential field that drives the current at the distant output electrode. The network acts less like a wire and more like a field-effect transistor.

### A Complete "Phase Diagram" of Functionality:

Our three maps ($G$, $B$, and $THD$) serve as a complete operational "phase diagram" for the network, showing how its function can be tuned by selecting the input amplitude ($U_0$) and frequency ($f_0$).

- Conductance ($G$-Map): This map shows the purely dissipative character of the network. The conductance is always positive ($G>0$), confirming the system is a passive load that consumes energy. The conductance is highest at large amplitudes and intermediate-to-high frequencies, indicating the parameter regime where the network is most "open" for charge transport and energy dissipation.

- Susceptance / Reactive Response ($B$-Map): This map reveals the network's memory and phase-shifting character. It contains a key discovery: the network undergoes a dynamic phase transition from being capacitive ($B>0$) at low frequencies to exhibiting an effective inductive-like response ($B<0$) at high frequencies. This emergent inductance is not due to magnetic fields but is a dynamic effect arising from the collective phase lags of the charge tunneling through the complex potential landscape—a significant finding for a purely electrostatic system.

- Nonlinearity ($THD$-Map): This map identifies the regime for optimal nonlinear signal processing. The "sweet spot" for harmonic generation is at low amplitudes ($U_0 \approx 20−30$ mV) and intermediate frequencies. In this region, the input signal is perfectly scaled to the internal Coulomb energy barriers, maximizing the distortion and complexity of the output.

### Synthesis:
Together, these maps provide a complete datasheet for our network. We can now treat it as a highly tunable, multi-functional circuit element. By choosing an operating point ($U_0,f_0$), we can configure the network to act as:

- A simple dissipative load (high $U_0$, any $f_0$)
- An efficient harmonic generator (low $U_0$, mid $f_0$) 
- A capacitive memory element (any $U_0$, low $f_0$)
- An inductive-like phase shifter (any $U_0$, high $f_0$)