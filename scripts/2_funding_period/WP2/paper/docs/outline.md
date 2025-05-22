# Paper: Time-Dependent Nanoparticle Networks

## 1. Introduction & Literature Review

* **Background on Time-Independent Nanoparticle Networks**: Recap of equilibrated studies and static nonlinearity.
* **Neuromorphic-Computing Context**: Dynamical response, memory, fading-memory, kernel richness; examples from reservoir computing and spiking networks.
* **Gap & Objectives**: Need for quantitative dynamical characterization under time-varying drive; role of two-type disorder in tuning time scale, nonlinearity, and memory.

## 2. Model & Simulation Tool

### 2.1. Network Geometry & Disorder

* 2D lattice with fixed inter-particle spacing.
* Two NP types: Diameter of a NP defined as $d_i = \mathcal{N}(\mu_{d_i},\sigma_{d_i}^2)$, with $\mu_{d_i} \in \{d_1, d_2\}$ and $\sigma_{d_i}=0$ by default.
* Two Junction types: Resistance of a junctios defined as $R_i = \mathcal{N}(\mu_{R_i},\sigma_{R_i}^2)$, with $\mu_{R_i} \in \{R_1, R_2\}$ and $\sigma_{R_i}=0$ by default.

### 2.2. KMC with Floating Output & Time-Dependent Boundaries

* Original KMC summary and modifications:

  * Floating output node.
  * Time-stepped boundary voltages for input and optional controls.
* Event rates and charge updates under time-dependent boundaries.

## 3. System Characterization

### 3.1. Transient (Step) Response & Tunable Time Scales

* **Protocol**: Single voltage step, record NP and Output potentials.
* **Large Output Capacitance Study**:

  * Vary $C_{ext} > C_{NP}$ to shift $\tau$ into ms regime.
* **Metrics**: Exponential fits to extract $\tau$; plots of $\tau$ vs. $C_{ext}$ and NP/junction types.

### 3.2. Frequency Response & Hammerstein Approximation

* **Protocol**: Sinusoidal sweep $f_0$ from $f_{min}$ to $f_{max}$.
* **Metrics**: Gain $|H(f)|$, phase $\varphi(f)$, quasi-static plateau and roll-off.
* **Modeling**: Fit to Hammerstein block model (static nonlinearity + linear filter).

### 3.3. Nonlinearity & Total Harmonic Distortion

* **Protocol**: Sinusoidal sweep $f_0$ from $f_{min}$ to $f_{max}$.
* **Metrics**: THD vs. frequency, and NP/junction configuration.

### 3.4. Memory Effects

* **Multi-Pulse Protocol**: Two step pulses separated by $\Delta t$; quantify second-response attenuation.
* **Optional Noise Protocol**: White-noise drive; compute short-term memory kernel or autocorrelation decay.
* **Figure**: Memory metric vs. $\Delta t$ and NP/junction configuration.

## 4. Parametric Control via Additional Electrodes

* **8-Electrode Extension**: 1 input, 1 floating output, 6 fixed-bias controls.
* **Protocol**: At given $f_0$ random control voltages or just varying two representative controls for heatmaps.
* **Studies**:

  * Control-bias effects on $\tau$.
  * Control-bias effects on THD/nonlinearity.

## 5. Conclusions & Outlook

* Summary, Applications, Neuromorphic Computing context
* Key takeaways and directions for future work.

## Supplementary

* **RC-Ladder Analogy**:
    * 1D NP string vs. classical RC ladder.
    * $\tau$ and $|H(f)|$ comparisons.
* **Raw output traces and model-fit residuals.**
