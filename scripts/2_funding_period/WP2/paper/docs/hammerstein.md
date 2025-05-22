## Hammerstein Model Identification Workflow

This document outlines the mathematical stages and workflow for building a Hammerstein model from experimental input–output data.

### 1. Static Nonlinearity Extraction

At sufficiently low excitation frequency, the system behaves as memoryless, so the output $y(t)$ satisfies:

$$
 y(t) \approx f_{\mathrm{static}}\bigl(u(t)\bigr)
$$

where:

* $u(t)$ is the input signal.
* $y(t)$ is the measured output.
* $f_{\mathrm{static}}(\cdot)$ is a nonlinear mapping (e.g., piecewise‐linear, polynomial, sigmoidal) that is estimated by fitting $y$ versus $u$ under quasi‐steady‐state conditions.

Once identified, $f_{\mathrm{static}}$ captures all *memoryless* distortion (the low‑frequency harmonic plateau).

---

### 2. Complex Frequency Response Function

For a set of higher test frequencies $\{f_0\}$, record steady‑state time series $u_k=u(t_k)$ and $y_k=y(t_k)$ over an integer number of periods. Compute the complex fundamental amplitudes via synchronous detection:

$$
 Y_1(f_0)
 = \frac{2}{N}\sum_{k=0}^{N-1}\bigl(y_k - \bar y\bigr)\,e^{-j2\pi f_0 t_k},
 \quad
 U_1(f_0)
 = \frac{2}{N}\sum_{k=0}^{N-1}\bigl(u_k - \bar u\bigr)\,e^{-j2\pi f_0 t_k}.
$$

The empirical frequency response is then:

$$
 \widehat G(j\omega_0) = \frac{Y_1(f_0)}{U_1(f_0)},
 \quad \omega_0=2\pi f_0.
$$

This complex-valued data captures both magnitude and phase of the dynamic block in a best linear approximation. Accordingly we throw away higher harmonics, nonlinearity, etc. which means nonlinearity is only produced by the static nonlinearity while $G(s)$ just filters and delays.

---

### 3. Dynamic Block Identification (All‑Pole Model)

Assume the dynamic block is an all‑pole system of order $n$:

$$
 G(s) = \frac{K}{\displaystyle \prod_{i=1}^n\bigl(1 + \tfrac s{p_i}\bigr)},
 \quad p_i>0,
$$

which in the frequency domain becomes:

$$
 G(j\omega)=\frac{K}{\prod_{i=1}^n\bigl(1 + j\omega/p_i\bigr)}.
$$

Fit the parameters $\{K,p_i\}$ by minimizing the least‑squares error over real and imaginary parts:

$$
 \min_{K,\{p_i\}}>0
 \sum_{k}\Bigl|\Re\{G(j\omega_k)\}-\Re\{\widehat G(j\omega_k)\}\Bigr|^2
 +\Bigl|\Im\{G(j\omega_k)\}-\Im\{\widehat G(j\omega_k)\}\Bigr|^2.
$$

Enforcing $p_i>0$ guarantees all poles are in the left half‑plane (stable).

---

### 4. Hammerstein Model Assembly

1. **Static block**: $w(t)=f_{\mathrm{static}}\bigl(u(t)\bigr)$.
2. **Linear block**: $y_{\mathrm{model}}(t) = G(s)\,w(t)$, where $G(s)$ has numerator

   $$
     N(s)=K\prod_{i=1}^n p_i,
   $$

   and denominator

   $$
     D(s)=\prod_{i=1}^n(s + p_i).
   $$

The complete Hammerstein model is therefore the cascade $u\to f_{\mathrm{static}}\to G(s)\to y$.