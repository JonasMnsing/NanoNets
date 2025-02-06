# Outline: Time dependence and disorder

**Goal**: We want to show that our system is able to process time series data to eventually predict or approximate dynamical systems.

## Introdcution

- Reference to first simulation paper (NP-Network as function approximator)
- Now, given its inherent memory (recurrent topology) we should be able to not only model $y=f(x)$ as before, but also $y=f(x,t)$ with network output $y$ and network input $x$ at time $t$
- Reference to Physical Computing / Reservoir Computing. Our system does **not** rely on training some external device (linear superposition in RC) but we only need to train our controls once to let the system itself output $y=f(x,t)$. 
- Reference about the importance of variable time scales, i.e. reason why we are interested in disorder

## Theory/Model

- Kinetic Monte Carlo for time dependent state evaluation $\vec{q}(t)$
- Definition of Disorder
  - Capactitance disorder (see. Capacitance Calculation)
  - Resistance disorder
  - topology disorder (skip this part?)
- Target variable now potential of a floating electrode or should we still stick to electric current of a grounded electrode. (Motivation Network of Networks)

## Results

- Defining the time scale of the system given uniform and disorderd properties using a step input function (see step input results)
- Defining nonlinear properties associated to time series processing such as Higher Harmonic generation / beat frequencies as well as their dependence on disorder (see harmonic generation results)
- NDR / NLS based on disorder (see NDR /NLS results)
- Memory Capacity given trained control electrode voltages
- Some benchmark example