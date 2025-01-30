# Capacitance Calculation
---
## NP-NP Interaction
For two nanoparticles $i$ and $j$ are assumed to be spherical conductors with radius $r_i$ and $r_j$ sit in an insulating environment of permittivity $\epsilon_{env}$. The spheres are connected to each other by an insulating molecule of permittivity $\epsilon_{mol}$. The distance between the sphere's centers is $d$.

For an isolated sphere in an environment with permittivity $\epsilon_{env}$, the self-capacitance is straightforward:

$$C_{ii} = 4\pi\epsilon_{env}r_i$$

The mutual capacitance between two spheres depends on both their sizes and the distance between them. When the spheres are not in contact, the mutual capacitance can be expressed as a series expansion arising from solving the Laplace's equation for the potential in between, where each term in the series represents the interaction between higher-order multipoles on the spheres:

$$C_{ij} = 2\pi\epsilon_{mol}(\frac{r_ir_j}{d})\sum_{n=0}^\infty(\frac{r_i^n+r_j^n}{d^n})$$

## NP-Electrode Interaction

When estimating the nanoparticle-electrode interaction one could assume that the electrode denoted $j$ is much larger than nanoparticle $i$, i.e. $r_j \rightarrow \infty$ resembling a spherical conductor near a conducting plane. This reduces the mutual capacitance to

$$ C_{ij} = 2\pi\epsilon_{mol}r_i $$

Eventually, if now one wants to estimate the potential of an floating electrode $U$ which is attached to a single nanoparticle $i$ of potential $\phi_i$ we get

$$U = \frac{C_{ii}}{C_{ii}+C_{ij}}\phi_i$$
