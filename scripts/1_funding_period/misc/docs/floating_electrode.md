# Potential estimation for a floating electrode
---

For a given nanoparticle network of $N_{NP}$ nanoparticles and $N_e$ electrodes, one electrode voltage is choosen to be floating, i.e variable during the simulation. The remaining electrodes might also vary in time, but are constant during the *KMC* procedure at a specific time step. As the floating electrode is connected to a particular nanoparticle it will depend on the nanoparticle's potential $\phi_{NP}$. 

The nanoparticle as it sits on an insulating $SiO_2$ environment has the ability to store charges by itself (isolated) defined by its self capacitance $C_{self} = 4\pi\epsilon_{SiO_2}r_{NP}$. The interaction between the charges on nanoparticle and electrode is represented by its mutual capacitance $C_{}


$$U_{out} = \frac{C_i}{C_{self}}$$


