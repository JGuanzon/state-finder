In the [statefindercode.ipynb](https://github.com/JGuanzon/state-finder/blob/main/statefindercode.ipynb), we use machine learning to optimise the parameters of an optical circuit towards generating high quality quantum resource states (in particular, we employ the global optimisation algorithm *basinhopping*). These resource states are needed to perform entanglement purification in a linear-optical system, as discussed in our paper "Achieving the ultimate end-to-end rate of a lossy quantum communication network" [1]. Already optimised parameter sets are found in the statefinderdata folder.

We acknowledge the papers "Production of photonic universal quantum gates enhanced by machine learning" [[2](https://doi.org/10.1103/PhysRevA.100.012326)] and "Progress towards practical qubit computation using approximate Gottesman-Kitaev-Preskill codes" [[3](https://doi.org/10.1103/PhysRevA.101.032315)], whose code we have used and modified here for our own purposes. We also acknowledge the libraries *strawberryfields* [[4](https://doi.org/10.22331/q-2019-03-11-129)] and *thewarlus* [[5](https://doi.org/10.21105/joss.01705)], which we used to perform the quantum simulation. 

[1] M. S. Winnel, J. J. Guanzon, N. Hosseinidehaj, and T. C. Ralph, "Achieving the ultimate end-to-end rate of a lossy quantum communication network," *to be published* (2021). \
[2] K. K. Sabapathy, H. Qi, J. Izaac, and C. Weedbrook, "Production of photonic universal quantum gates enhanced by machine learning," [Physical Review A **100**, 012326 (2019)](https://doi.org/10.1103/PhysRevA.100.012326). \
[3] I. Tzitrin, J. E. Bourassa, N. C. Menicucci, and K. K. Sabapathy, "Progress towards practical qubit computation using approximate Gottesman-Kitaev-Preskill codes," [Physical Review A **101**, 032315 (2020)](https://doi.org/10.1103/PhysRevA.101.032315). \
[4] N. Killoran, J. Izaac, N. Quesada, V. Bergholm, M. Amy, and C. Weedbrook, "Strawberry Fields: a software platform for photonic quantum computing," [Quantum **3**, 129 (2019)](https://doi.org/10.22331/q-2019-03-11-129). \
[5] B. Gupt, J. Izaac, and N. Quesada, "The Walrus: a library for the calculation of hafnians, Hermite polynomials and Gaussian boson sampling," [Journal of Open Source Software **4**, 1705 (2019)](https://doi.org/10.21105/joss.01705).  
