# MACEModels.jl

[![Build Status](https://github.com/alexsp32/MACEModel.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/alexsp32/MACEModel.jl/actions/workflows/CI.yml?query=branch%3Amain)

This package provides an interface between NQCModels.jl and the MACE code, allowing for the use of pre-trained models for dynamics simulations. By reducing the number of copying operations between the two packages, performance is very slightly improved, while also enabling automatic setup of the Julia-Python interface through CondaPkg.jl. 

**The current version of this package uses MACE v0.3.3 to evaluate models. Any models trained on later versions may lead to unexpected or incorrect results.**

Currently, only the evaluation of forces and energies is supported. Models that yield additional information might cause errors. 
