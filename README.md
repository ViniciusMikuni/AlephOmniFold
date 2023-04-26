# Aleph OmniFold implementation

To run a single [OmniFold](https://dx.doi.org/10.1103/PhysRevLett.124.182001) iteration just run:

```bash
cd scripts
python aleph.py
```

The number of iterations and other parameters are stored in the config file named ```config_omnifold.json```

The Unfolded results can be made using the plotting script:

```bash
python plot.py
```

# Running Bootstraps
You can run each individual bootstrap with:

```bash
python aleph.py --strapn X
```

where X represents an integer between 1 and the total number of bootstraps you plan to run. When X=0 no boostraps are run (default). Similarly, to evaluate the bootstrap you can run:

```bash
python plot.py --strapn X
```

