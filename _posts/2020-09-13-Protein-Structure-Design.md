---
layout: post
title:  "Protein Structure Design"
date:   2020-09-13 10:00:00
categories: Research Protein ML
excerpt_separator: <!--more-->
---

Walkthrough of how to use MaSIF to search for DNA mimicking proteins in the PDB.

<!--more-->

### MaSIF Search

All the necessary code can be found here: https://github.com/tianyu-lu/protein-design. The main script is `get_mimics_surf.py` which runs all the preprocessing steps required to generate MaSIF fingerprints for each pdb file in the current working directory. On Compute Canada, the shell script to submit is `get_surf.sh`. This script starts a Singularity container (the Docker equivalent on Compute Canada) and runs `get_mimics_surf.py`. Currently the script records in `results.out` the residue indices that are covered by the five patches with the smallest Euclidean distance to ebox fingerprints, along with the pdb id the patches belong to and the Euclidean distances. The two ebox fingerprints have indices 274 and 2333 (line 183 of `get_mimics_surf.py`). Visualized below, these are the patches that interact with Myc and Max. 

### Input Features for DNA Surfaces

Each of the following files were modified to extend MaSIF's preprocessing to DNA surfaces.

`chemistry.py`: lines 41-49, 69-93.
`computeHydrophobilicy.py`: lines 26-29.
`xyzrn.py`: line 25

#### Utility Scripts

`delete.py` removes all the intermediate files and keeps the original pdb files and the final `results.out` file. In particular it removes the `*_ignh.pdb` files generated by the Reduce program. Removing intermediate files is needed to save disk space and to keep below Cedar/Graham's 1 million files quota. 

`get_results.py` moves all the `results.out` files from each PDB subfolder into a results folder.

`unfinished.py` let you compute the `results.out` file even if not all the surface patches in a subfolder have been computed. 

`get_ebox_surf.py` computes the MaSIF fingerprints for all vertices on the ebox DNA. 

#### Caveats

1. `pdb2pqr` requires an absolute path to the location of its install. 
2. The order of `import pymesh` with other import statements matters. Pymesh should be imported first.
3. `results.out` does not record which of the two ebox patchs the fingerprint matches.
4. The code has only been tested on individual chains of PDB structures. Nevertheless, the code should still work for matching surfaces of dimers, trimers, etc.
5. On Compute Canada, you must have numpy version 1.16.1, otherwise some imports will not work. If you encounter this error `Could not install packages due to an EnvironmentError: Could not find a suitable TLS CA certificate bundle, invalid path: /etc/pki/tls/certs/ca-bundle.crt`, you can email Compute Canada and have someone give you a copy of the `CC ca.bundle.crt`. After that, running ```
singularity run docker://pablogainza/masif
export REQUESTS_CA_BUNDLE="/home/<USERNAME>/.config/pip/ca-bundle.crt"
pip3.6 install --upgrade --user numpy==1.16.1
```
should do the trick.

### Protein Backbone Search and Design Slides

<iframe src="https://utoronto-my.sharepoint.com/personal/tianyu_lu_mail_utoronto_ca/_layouts/15/Doc.aspx?sourcedoc={4e2f50a3-677a-48da-91a2-be9bd8ff47bf}&amp;action=embedview&amp;wdAr=1.7777777777777777" width="610px" height="367px" frameborder="0">This is an embedded <a target="_blank" href="https://office.com">Microsoft Office</a> presentation, powered by <a target="_blank" href="https://office.com/webapps">Office</a>.</iframe>

### Classical Protein Design Pipeline

Steps involved in generating protein sequences to optimize binding affinity to another protein. Covers docking with HADDOCK, molecular dynamics simulations with GROMACS, simulated annealing in RosettaDesign, and free energy calculations using non-equilibrium MD for filtering.

<iframe src="https://utoronto-my.sharepoint.com/personal/tianyu_lu_mail_utoronto_ca/_layouts/15/Doc.aspx?sourcedoc={8fa06acd-7e34-4059-88ff-68011c50000e}&amp;action=embedview&amp;wdAr=1.7777777777777777" width="610px" height="367px" frameborder="0">This is an embedded <a target="_blank" href="https://office.com">Microsoft Office</a> presentation, powered by <a target="_blank" href="https://office.com/webapps">Office</a>.</iframe>

