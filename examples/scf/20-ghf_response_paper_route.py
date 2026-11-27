#!/usr/bin/env python

'''Direct-AO GHF J/K response through the DEFK2021 paper route.

Usage:
    python 20-ghf_response_paper_route.py [outdir]

The example needs the qcbench-pyscf-routes fork on PYTHONPATH, or installed.
For GKS, use dft.GKS and set its jk_route attribute in the same way.
'''

import sys

import numpy
import pyscf
from pyscf import gto, scf
from pyscf.scf import ghf

BASIS = 'cc-pVDZ'


def main(outdir):
    if not hasattr(ghf.GHF, 'record_binary'):
        raise SystemExit('This example needs the fork '
                         '(pyscf/scf/paper_jk_response.py). The imported pyscf '
                         'is %s' % pyscf.__file__)
    print('pyscf %s from %s' % (pyscf.__version__, pyscf.__file__))
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis=BASIS, symmetry=False, verbose=0)
    nao = mol.nao
    rng = numpy.random.default_rng(20261105)
    z = rng.standard_normal((2 * nao, 2 * nao))
    z = z + 1j * rng.standard_normal((2 * nao, 2 * nao))
    response_density = z + z.conj().T

    mf = scf.GHF(mol)
    mf.jk_route = 'paper'
    with mf.record_binary(outdir, all_calls=True, print_calls=False,
                          system=dict(perturbation_order=1)) as rec:
        vj, vk = mf.get_jk(mol, response_density, hermi=1)

    mf_ref = scf.GHF(mol)
    vj_ref, vk_ref = mf_ref.get_jk(mol, response_density, hermi=1)
    print('max|C - C_default| = %.2e   max|K - K_default| = %.2e'
          % (numpy.max(numpy.abs(vj - vj_ref)), numpy.max(numpy.abs(vk - vk_ref))))
    print('max|(C-K) - (C-K)_default| = %.2e'
          % numpy.max(numpy.abs((vj - vk) - (vj_ref - vk_ref))))
    print('%d contractions saved to %s (see manifest.json there)'
          % (rec.n_calls, outdir))
    for call in rec.calls:
        print('    %-28s spin=%-5s %-12s A%-18s B%-12s C%s'
              % (call['case'], call['spin_block'], call['equation'],
                 call['A']['shape'], call['B']['shape'], call['C']['shape']))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'ghf_response_paper_route_tensors')
