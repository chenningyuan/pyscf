#!/usr/bin/env python

'''Two-component RI-JK through the Armbruster et al. paper route.

Usage:
    python 20-ghf_paper_route.py [outdir]

The example needs the qcbench-pyscf-routes fork on PYTHONPATH, or installed.
For an SCF calculation, set mf.with_df.jk_route = 'paper' before mf.run().
'''

import sys

import numpy
import pyscf
from pyscf import gto, scf
from pyscf.df import DF

BASIS = 'sto-3g'
AUXBASIS = 'cc-pvdz-jkfit'


def make_molecule():
    return gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                 basis=BASIS, symmetry=False, verbose=0)


def complex_spinors(mol):
    rhf = scf.RHF(mol).run(conv_tol=1e-12)
    nao = mol.nao_nr()
    nso = 2 * nao
    nocc = 2 * int(numpy.count_nonzero(rhf.mo_occ > 0))
    spinor_mo = numpy.zeros((nso, nso))
    spinor_mo[:nao, :nao] = rhf.mo_coeff
    spinor_mo[nao:, nao:] = rhf.mo_coeff
    rng = numpy.random.default_rng(20261103)
    q, _ = numpy.linalg.qr(rng.standard_normal((nso, nso))
                           + 1j * rng.standard_normal((nso, nso)))
    return spinor_mo @ q[:, :nocc]


def main(outdir):
    if not hasattr(DF, 'record_binary'):
        raise SystemExit('This example needs the fork (pyscf/df/paper_jk.py). '
                         'The imported pyscf is %s' % pyscf.__file__)
    print('pyscf %s from %s' % (pyscf.__version__, pyscf.__file__))
    mol = make_molecule()
    coefficients = complex_spinors(mol)

    mf = scf.GHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.with_df.jk_route = 'paper'
    dm = mf.make_rdm1(mo_coeff=coefficients,
                      mo_occ=numpy.ones(coefficients.shape[1]))
    with mf.with_df.record_binary(outdir, print_calls=False) as rec:
        vj, vk = mf.get_jk(mol, dm, hermi=1)
        energy = mf.with_df.energy_elec_paper_route(dm, mf.get_hcore(), vj - vk)

    mf_ref = scf.GHF(mol).density_fit(auxbasis=AUXBASIS)
    vj_ref, vk_ref = mf_ref.get_jk(mol, dm, hermi=1)
    energy_ref = mf_ref.energy_elec(dm=dm, h1e=mf_ref.get_hcore(),
                                    vhf=vj_ref - vk_ref)[0]
    print('max|J - J_default| = %.2e   max|K - K_default| = %.2e'
          % (numpy.max(numpy.abs(vj - vj_ref)), numpy.max(numpy.abs(vk - vk_ref))))
    print('E_elec paper route %.12f   default %.12f   difference %.2e'
          % (energy.real, energy_ref, abs(energy.real - energy_ref)))
    print('imaginary component before taking Re = %.2e' % abs(energy.imag))
    print('%d contractions, %d saved to %s (see manifest.json there)'
          % (rec.n_calls, len(rec.calls), outdir))
    for call in rec.calls:
        print('    %-28s %-12s A%-16s B%-16s C%s'
              % (call['case'], call['equation'], call['A']['shape'],
                 call['B']['shape'], call['C']['shape']))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'ghf_paper_route_tensors')
