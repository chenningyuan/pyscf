#!/usr/bin/env python

'''
GDF Hartree-Fock with k-points through the binary contractions of the paper route

J and K of a closed-shell k-point density and the electronic energy are built
as the binary contractions of Ye and Berkelbach (2026, arXiv:2607.28554):
density Eq. (82), Coulomb Eqs. (85a)-(85b), exchange Eqs. (86a)-(86b), energy
Eq. (84).  Every tensor is held and contracted whole, and the operands of
every contraction can be printed and saved.

Change the system in make_cell(); nothing else needs to be touched.

Usage:
    python 20-k_points_gdf_paper_route.py [outdir]

The example needs this fork on the import path (PYTHONPATH), or the fork
installed.  Compiled libraries of an installed PySCF 2.14.0 can be linked
instead of building this fork:

    ln -s <installed pyscf>/lib/*.so <this fork>/pyscf/lib/
'''

import sys

import numpy
import pyscf
from pyscf import lib
from pyscf.pbc import gto, scf
from pyscf.pbc.df import GDF

BASIS = 'gth-dzvp'
PSEUDO = 'gth-pade'
# k-point pair whose operands are saved: one pair per binary contraction.
# kptij='all' saves every call.
KPTIJ = (0, 1)


def make_cell():
    # diamond primitive cell; any cell, basis and pseudopotential work
    cell = gto.Cell()
    cell.atom = 'C 0 0 0; C 0.89175 0.89175 0.89175'
    cell.a = numpy.eye(3) * 3.567
    cell.basis = BASIS
    cell.pseudo = PSEUDO
    cell.verbose = 4
    return cell.build()


def main(outdir):
    if not hasattr(GDF, 'record_binary'):
        raise SystemExit('This example needs the fork (pyscf/pbc/df/paper_jk.py) on the '
                         'import path.  The imported pyscf is %s' % pyscf.__file__)
    print('pyscf %s from %s' % (pyscf.__version__, pyscf.__file__))

    cell = make_cell()
    kpts = cell.make_kpts([2, 1, 1], scaled_center=[.125, .2, .075])

    # 1. The SCF through the paper route: its Fock matrix and its energy are
    #    built by the binary contractions in every iteration.
    mf = scf.KRHF(cell, kpts).density_fit(auxbasis='weigend')
    mf.with_df.jk_route = 'paper'
    mf.conv_tol = 1e-11
    mf.run()

    # 2. The same system with the default implementation of PySCF.
    mf_ref = scf.KRHF(cell, kpts).density_fit(auxbasis='weigend')
    mf_ref.conv_tol = 1e-11
    mf_ref.run()
    print('E_tot   paper route %.12f   default %.12f   difference %.2e' %
          (mf.e_tot, mf_ref.e_tot, abs(mf.e_tot - mf_ref.e_tot)))

    # 3. The operands of the converged orbitals, printed and saved.
    #    exxdiv=None gives the bare J and K of Eqs. (85b) and (86b).
    mydf = mf.with_df
    hcore = mf.get_hcore()
    # the canonical occupied orbitals are passed by tagging the density
    dm = lib.tag_array(mf.make_rdm1(), mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
    with mydf.record_binary(outdir, kptij=KPTIJ,
                            system=dict(basis=BASIS, pseudo=PSEUDO)) as rec:
        vj, vk = mydf.get_jk(dm, hermi=1, exxdiv=None)
        e_h, e_j, e_k = mydf.energy_elec_paper_route(dm, hcore, vj, vk)

    dm_ref = mf_ref.make_rdm1()
    vj_ref, vk_ref = mf_ref.with_df.get_jk(dm_ref, hermi=1, exxdiv=None)
    e_ref = mf_ref.energy_elec(dm_kpts=dm_ref, h1e_kpts=mf_ref.get_hcore(),
                               vhf_kpts=vj_ref - vk_ref * .5)
    print('max|J - J_default| = %.2e   max|K - K_default| = %.2e' %
          (numpy.max(numpy.abs(vj - vj_ref)), numpy.max(numpy.abs(vk - vk_ref))))
    print('E_h = %.12f   E_J = %.12f   E_K = %.12f' % (e_h, e_j, e_k))
    print('E_elec (without the Ewald term)  paper route %.12f   default %.12f   '
          'difference %.2e' % (e_h + e_j + e_k, e_ref[0], abs(e_h + e_j + e_k - e_ref[0])))
    print('%d contractions printed, %d saved to %s (see manifest.json there)' %
          (rec.n_calls, len(rec.calls), outdir))
    for call in rec.calls:
        print('    %-20s %-12s k=%-6s A%-16s B%-16s C%s' %
              (call['case'], call['equation'], '-'.join(map(str, call['kptij'])),
               call['A']['shape'], call['B']['shape'], call['C']['shape']))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'gdf_paper_route_tensors')
