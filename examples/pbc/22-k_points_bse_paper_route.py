#!/usr/bin/env python

'''Periodic static RI-BSE through the binary contractions of Zhou et al. (2025).

PySCF 2.14 provides molecular BSE but no native k-point BSE driver.  This
example uses the fork's periodic paper route for response Eq. (44), screening
Eqs. (39)-(40), kernels Eqs. (45)-(46), and the TDA response/eigensolve.  The
default alpha=1 is the spatial unit-kernel validation; use alpha=2 for singlet
or alpha=0 for triplet, with appropriate QP energies for a production BSE@GW
calculation.

Usage:
    python 22-k_points_bse_paper_route.py [outdir]
'''

import sys

import numpy
import pyscf
from pyscf.pbc import gto, scf
from pyscf.pbc.gw import paper_krbse

BASIS = 'gth-dzvp'
AUXBASIS = 'weigend'
PSEUDO = 'gth-pade'
KMESH = (2, 1, 1)
ALPHA = 1.0
KPTIJ = None  # (0, 1) saves that pair; 'all' saves every call


def make_cell():
    cell = gto.Cell()
    cell.atom = 'C 0 0 0; C 0.89175 0.89175 0.89175'
    cell.a = numpy.eye(3) * 3.567
    cell.basis = BASIS
    cell.pseudo = PSEUDO
    cell.verbose = 0
    return cell.build()


def main(outdir):
    if not hasattr(paper_krbse, 'KRBSE'):
        raise SystemExit('This example needs the qcbench-pyscf-routes fork; imported %s'
                         % pyscf.__file__)
    print('pyscf %s from %s' % (pyscf.__version__, pyscf.__file__))
    cell = make_cell()
    kpts = cell.make_kpts(KMESH, scaled_center=[.125, .2, .075])
    mf = scf.KRHF(cell, kpts).density_fit(auxbasis=AUXBASIS)
    mf.conv_tol = 1e-11
    mf.run()
    assert mf.converged

    bse = paper_krbse.KRBSE(mf, alpha=ALPHA)
    with bse.record_binary(outdir, kptij=KPTIJ, print_calls=False,
                           system={'basis': BASIS, 'auxbasis': AUXBASIS,
                                   'pseudo': PSEUDO, 'kmesh': KMESH}) as rec:
        paper_krbse.paper_route(bse)
        x = numpy.ones((bse.nkpts, bse.nocc, bse.nvir), dtype=complex)
        paper_krbse.sigma_response(bse, bse.a_matrix, x)
        exci, _, _ = bse.kernel()
    print('Lowest three TDA eigenvalues:', exci[:3])
    print('%d contractions, %d saved to %s (see manifest.json)'
          % (rec.n_calls, len(rec.calls), outdir))
    for call in rec.calls:
        print('    %-24s %-12s k=%-6s q=%-3s A%-16s B%-16s C%s'
              % (call['case'], call['equation'], '-'.join(map(str, call['kptij'])),
                 call['q'], call['A']['shape'], call['B']['shape'], call['C']['shape']))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'bse_paper_route_tensors')
