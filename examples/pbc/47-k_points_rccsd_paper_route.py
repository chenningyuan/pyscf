#!/usr/bin/env python

'''Periodic non-RI RCCSD through the HPTB2004 paper-route binaries.

Both calculations use the same complex two-k-point reference and full MO
integrals.  The opt-in paper route performs every tabled F/L/W, T1/T2, and
energy contraction through its recordable binary dispatcher; native PySCF
provides the reference and the outer CCSD iteration/denominator machinery.

Usage:
    python 47-k_points_rccsd_paper_route.py [outdir]
'''

import sys

import numpy
import pyscf
from pyscf.pbc import cc, gto, scf

KPTIJ = None  # 'all' saves every iteration; an exact tuple selects one call


def make_cell():
    cell = gto.Cell()
    cell.atom = 'He 0 0 0; He 1.685068664391 1.685068664391 1.685068664391'
    cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
    cell.a = '''0 3.370137329 3.370137329
                3.370137329 0 3.370137329
                3.370137329 3.370137329 0'''
    cell.unit = 'B'
    cell.mesh = [15] * 3
    cell.precision = 1e-9
    cell.verbose = 0
    return cell.build()


def main(outdir):
    if not hasattr(cc.kccsd_rhf.RCCSD, 'record_binary'):
        raise SystemExit('This example needs the qcbench-pyscf-routes fork; imported %s'
                         % pyscf.__file__)
    print('pyscf %s from %s' % (pyscf.__version__, pyscf.__file__))
    cell = make_cell()
    kpts = cell.make_kpts([2, 1, 1], scaled_center=[.125, .2, .075])
    mf = scf.KRHF(cell, kpts, exxdiv=None).run(conv_tol=1e-11)
    assert mf.converged

    native = cc.KRCCSD(mf)
    native.direct = False
    native.conv_tol = 1e-9
    native.kernel()
    assert native.converged

    paper = cc.KRCCSD(mf)
    paper.direct = False
    paper.ccsd_route = 'paper'
    paper.conv_tol = 1e-9
    with paper.record_binary(outdir, kptij=KPTIJ, print_calls=False,
                             system={'basis': 'custom He s basis', 'kmesh': (2, 1, 1)}) as rec:
        paper.kernel(eris=native.eris)
    assert paper.converged
    print('Native/paper correlation energies: %.12f %.12f' % (native.e_corr, paper.e_corr))
    print('max|T1 paper - native| = %.3e' % numpy.max(numpy.abs(paper.t1 - native.t1)))
    print('max|T2 paper - native| = %.3e' % numpy.max(numpy.abs(paper.t2 - native.t2)))
    print('%d contractions, %d case types saved to %s'
          % (rec.n_calls, len(rec.calls), outdir))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'rccsd_paper_route_tensors')
