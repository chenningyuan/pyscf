#!/usr/bin/env python

'''Canonical periodic KRCCSD(T) through the four-case paper route.

The native KRCCSD calculation supplies converged amplitudes and full ERIs.
The slow native (T) kernel is the reference; the opt-in paper route records
both connected W terms, the disconnected T1 term, and the pw/rw energy binary.
Non-HF references require an additional fov*t2 term and are rejected by this
four-case canonical corpus route.

Usage:
    python 48-k_points_rccsd_t_paper_route.py [outdir]
'''

import sys

import numpy
import pyscf
from pyscf.pbc import cc, gto, scf
from pyscf.pbc.cc import kccsd_t_rhf_slow

KPTIJ = None  # 'all' saves every momentum sector


def make_cell():
    cell = gto.Cell()
    cell.atom = 'He 0 0 0; He 1.7 0 0'
    cell.basis = 'gth-dzv'
    cell.pseudo = 'gth-pade'
    cell.a = numpy.diag([3.4, 3.4, 3.4])
    cell.verbose = 0
    return cell.build()


def main(outdir):
    if not hasattr(cc.kccsd_rhf.RCCSD, 'record_binary'):
        raise SystemExit('This example needs the qcbench-pyscf-routes fork; imported %s'
                         % pyscf.__file__)
    cell = make_cell()
    kpts = cell.make_kpts([1, 1, 1])
    mf = scf.KRHF(cell, kpts=kpts, exxdiv=None).run(conv_tol=1e-12)
    mycc = cc.KRCCSD(mf)
    mycc.conv_tol = 1e-11
    mycc.conv_tol_normt = 1e-10
    mycc.max_cycle = 300
    eris = mycc.ao2mo()
    mycc.kernel(eris=eris)
    assert mycc.converged
    reference = kccsd_t_rhf_slow.kernel(mycc, eris, verbose=0)

    mycc.ccsd_t_route = 'paper'
    with mycc.record_binary(outdir, route='ccsd_t', print_calls=False,
                            system={'basis': 'gth-dzv', 'kmesh': (1, 1, 1)}) as rec:
        energy = mycc.ccsd_t(eris=eris)
    print('Native/paper (T) corrections: %.15g %.15g' % (reference, energy))
    print('Absolute error: %.3e' % abs(reference - energy))
    print('%d contractions, %d case types saved to %s' % (rec.n_calls, len(rec.calls), outdir))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'rccsd_t_paper_route_tensors')
