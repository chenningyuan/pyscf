#!/usr/bin/env python

'''Direct SOMF/DL-CPSCF contractions through the GVO2019 paper route.

Usage:
    python 20-somf_cpscf_paper_route.py [outdir]

The example needs the qcbench-pyscf-routes fork on PYTHONPATH, or installed.
It evaluates one DL-CPSCF response step at each of two Laplace points, not a
converged response or a complete g-tensor.
'''

import sys

import numpy
import pyscf
from pyscf import gto, scf
from pyscf.scf import paper_somf_cpscf

BASIS = 'sto-3g'
LAPLACE_POINTS = ((.13, .6), (.31, .4))
EXCHANGE_FRACTION = .2
ELECTRON_SPIN = .5


def main(outdir):
    print('pyscf %s from %s' % (pyscf.__version__, pyscf.__file__))
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis=BASIS, unit='Angstrom', symmetry=False, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    nao = mol.nao_nr()
    rng = numpy.random.default_rng(20261103)
    spin_response = .02 * (rng.standard_normal((nao, nao))
                           + 1j * rng.standard_normal((nao, nao)))
    response_density = .02 * (rng.standard_normal((nao, nao))
                              + 1j * rng.standard_normal((nao, nao)))
    response_density -= response_density.T
    h_bp = .02 * (rng.standard_normal((nao, nao))
                  + 1j * rng.standard_normal((nao, nao)))

    route = paper_somf_cpscf.SOMFCPSCF(
        mf, exchange_fraction=EXCHANGE_FRACTION, electron_spin=ELECTRON_SPIN)
    with route.record_binary(outdir, all_calls=True, print_calls=False) as rec:
        result = paper_somf_cpscf.paper_route(
            route, LAPLACE_POINTS, h_bp, response_density, spin_response)

    density = mf.make_rdm1().astype(numpy.complex128)
    gsoc = mol.intor('int2e_p1vxp1', comp=3, aosym='s1').reshape(
        3, nao, nao, nao, nao).astype(numpy.complex128)
    factor_j = -paper_somf_cpscf.ALPHA**2 * paper_somf_cpscf.GPRIME / 4
    factor_k = 3 * paper_somf_cpscf.ALPHA**2 * paper_somf_cpscf.GPRIME / 8
    ref_j = factor_j * numpy.einsum('quvrs,rs->quv', gsoc, density)
    ref_k = (factor_k * numpy.einsum('qurvs,rs->quv', gsoc, density)
             + factor_k * numpy.einsum('qsvur,rs->quv', gsoc, density))
    ref_response_k = EXCHANGE_FRACTION * mf.get_k(mol, response_density.T, hermi=0)
    print('SOMF J error = %.2e   SOMF K error = %.2e   CPSCF K error = %.2e'
          % (numpy.max(numpy.abs(result.somf.coulomb - ref_j)),
             numpy.max(numpy.abs(result.somf.exchange - ref_k)),
             numpy.max(numpy.abs(result.steps[0].exchange_response - ref_response_k))))
    response_error = max(numpy.max(numpy.abs(
        step.laplace_response - step.virtual_pseudo_density
        @ step.response_matrix @ step.occupied_pseudo_density))
        for step in result.steps)
    print('maximum Laplace response error = %.2e' % response_error)
    print('SOMF-J property =', result.property)
    print('%d contractions saved to %s (see manifest.json there)'
          % (rec.n_calls, outdir))
    for call in rec.calls:
        print('    %-32s q=%-4s tau=%-4s %-12s'
              % (call['case'], call['component'], call['laplace'], call['equation']))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'somf_cpscf_paper_route_tensors')
