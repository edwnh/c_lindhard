import numpy as np
import ctypes

lib = np.ctypeslib.load_library('liblindhard', '.')
lib.par_eigh.argtypes = [
    np.ctypeslib.ndpointer(np.complex128, flags="C_CONTIGUOUS"), # a on input, eigenvectors on output
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"), # eigenvalues
    ctypes.c_size_t, # n_mat
    ctypes.c_int, # n
    ctypes.c_int, # lower (column major)
    ctypes.c_int, # n_threads
]
lib.par_eigh.restype = None

# NOTE: np.linalg.eigh(A) returns e, U such that A == np.dot(U*e, U.conj().T)
# par_eigh(A) returns e, Udag such that A == np.dot(Udag.conj().T*e, Udag)
# test with the following code

# a = np.random.randn(6, 6) + 1j*np.random.randn(6,6)
# a = a + a.conj().T
# e, u = np.linalg.eigh(a, UPLO='U')

# print(np.max(np.abs(a - np.dot(u*e, u.conj().T))))

# e, u = par_eigh(a.copy(), lower=0)
# print(np.max(np.abs(a - np.dot(u*e, u.conj().T))))
# print(np.max(np.abs(a - np.dot(u.T.conj()*e, u))))

def par_eigh(a, lower=1, n_threads=1):
    start_shape = a.shape
    a.shape = -1, start_shape[-2], start_shape[-1]
    eigvals = np.zeros(a.shape[:2], dtype=np.float64)
    lib.par_eigh(a, eigvals, a.shape[0], a.shape[1], 1 - lower, n_threads) # 1-lower because col <-> row major
    a.shape = start_shape
    eigvals.shape = start_shape[:-1]
    return eigvals, a

C_calc_chi0 = lib.calc_chi0_binned # if desired, change to lib.calc_chi0_exact
C_calc_chi0.argtypes = [
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"), # chi0_real
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"), # chi0_imag
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"), # ek
    np.ctypeslib.ndpointer(np.complex128, flags="C_CONTIGUOUS"), # Ukdag
    np.ctypeslib.ndpointer(np.complex128, flags="C_CONTIGUOUS"), # sub_phase
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"), # ws
    ctypes.c_size_t, # nw
    ctypes.c_double, # gamma
    np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"), # qs
    ctypes.c_size_t, # nq
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, # L3, L2, L1
    ctypes.c_size_t, # Nband
    ctypes.c_size_t, # n_threads
]
C_calc_chi0.restype = None

def calc_chi0(qs, ws, gamma, ek, Ukdag, x_i, n_threads=1):
    L3, L2, L1, Nband, Norb = Ukdag.shape
    assert Norb == Nband

    chi0_real = np.zeros((len(qs), len(ws)), dtype=np.float64)
    chi0_imag = np.zeros((len(qs), len(ws)), dtype=np.float64)
    sublattice_phase = np.exp(-2j*np.pi * (qs[:, 0, None]*x_i[:, 0]/L1 +
                                           qs[:, 1, None]*x_i[:, 1]/L2 +
                                           qs[:, 2, None]*x_i[:, 2]/L3))
    C_calc_chi0(
        chi0_real, chi0_imag,
        ek, Ukdag, sublattice_phase,
        ws, len(ws), gamma,
        qs, len(qs),
        L3, L2, L1, Nband,
        n_threads)
    return chi0_real + chi0_imag*1j

C_calc_chi0_band = lib.calc_chi0_band_exact
C_calc_chi0_band.argtypes = C_calc_chi0.argtypes
C_calc_chi0_band.restype = None

def calc_chi0_band(qs, ws, gamma, ek, Ukdag, x_i, n_threads=1):
    L3, L2, L1, Nband, Norb = Ukdag.shape
    assert Norb == Nband

    chi0_real = np.zeros((Nband, Nband, len(qs), len(ws)), dtype=np.float64)
    chi0_imag = np.zeros((Nband, Nband, len(qs), len(ws)), dtype=np.float64)
    sublattice_phase = np.exp(-2j*np.pi * (qs[:, 0, None]*x_i[:, 0]/L1 +
                                           qs[:, 1, None]*x_i[:, 1]/L2 +
                                           qs[:, 2, None]*x_i[:, 2]/L3))
    C_calc_chi0_band(
        chi0_real, chi0_imag,
        ek, Ukdag, sublattice_phase,
        ws, len(ws), gamma,
        qs, len(qs),
        L3, L2, L1, Nband,
        n_threads)
    return chi0_real + chi0_imag*1j

C_calc_chi0_allorb = lib.calc_chi0_allorb_exact
C_calc_chi0_allorb.argtypes = C_calc_chi0.argtypes
C_calc_chi0_allorb.restype = None

def calc_chi0_allorb(qs, ws, gamma, ek, Ukdag, x_i, n_threads=1):
    L3, L2, L1, Nband, Norb = Ukdag.shape
    assert Norb == Nband

    chi0_real = np.zeros((Norb, Norb, len(qs), len(ws)), dtype=np.float64)
    chi0_imag = np.zeros((Norb, Norb, len(qs), len(ws)), dtype=np.float64)
    sublattice_phase = np.exp(-2j*np.pi * (qs[:, 0, None]*x_i[:, 0]/L1 +
                                           qs[:, 1, None]*x_i[:, 1]/L2 +
                                           qs[:, 2, None]*x_i[:, 2]/L3))
    C_calc_chi0_allorb(
        chi0_real, chi0_imag,
        ek, Ukdag, sublattice_phase,
        ws, len(ws), gamma,
        qs, len(qs),
        L3, L2, L1, Nband,
        n_threads)
    return chi0_real + chi0_imag*1j

C_calc_chi0_orbij = lib.calc_chi0_orbij_exact
C_calc_chi0_orbij.argtypes = [
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"), # chi0_real
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"), # chi0_imag
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"), # ek
    np.ctypeslib.ndpointer(np.complex128, flags="C_CONTIGUOUS"), # Ukdag
    np.ctypeslib.ndpointer(np.complex128, flags="C_CONTIGUOUS"), # sub_phase
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"), # ws
    ctypes.c_size_t, # nw
    ctypes.c_double, # gamma
    np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"), # qs
    ctypes.c_size_t, # nq
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, # L3, L2, L1
    ctypes.c_size_t, # Nband
    ctypes.c_size_t, # orbi
    ctypes.c_size_t, # orbj
    ctypes.c_size_t, # n_threads
]
C_calc_chi0_orbij.restype = None

def calc_chi0_orbij(qs, ws, gamma, ek, Ukdag, x_i, orbi, orbj, n_threads=1):
    L3, L2, L1, Nband, Norb = Ukdag.shape
    assert Norb == Nband

    chi0_real = np.zeros((len(qs), len(ws)), dtype=np.float64)
    chi0_imag = np.zeros((len(qs), len(ws)), dtype=np.float64)
    sublattice_phase = np.exp(-2j*np.pi * (qs[:, 0, None]*x_i[:, 0]/L1 +
                                           qs[:, 1, None]*x_i[:, 1]/L2 +
                                           qs[:, 2, None]*x_i[:, 2]/L3))
    C_calc_chi0_orbij(
        chi0_real, chi0_imag,
        ek, Ukdag, sublattice_phase,
        ws, len(ws), gamma,
        qs, len(qs),
        L3, L2, L1, Nband,
        orbi, orbj,
        n_threads)
    return chi0_real + chi0_imag*1j
