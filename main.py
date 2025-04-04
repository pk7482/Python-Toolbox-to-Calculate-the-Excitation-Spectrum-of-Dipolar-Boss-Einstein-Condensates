import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc
from scipy.fft import fft, ifft, fftfreq
from scipy.linalg import eig
import scipy

# === USER INPUT ===
N_b = int(input('Enter The Value Of N_b: '))

# === INITIALIZATION ===
x = np.linspace(-75.00, 74.95, 3000)
dx = x[1] - x[0]
Nx = 3000
lx = Nx * dx
N_atom = 20000

# Physical constants
w_x = 20
w_perp = 150
lambda_x = w_x / w_perp
amu = 1.66e-27
hbar = 1.054560653e-34
au = 0.529177208e-10
m = 164 * amu
omega = 2 * np.pi * w_perp
aosc = np.sqrt(hbar / (m * omega))
a_s = 55.5 * au / aosc       #56
a_dd = 130.8 * au / aosc

# Interaction strengths
g = 4 * np.pi * N_atom * a_s
g_dd = 4 * np.pi * N_atom * a_dd
pir = (lambda_x / np.pi)  0.25
eps_dd = a_dd / a_s
mu = 9.64788157    # 9.67536

# LOAD PHI DATA
try:
    data = np.loadtxt('tmp_solution_file.dat')
    phi = np.sqrt(data[:, 1]) * np.exp(1j * data[:, 2])
except Exception as e:
    print("Error loading 'tmp_solution_file.dat':", e)
    exit()

# FFT wavenumbers
k = fftfreq(Nx, d=dx) * 2 * np.pi

# INTEGRATION WRAPPERS
def simp(F, dx):
    return scipy.integrate.simpson(F, dx=dx)

def simpc(F, dx):
    return scipy.integrate.simpson(F.real, dx=dx) + 1j * scipy.integrate.simpson(F.imag, dx=dx)

# HARMONIC OSCILLATOR BASIS
def ho_basis(nx, nb, pir, x, lambda_x):
    psix = np.zeros((nb, nx), dtype=np.complex128)
    exp_term = np.exp(-lambda_x * x  2 / 2.0)

    psix[0, :] = pir * exp_term
    if nb > 1:
        psix[1, :] = pir * np.sqrt(2 * lambda_x) * x * exp_term

    for i in range(2, nb):
        psix[i, :] = (np.sqrt(2.0 / i) * np.sqrt(lambda_x) * x * psix[i - 1, :]
                      - np.sqrt((i - 1) / i) * psix[i - 2, :])

    np.savetxt("basis.txt", np.column_stack([x, psix[nb - 1, :].real]))
    return psix

# CALCULATE U_k
def Cal_U_k(g, g_dd, k):
    u = k  2 / 2
    result = np.where(
        u == 0,
        g / (2 * np.pi) + (g_dd / (2 * np.pi)) * 0.5,
        np.where(
            u > 700,
            g / (2 * np.pi) - (g_dd / (2 * np.pi)),
            g / (2 * np.pi) + (g_dd / (2 * np.pi)) * (3 * ((-u) * np.exp(u) * sc.exp1(u) + 1) / 2 - 1)
        )
    )

    plt.plot(result)
    plt.show()
    return result

# QUANTUM FLUCTUATION CORRECTION
gamma = (2 * N_atom  1.5) * ((128 / 3) * np.sqrt(np.pi * a_s  5) * (1 + (3 / 2) * eps_dd  2)) / (5 * np.pi  1.5)

# CALCULATE X MATRICES
def Cal_X(phi1, phi2, psix, U_k, dx):
    nb, nx = psix.shape
    X = np.zeros((nb, nb), dtype=np.complex128)

    for j in range(nb):
        temp = phi2 * psix[j, :]
        tempF = fft(temp)
        tempF *= U_k
        temp = ifft(tempF)

        for i in range(nb):
            integrand = np.conj(psix[i, :]) * phi1 * temp
            X[i, j] = simp(integrand, dx)

    return X

# CALCULATE L COMPONENT
def Cal_L(phi, U_k, g, g_dd, k, N_atom, a_s, a_dd):
    total_energy = (N_b + 1 / 2) * lambda_x
    density_k = fft(np.abs(phi)  2)
    Phi_d_x = ifft(U_k * density_k)
    nonlinear_term = gamma * np.abs(phi)  3
    return total_energy, Phi_d_x, nonlinear_term

#BdG FUNCTION
def BdG(mu, phi, X1, X2, X3, X4, Nx, dx, N_atom, N_b, lambda_x, psix, DPR, gamma):
    MATRIX = np.zeros((2 * N_b, 2 * N_b), dtype=np.complex128)

    for j in range(N_b):
        MATRIX[j, j] = (j + 0.5) * lambda_x - mu
        MATRIX[j + N_b, j + N_b] = (j + 0.5) * lambda_x - mu

    M = np.abs(phi)  3
    M12 = phi  2 * np.abs(phi)
    M21 = np.conj(phi)  2 * np.abs(phi)

    for i in range(N_b):
        for j in range(N_b):


            term1 = scipy.integrate.simpson(M * psix[i, :] * psix[j, :], x=x)
            term2 = scipy.integrate.simpson(M12 * psix[i, :] * psix[j, :], x=x)
            term3 = scipy.integrate.simpson(M21 * psix[i, :] * psix[j, :], x=x)
            tmp_dpr = scipy.integrate.simpson(psix[i, :] * DPR * psix[j, :], x=x)
MATRIX[i, j] += tmp_dpr + X1[i, j] + (5.0 / 2.0) * gamma * term1
            MATRIX[i, j + N_b] = (3.0 / 2.0) * gamma * term2 + X2[i, j]
            MATRIX[i + N_b, j] = -( (3.0 / 2.0) * gamma * term3 + X3[i, j] )
            MATRIX[i + N_b, j + N_b] = -( MATRIX[i + N_b, j + N_b] + tmp_dpr + (5.0 / 2.0) * gamma * term1 + X4[i, j] )

    W, VR = np.linalg.eig(MATRIX)

    idx = np.argsort(np.abs(W))
    W_sorted = W[idx]/lambda_x

    print("\n=== BdG Eigenvalues ===")
    for i in range(2 * N_b):
        print(f"Mode {i+1:4d}: {W_sorted[i].real:12.6f}  {W_sorted[i].imag:12.6f}")

    return W_sorted, VR

# def BdG(mu, phi, X1, X2, X3, X4, nx, dx, natoms, nb, lambda_x, psix, dpr, gmqf):
#     MATRIX = np.zeros((2 * nb, 2 * nb), dtype=np.complex128)

#     for j in range(nb):
#         MATRIX[j, j] = (j + 0.5) * lambda_x - mu
#         MATRIX[j + nb, j + nb] = (j + 0.5) * lambda_x - mu

#         M = np.abs(phi)  3
#         M12 = phi  2 * np.abs(phi)
#         M21 = np.conj(phi) ** 2 * np.abs(phi)

#         for i in range(nb):
#             for j in range(nb):


#                 term1 = scipy.integrate.simpson(M * psix[i, :] * psix[j, :], x=x)
#                 term2 = scipy.integrate.simpson(M12 * psix[i, :] * psix[j, :], x=x)
#                 term3 = scipy.integrate.simpson(M21 * psix[i, :] * psix[j, :], x=x)
#                 tmp_dpr = scipy.integrate.simpson(psix[i, :] * dpr * psix[j, :], x=x)

#                 MATRIX[i, j] += tmp_dpr + X1[i, j] + (5.0 / 2.0) * gmqf * term1
#                 MATRIX[i, j + nb] = (3.0 / 2.0) * gmqf * term2 + X2[i, j]
#                 MATRIX[i + nb, j] = -( (3.0 / 2.0) * gmqf* term3 + X3[i, j] )
#                 MATRIX[i + nb, j + nb] = -( MATRIX[i + nb, j + nb] + tmp_dpr + (5.0 / 2.0) * gmqf * term1 + X4[i, j] )

#          W, VR = np.linalg.eig(MATRIX)

#          dx = np.argsort(np.abs(W))
#          W_sorted = W[idx]/lambda_x

#          print("\n=== BdG Eigenvalues ===")
#          for i in range(2 * nb):
#               print(f"Mode {i+1:4d}: {W_sorted[i].real:12.6f} {W_sorted[i].imag:12.6f}")

#      return W_sorted, VR 

# COMPUTATIONS
psix = ho_basis(Nx, N_b, pir, x, lambda_x)
U_k = Cal_U_k(g, g_dd, k)

phi_star_x = np.conj(phi)

X1 = Cal_X(phi, phi_star_x, psix, U_k, dx)
X2 = Cal_X(phi, phi, psix, U_k, dx)
X3 = Cal_X(phi_star_x, phi_star_x, psix, U_k, dx)
X4 = Cal_X(phi_star_x, phi, psix, U_k, dx)

total_En, DPR, GMQF = Cal_L(phi, U_k, g, g_dd, k, N_atom, a_s, a_dd)

#RUN BdG ANALYSIS
W_sorted, VR_sorted = BdG(mu, phi, X1, X2, X3, X4, Nx, dx, N_atom, N_b, lambda_x, psix, DPR, gamma)
