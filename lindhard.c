#include <complex.h>
#include <stdint.h>
#include <omp.h>

#define MEM_ALIGN 64
#define PAD_UP(size, align) ((((size) + (align) - 1)/(align)) * (align))
#define my_alloc(size) aligned_alloc(MEM_ALIGN, PAD_UP((size), MEM_ALIGN))

#if defined(_WIN32) || defined(_WIN64)
	#define EXPORT __declspec(dllexport)
	#define aligned_alloc(alignment, size) mkl_malloc((size), (alignment))
	#define free(ptr) mkl_free((ptr))
	#include <mkl.h>
#elif __APPLE__
	#define EXPORT
	#include <Accelerate/Accelerate.h>
	#define MKL_Complex16 double complex
	#define zheev zheev_
#else
	#define EXPORT
	#include <stdlib.h>
	#include <mkl.h>
#endif


EXPORT void par_eigh(
		double complex *restrict a,
		double *restrict eigvals,
		const size_t n_mat, const int n, const int lower,
		const int n_threads)
{
	const int lrwork = PAD_UP(3*n - 2, MEM_ALIGN/sizeof(double));
	double complex lwork_ = 0;
	double *restrict rwork = my_alloc(n_threads*lrwork*sizeof(double));

	zheev("V", lower ? "L" : "U", &n, (MKL_Complex16 *)a, &n, eigvals,
		(MKL_Complex16 *)&lwork_, &(int){-1}, rwork, &(int){0});
	const int lwork = PAD_UP((int)(creal(lwork_)), MEM_ALIGN/sizeof(MKL_Complex16));
	MKL_Complex16 *restrict work = my_alloc(n_threads*lwork*sizeof(MKL_Complex16));

	#pragma omp parallel num_threads(n_threads)
	{
		const int thread = omp_get_thread_num();
		#pragma omp for schedule(static)
		for (size_t i = 0; i < n_mat; i++) {
			zheev("V", lower ? "L" : "U", &n, (MKL_Complex16 *)&a[i*n*n], &n, &eigvals[i*n],
				&work[thread*lwork], &lwork, &rwork[thread*lrwork], &(int){0});
			// uncomment below to conjugate transpose
			// mkl_zimatcopy('C', 'C', n, n, (MKL_Complex16){1.0, 0.0}, (MKL_Complex16 *)&a[i*n*n], n, n);
		}
	}

	free(rwork);
	free(work);
}

EXPORT void calc_chi0_exact(
		double *restrict chi0_real, double *restrict chi0_imag,
		const double *restrict ek,
		const double complex *restrict Ukdag,
		const double complex *restrict sub_phase,
		const double *restrict ws, const size_t nw, const double gamma,
		const int32_t *restrict qs, const size_t nq,
		const size_t L3, const size_t L2, const size_t L1, const size_t Nband,
		const int n_threads)
{
	const double gamma_sq = gamma*gamma;

	#pragma omp parallel for num_threads(n_threads)
	for (size_t iq = 0; iq < nq; iq++) {
		for (size_t ik3 = 0; ik3 < L3; ik3++)
		for (size_t ik2 = 0; ik2 < L2; ik2++)
		for (size_t ik1 = 0; ik1 < L1; ik1++) {
			const size_t ik = ik1 + ik2*L1 + ik3*L2*L1;
			const size_t ikq3 = (ik3 + qs[2 + iq*3])%L3;
			const size_t ikq2 = (ik2 + qs[1 + iq*3])%L2;
			const size_t ikq1 = (ik1 + qs[0 + iq*3])%L1;
			const size_t ikq = ikq1 + ikq2*L1 + ikq3*L2*L1;
			for (size_t b = 0; b < Nband; b++)
			for (size_t a = 0; a < Nband; a++) {
				const double ebk = ek[b + ik*Nband];
				const double eakq = ek[a + ikq*Nband];
				const double ee = ebk - eakq;
				if (eakq <= 0 || ebk >= 0) continue;
				// ignore cases where eakq == 0 or ebk == 0

				double complex UU = 0.0;
				const size_t offset_k = b*Nband + ik*Nband*Nband;
				const size_t offset_kq = a*Nband + ikq*Nband*Nband;
				for (size_t i = 0; i < Nband; i++) {
					UU += sub_phase[i + iq*Nband]*Ukdag[i + offset_k]*conj(Ukdag[i + offset_kq]);
				}
				const double UUUU = creal(UU)*creal(UU) + cimag(UU)*cimag(UU);
				if (UUUU < 1e-12)
					continue;

				const double e2 = ee*ee;
				for (size_t iw = 0; iw < nw; iw++) {
					const double w = ws[iw];
					const double w2 = w*w;
					const double w2e2y2 = w2 - e2 - gamma_sq;
					const double frac = UUUU*ee/(w2e2y2*w2e2y2 + 4*w2*gamma_sq);
					chi0_real[iw + iq*nw] += -2*w2e2y2*frac;
					chi0_imag[iw + iq*nw] += 4*w*gamma*frac;
				}
			}
		}
	}

	const double invN = 1.0/(L3*L2*L1);
	for (size_t iq = 0; iq < nq; iq++)
	for (size_t iw = 0; iw < nw; iw++) {
		chi0_real[iw + iq*nw] *= invN;
		chi0_imag[iw + iq*nw] *= invN;
	}
}


EXPORT void calc_chi0_binned(
		double *restrict chi0_real, double *restrict chi0_imag,
		const double *restrict ek,
		const double complex *restrict Ukdag,
		const double complex *restrict sub_phase,
		const double *restrict ws, const size_t nw, const double gamma,
		const int32_t *restrict qs, const size_t nq,
		const size_t L3, const size_t L2, const size_t L1, const size_t Nband,
		const int n_threads)
{
	const double invN = 1.0/(L3*L2*L1);
	const double gamma_sq = gamma*gamma;

	const double BINS_PER_GAMMA = 64.0; // this could be a function parameter.
	const double dw = gamma/BINS_PER_GAMMA; // gamma/64 is a reasonably fast and accurate option
	double ek_min = ek[0];
	double ek_max = ek[0];
	for (size_t i = 0; i < L3*L2*L1*Nband; i++) {
		if (ek[i] > ek_max)
			ek_max = ek[i];
		else if (ek[i] < ek_min)
			ek_min = ek[i];
	}
	const size_t nbin = PAD_UP((size_t)((ek_max - ek_min)/dw) + 4, MEM_ALIGN/sizeof(double));
	double *buffer = my_alloc(n_threads*nbin * sizeof(double));

	#pragma omp parallel num_threads(n_threads)
	{
		double *restrict weights = buffer + omp_get_thread_num()*nbin;
		#pragma omp for schedule(dynamic, 1)
		for (size_t iq = 0; iq < nq; iq++) {
			for (size_t ib = 0; ib < nbin; ib++) weights[ib] = 0.0;

			for (size_t ik3 = 0; ik3 < L3; ik3++)
			for (size_t ik2 = 0; ik2 < L2; ik2++)
			for (size_t ik1 = 0; ik1 < L1; ik1++) {
				const size_t ik = ik1 + ik2*L1 + ik3*L2*L1;
				const size_t ikq3 = (ik3 + qs[2 + iq*3])%L3;
				const size_t ikq2 = (ik2 + qs[1 + iq*3])%L2;
				const size_t ikq1 = (ik1 + qs[0 + iq*3])%L1;
				const size_t ikq = ikq1 + ikq2*L1 + ikq3*L2*L1;
				for (size_t b = 0; b < Nband; b++)
				for (size_t a = 0; a < Nband; a++) {
					const double ebk = ek[b + ik*Nband];
					const double eakq = ek[a + ikq*Nband];
					if (eakq <= 0 || ebk >= 0) continue;
					// ignore cases where eakq == 0 or ebk == 0

					double complex UU = 0.0;
					const size_t offset_k = b*Nband + ik*Nband*Nband;
					const size_t offset_kq = a*Nband + ikq*Nband*Nband;
					for (size_t i = 0; i < Nband; i++) {
						UU += sub_phase[i + iq*Nband]*Ukdag[i + offset_k]*conj(Ukdag[i + offset_kq]);
					}
					const double UUUU = creal(UU)*creal(UU) + cimag(UU)*cimag(UU);
					if (UUUU < 1e-12)
						continue;

					// effectively moves pole to nearest bin
					// size_t ib = (size_t)((eakq - ebk)/dw + 0.5);
					// if (ib >= nbin) continue; // guaranteed to not happen given definition of nbin
					// weights[ib] += UUUU;

					// effectively splits pole into two, separated by dw, maintaining center of "mass"
					const double bin_center = (eakq - ebk)/dw;
					const size_t ib = (size_t)bin_center;
					const double frac_hi = bin_center - ib;
					// if (ib >= nbin - 1) continue; // guaranteed to not happen given definition of nbin
					weights[ib] += UUUU*(1 - frac_hi);
					weights[ib + 1] += UUUU*frac_hi;
				}
			}

			for (size_t ib = 0; ib < nbin; ib++) {
				const double weight = weights[ib];
				if (weight < 1e-12)
					continue;
				const double ee = -dw*ib;
				const double e2 = ee*ee;
				for (size_t iw = 0; iw < nw; iw++) {
					const double w = ws[iw];
					const double w2 = w*w;
					const double w2e2y2 = w2 - e2 - gamma_sq;
					const double frac = weight*ee/(w2e2y2*w2e2y2 + 4*w2*gamma_sq);
					chi0_real[iw + iq*nw] += -2*w2e2y2*frac;
					chi0_imag[iw + iq*nw] += 4*w*gamma*frac;
				}
			}

			for (size_t iw = 0; iw < nw; iw++) {
				chi0_real[iw + iq*nw] *= invN;
				chi0_imag[iw + iq*nw] *= invN;
			}
		}
	}
	free(buffer);
}
