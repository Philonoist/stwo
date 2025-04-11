use cudarc::driver::{DevicePtr, LaunchAsync, LaunchConfig};
use stwo_prover::core::backend::{Col, Column, ColumnOps, CpuBackend};
use stwo_prover::core::circle::{CirclePoint, Coset};
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
};
use stwo_prover::core::poly::twiddles::TwiddleTree;
use stwo_prover::core::poly::{BitReversedOrder, NaturalOrder};

use crate::{CudaBackend, CudaColumn, CUDA_CTX};

// Constants matching the SIMD implementation
const MIN_FFT_LOG_SIZE: u32 = 5;
const CACHED_FFT_LOG_SIZE: u32 = 16;

impl PolyOps for CudaBackend {
    type Twiddles = Vec<u32>;

    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // For now, fallback to CPU implementation for this operation
        let eval = CpuBackend::new_canonical_ordered(coset, values.into_cpu_vec());
        CircleEvaluation::new(
            eval.domain,
            Col::<CudaBackend, BaseField>::from_iter(eval.values),
        )
    }

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddles: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        let log_size = eval.values.length.ilog2();
        if log_size < MIN_FFT_LOG_SIZE {
            let cpu_poly = eval.to_cpu().interpolate();
            return CirclePoly::new(cpu_poly.coeffs.into_iter().collect());
        }

        // We need to implement the inverse FFT (IFFT) logic
        // For now, fallback to CPU implementation
        let cpu_poly = eval.to_cpu().interpolate();
        CirclePoly::new(cpu_poly.coeffs.into_iter().collect())
    }

    fn eval_at_point(
        _poly: &CirclePoly<Self>,
        _point: CirclePoint<stwo_prover::core::fields::qm31::SecureField>,
    ) -> stwo_prover::core::fields::qm31::SecureField {
        // Fallback to CPU implementation for now
        unimplemented!("CUDA eval_at_point not yet implemented")
    }

    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self> {
        // Implementation of polynomial extension
        poly.evaluate(CanonicCoset::new(log_size).circle_domain())
            .interpolate()
    }

    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let log_size = domain.log_size();
        let fft_log_size = poly.log_size();
        assert!(
            log_size >= fft_log_size,
            "Can only evaluate on larger domains"
        );

        if fft_log_size < MIN_FFT_LOG_SIZE {
            let cpu_poly: CirclePoly<CpuBackend> = CirclePoly::new(poly.coeffs.to_cpu());
            let cpu_eval = cpu_poly.evaluate(domain);
            return CircleEvaluation::new(
                cpu_eval.domain,
                Col::<CudaBackend, BaseField>::from_iter(cpu_eval.values),
            );
        }

        // This is where we'd implement the forward FFT (RFFT)
        let mut res = Col::<CudaBackend, BaseField>::zeros(domain.size());

        // Get the CUDA FFT kernel - this would be a wrapper that manages multiple kernel calls
        let kernel = CUDA_CTX.get_func("rfft", "fft_small").unwrap();

        // Launch configuration
        let cfg = LaunchConfig::for_num_elems(domain.size());

        // Launch the kernel
        // NOTE: This is simplified - need actual implementation with proper twiddles handling
        unsafe {
            kernel
                .clone()
                .launch(
                    cfg,
                    (
                        &poly.coeffs.buffer,
                        &mut res.buffer,
                        &twiddles.twiddles,
                        log_size,
                    ),
                )
                .unwrap();
        }

        CircleEvaluation::new(domain, res)
    }

    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        // For simplicity, use the CPU implementation to compute twiddles
        // and then convert to the format needed by CUDA
        let cpu_twiddles = CpuBackend::precompute_twiddles(coset);

        // Convert CPU twiddles to CUDA format
        let twiddles = cpu_twiddles.twiddles.clone();
        let itwiddles = cpu_twiddles.itwiddles.clone();

        TwiddleTree {
            root_coset: coset,
            twiddles,
            itwiddles,
        }
    }
}

// Add tests for the CUDA FFT implementation
#[cfg(test)]
mod tests {
    use stwo_prover::core::backend::CpuBackend;

    use super::*;

    #[test]
    fn test_fft_small() {
        // Test small FFT with random data
        const LOG_SIZE: u32 = 8;
        let domain = CanonicCoset::new(LOG_SIZE).circle_domain();

        let values: Vec<BaseField> = (0..(1 << LOG_SIZE)).map(|i| BaseField::from(i)).collect();

        let poly = CirclePoly::<CudaBackend>::new(Col::<CudaBackend, BaseField>::from_iter(
            values.clone(),
        ));

        // Get twiddles
        let twiddles = CudaBackend::precompute_twiddles(domain.half_coset);

        // Evaluate using CUDA
        let cuda_eval = poly.evaluate(domain, &twiddles);

        // Compare with CPU implementation
        let cpu_poly = CirclePoly::<CpuBackend>::new(values);
        let cpu_twiddles = CpuBackend::precompute_twiddles(domain.half_coset);
        let cpu_eval = cpu_poly.evaluate(domain, &cpu_twiddles);

        assert_eq!(cuda_eval.values.to_cpu(), cpu_eval.values);
    }
}
