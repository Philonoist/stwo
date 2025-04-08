use cudarc::driver::{DevicePtr, LaunchAsync, LaunchConfig};
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::backend::{Col, Column, ColumnOps};
use stwo_prover::core::channel::Blake2sChannel;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::proof_of_work::GrindOps;
use stwo_prover::core::vcs::blake2_hash::Blake2sHash;
use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use stwo_prover::core::vcs::ops::MerkleOps;

use crate::{CudaBackend, CudaColumn, CUDA_CTX};

impl ColumnOps<Blake2sHash> for CudaBackend {
    type Column = CudaColumn<Blake2sHash>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOps<Blake2sMerkleHasher> for CudaBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, Blake2sHash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, Blake2sHash> {
        let cfg = LaunchConfig::for_num_elems(1 << log_size);

        // Allocate pointer buf.
        let col_ptrs = CUDA_CTX
            .htod_copy(
                columns
                    .iter()
                    .map(|column| *column.buffer.device_ptr())
                    .collect(),
            )
            .unwrap();
        let mut res = Col::<Self, Blake2sHash>::zeros(1 << log_size);

        if let Some(prev_layer) = prev_layer {
            let kernel = CUDA_CTX
                .get_func("blake2s", "commit_layer_with_parent")
                .unwrap();
            unsafe {
                kernel.clone().launch(
                    cfg,
                    (
                        &mut res.buffer,
                        &prev_layer.buffer,
                        &col_ptrs,
                        1 << log_size,
                        columns.len(),
                    ),
                )
            }
            .unwrap();
        } else {
            let kernel = CUDA_CTX
                .get_func("blake2s", "commit_layer_no_parent")
                .unwrap();
            unsafe {
                kernel.clone().launch(
                    cfg,
                    (&mut res.buffer, &col_ptrs, 1 << log_size, columns.len()),
                )
            }
            .unwrap();
        }

        res
    }
}

impl GrindOps<Blake2sChannel> for CudaBackend {
    fn grind(channel: &Blake2sChannel, pow_bits: u32) -> u64 {
        // Ensure we don't exceed what we can handle
        assert!(pow_bits <= 63, "pow_bits must be <= 63");

        // Get digest from channel
        let digest = channel.digest();

        // Convert digest bytes to u32 array
        let digest_words: Vec<u32> = digest
            .0
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        // Copy digest to device
        let d_digest = CUDA_CTX.htod_copy(digest_words).unwrap();

        // Constants for kernel launch
        const NONCES_PER_THREAD: u32 = 8;
        const BATCH_SIZE: u32 = 128 * 1024; // 128K threads

        // Initialize result to a value that indicates "not found"
        let mut d_result = CUDA_CTX
            .htod_copy(vec![BATCH_SIZE * NONCES_PER_THREAD])
            .unwrap();

        // Get the kernel
        let kernel = CUDA_CTX.get_func("blake2s", "grind_blake2s").unwrap();
        let config = LaunchConfig::for_num_elems(BATCH_SIZE);

        let mut base_nonce: u64 = 0;
        loop {
            // Reset result for this batch
            CUDA_CTX
                .htod_copy_into(vec![BATCH_SIZE * NONCES_PER_THREAD], &mut d_result)
                .unwrap();

            unsafe {
                kernel
                    .clone()
                    .launch(
                        config,
                        (&d_digest, &mut d_result, pow_bits, BATCH_SIZE, base_nonce),
                    )
                    .unwrap();
            }

            // Copy result back to host
            let h_result = CUDA_CTX.dtoh_sync_copy(&d_result).unwrap();

            if h_result[0] < BATCH_SIZE * NONCES_PER_THREAD {
                // We found a valid nonce in this batch
                return base_nonce + h_result[0] as u64;
            }

            // Try the next batch
            base_nonce += (BATCH_SIZE * NONCES_PER_THREAD) as u64;

            // Safety check to avoid infinite loop
            if base_nonce > u64::MAX - (BATCH_SIZE * NONCES_PER_THREAD) as u64 {
                panic!("Failed to find a valid nonce after exhausting nonce space");
            }
        }
    }
}

#[test]
fn test_blake2s() {
    use stwo_prover::core::backend::CpuBackend;

    // First layer.
    const LOG_SIZE: u32 = 9;
    let cols: Vec<Col<CudaBackend, BaseField>> = (0..35)
        .map(|i| {
            (0..(1 << (LOG_SIZE + 1)))
                .map(|j| BaseField::from(i * j))
                .collect()
        })
        .collect();
    let cpu_cols: Vec<_> = cols.iter().map(|c| c.to_cpu()).collect();

    let layer = CudaBackend::commit_on_layer(LOG_SIZE + 1, None, &cols.iter().collect::<Vec<_>>());
    let cpu_layer = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
        LOG_SIZE + 1,
        None,
        &cpu_cols.iter().collect::<Vec<_>>(),
    );
    assert_eq!(layer.to_cpu(), cpu_layer);

    // Next layer.
    let cols: Vec<Col<CudaBackend, BaseField>> = (0..16)
        .map(|i| {
            (0..(1 << LOG_SIZE))
                .map(|j| BaseField::from(i * j + 8))
                .collect()
        })
        .collect();
    let cpu_cols: Vec<_> = cols.iter().map(|c| c.to_cpu()).collect();
    let layer =
        CudaBackend::commit_on_layer(LOG_SIZE, Some(&layer), &cols.iter().collect::<Vec<_>>());
    let cpu_layer = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
        LOG_SIZE,
        Some(&cpu_layer),
        &cpu_cols.iter().collect::<Vec<_>>(),
    );

    assert_eq!(layer.to_cpu(), cpu_layer);
}

#[test]
fn test_blake2s_grind() {
    use std::time::Instant;

    use stwo_prover::core::backend::CpuBackend;
    use stwo_prover::core::channel::{Blake2sChannel, Channel};

    // Create a channel
    let channel = Blake2sChannel::default();

    // Test with various POW bit requirements
    for pow_bits in [8, 15, 16, 17, 26, 32, 35] {
        println!("Testing with {} pow bits", pow_bits);

        // Time the CUDA implementation
        let cuda_start = Instant::now();
        let cuda_nonce = CudaBackend::grind(&channel, pow_bits);
        let cuda_duration = cuda_start.elapsed();

        // Verify the nonce satisfies the POW requirement
        let mut channel_copy = channel.clone();
        channel_copy.mix_u64(cuda_nonce);
        let actual_zeros = channel_copy.trailing_zeros();

        assert!(
            actual_zeros >= pow_bits,
            "Found nonce {} has {} trailing zeros, which doesn't satisfy requirement of {} bits",
            cuda_nonce,
            actual_zeros,
            pow_bits
        );

        println!(
            "CUDA found valid nonce {} with {} trailing zeros in {:?}",
            cuda_nonce, actual_zeros, cuda_duration
        );

        // Only compare with CPU for small pow_bits
        if pow_bits <= 26 {
            let cpu_start = Instant::now();
            let cpu_nonce = SimdBackend::grind(&channel, pow_bits);
            let cpu_duration = cpu_start.elapsed();

            let mut cpu_channel = channel.clone();
            cpu_channel.mix_u64(cpu_nonce);

            println!(
                "CPU found nonce {} with {} trailing zeros in {:?}",
                cpu_nonce,
                cpu_channel.trailing_zeros(),
                cpu_duration
            );

            println!(
                "CUDA speedup: {:.2}x",
                cpu_duration.as_secs_f64() / cuda_duration.as_secs_f64()
            );
        }
        println!("-----------------------------------");
    }
}
