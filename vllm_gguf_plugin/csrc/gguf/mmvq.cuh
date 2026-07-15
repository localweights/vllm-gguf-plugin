// copied and adapted from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmvq.cu
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void mul_mat_vec_q(const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows, const int nvecs) {
    const auto row = blockIdx.x*blockDim.y + threadIdx.y;
    const auto vec = blockIdx.y;

    if (row >= nrows || vec >= nvecs) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;


    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (auto i = threadIdx.x / (qi/vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = vec*(nrows_y/QK8_1) + i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs  = vdr * (threadIdx.x % (qi/vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
        tmp += VLLM_SHFL_XOR_SYNC(tmp, mask);
    }

    if (threadIdx.x == 0) {
        dst[vec*nrows + row] = tmp;
    }
}

// Multi-column variant: one thread accumulates ALL dst columns for its row so
// each weight (x) block is fetched from DRAM once and reused from L1 for the
// remaining columns. The per-column accumulation order is identical to
// mul_mat_vec_q, so results are bitwise-identical. Used for small batch
// (MTP verify runs the target model over 2 tokens/step; the 1-col grid-y
// layout re-reads the full weight matrix per column -> ~2x kernel time).
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, int ncols_dst, vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void mul_mat_vec_q_ncols(const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows, const int nvecs) {
    const auto row = blockIdx.x*blockDim.y + threadIdx.y;

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;

    float tmp[ncols_dst];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        tmp[j] = 0.0f;
    }

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (auto i = threadIdx.x / (qi/vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iqs  = vdr * (threadIdx.x % (qi/vdr)); // x block quant index when casting the quants to int

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            if (j < nvecs) {
                const int iby = j*(nrows_y/QK8_1) + i * (qk/QK8_1); // y block index that aligns with ibx
                tmp[j] += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
            }
        }
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
            tmp[j] += VLLM_SHFL_XOR_SYNC(tmp[j], mask);
        }
        if (threadIdx.x == 0 && j < nvecs) {
            dst[j*nrows + row] = tmp[j];
        }
    }
}

// Dedicated iq4_xs 2-column kernel using the fused 2-col vec dot: the weight
// block is decoded once per (thread, block) and dotted against both activation
// columns from registers. Bitwise-identical per-column math.
template <typename scalar_t>
static __global__ void mul_mat_vec_iq4_xs_q8_1_2col(const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows) {
    constexpr int qk = QK_K;
    constexpr int qi = QI4_XS;
    constexpr int vdr = 1;

    const auto row = blockIdx.x*blockDim.y + threadIdx.y;

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;

    float tmp0 = 0.0f;
    float tmp1 = 0.0f;

    const block_iq4_xs * x = (const block_iq4_xs *) vx;
    const block_q8_1   * y = (const block_q8_1 *) vy;

    for (auto i = threadIdx.x / (qi/vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index (col 0) that aligns with ibx

        const int iqs  = vdr * (threadIdx.x % (qi/vdr));

        vec_dot_iq4_xs_q8_1_2col(&x[ibx], &y[iby], &y[(nrows_y/QK8_1) + iby], iqs, tmp0, tmp1);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
        tmp0 += VLLM_SHFL_XOR_SYNC(tmp0, mask);
        tmp1 += VLLM_SHFL_XOR_SYNC(tmp1, mask);
    }

    if (threadIdx.x == 0) {
        dst[row] = tmp0;
        dst[nrows + row] = tmp1;
    }
}

// Dedicated q5_K / q6_K 2-column kernels using the fused 2-col vec_dot: the
// weight block is decoded once per (thread, block) and dotted against both
// activation columns from registers. Bitwise-identical per-column math.
// Mirrors mul_mat_vec_iq4_xs_q8_1_2col (758367d) for the K-quant path used by
// the nextn draft head (q8_0, already fused) chain's lm_head (q6_K) and other
// q5_K/q6_K serving tensors.
template<typename scalar_t>
static __global__ void mul_mat_vec_q5_K_q8_1_2col(const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows) {
    constexpr int qk = QK_K;
    constexpr int qi = QI5_K;
    constexpr int vdr = VDR_Q5_K_Q8_1_MMVQ;

    const auto row = blockIdx.x*blockDim.y + threadIdx.y;

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;

    float tmp0 = 0.0f;
    float tmp1 = 0.0f;

    const block_q5_K * x = (const block_q5_K *) vx;
    const block_q8_1  * y = (const block_q8_1 *) vy;

    for (auto i = threadIdx.x / (qi/vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index (col 0) that aligns with ibx

        const int iqs  = vdr * (threadIdx.x % (qi/vdr));

        vec_dot_q5_K_q8_1_2col(&x[ibx], &y[iby], &y[(nrows_y/QK8_1) + iby], iqs, tmp0, tmp1);
    }

#pragma unroll
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
        tmp0 += VLLM_SHFL_XOR_SYNC(tmp0, mask);
        tmp1 += VLLM_SHFL_XOR_SYNC(tmp1, mask);
    }

    if (threadIdx.x == 0) {
        dst[row] = tmp0;
        dst[nrows + row] = tmp1;
    }
}

template<typename scalar_t>
static __global__ void mul_mat_vec_q6_K_q8_1_2col(const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows) {
    constexpr int qk = QK_K;
    constexpr int qi = QI6_K;
    constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;

    const auto row = blockIdx.x*blockDim.y + threadIdx.y;

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;

    float tmp0 = 0.0f;
    float tmp1 = 0.0f;

    const block_q6_K * x = (const block_q6_K *) vx;
    const block_q8_1  * y = (const block_q8_1 *) vy;

    for (auto i = threadIdx.x / (qi/vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index (col 0) that aligns with ibx

        const int iqs  = vdr * (threadIdx.x % (qi/vdr));

        vec_dot_q6_K_q8_1_2col(&x[ibx], &y[iby], &y[(nrows_y/QK8_1) + iby], iqs, tmp0, tmp1);
    }

#pragma unroll
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
        tmp0 += VLLM_SHFL_XOR_SYNC(tmp0, mask);
        tmp1 += VLLM_SHFL_XOR_SYNC(tmp1, mask);
    }

    if (threadIdx.x == 0) {
        dst[row] = tmp0;
        dst[nrows + row] = tmp1;
    }
}

template<typename scalar_t>
static void mul_mat_vec_q4_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q4_1_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q5_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q5_1_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

// Shared small-batch dispatch: for nvecs 2..8 use the multi-column dst kernel
// (each weight block fetched once and reused for all dst columns) instead of
// the grid-y layout that re-reads the whole weight matrix per column
// (measured 1.8-2.2x batch-2 cost on q5_K/q6_K decode tensors). Per-column
// accumulation order matches mul_mat_vec_q exactly -> bitwise-identical.
// Same treatment 758367d applied to iq4_xs/iq4_nl.
template<typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static void mul_mat_vec_q_dispatch_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    // 2 rows (warps) per block for the ncols kernels: a 1-warp block caps
    // resident warps/SM at Ampere's 16-blocks/SM limit; 2 warps/block doubles
    // latency hiding. Launch geometry only -> bitwise-identical results.
    constexpr int MMV_Y2 = 2;
    const int block_num_y2 = (nrows + MMV_Y2 - 1) / MMV_Y2;
    const dim3 block_dims2(WARP_SIZE, MMV_Y2, 1);
    if (nvecs == 2) {
        const dim3 block_nums(block_num_y2, 1, 1);
        mul_mat_vec_q_ncols<scalar_t, qk, qi, block_q_t, vdr, 2, vec_dot_q_cuda>
            <<<block_nums, block_dims2, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    } else if (nvecs > 2 && nvecs <= 4) {
        const dim3 block_nums(block_num_y2, 1, 1);
        mul_mat_vec_q_ncols<scalar_t, qk, qi, block_q_t, vdr, 4, vec_dot_q_cuda>
            <<<block_nums, block_dims2, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    } else if (nvecs > 4 && nvecs <= 8) {
        const dim3 block_nums(block_num_y2, 1, 1);
        mul_mat_vec_q_ncols<scalar_t, qk, qi, block_q_t, vdr, 8, vec_dot_q_cuda>
            <<<block_nums, block_dims2, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    } else {
        const dim3 block_nums(block_num_y, nvecs, 1);
        mul_mat_vec_q<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    }
}

template<typename scalar_t>
static void mul_mat_vec_q8_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mul_mat_vec_q_dispatch_cuda<scalar_t, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q2_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q3_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q4_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

// rows_per_block occupancy tune (P3 sub-item 2), microbench-confirmed
// (microbench_p3.py, RTX 3090 Ti sm_86): lm_head is q6_K, 248320 rows x 5120
// cols -- far more rows than ssm_out/attn_gate (5120) or ffn (17408). At
// MMV_Y2=2 lm_head's batch-2/batch-1 ratio measured 1.317 (worse than the
// generic path this replaces); MMV_Y2=4 (4 warps/block, still under Ampere's
// 16-blocks/SM ceiling) measured 1.040. ssm_out/attn_gate/ffn shapes measured
// best at MMV_Y2=2 in both configs, so lm_head needs its own bucket, not a
// blanket MMV_Y2 bump. Threshold nrows > 65536 is the only class boundary
// that separates lm_head from every other shape in this model. Pure
// launch-geometry switch -> bitwise-identical (see bitwise_check2.py int16
// gate, all shapes/batches 1-8, 0 mismatches).
template<typename scalar_t>
static void mul_mat_vec_q5_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    if (nvecs == 2) {
        const int mmv_y2 = (nrows > 65536) ? 4 : 2;
        const dim3 block_dims2(WARP_SIZE, mmv_y2, 1);
        const dim3 block_nums((nrows + mmv_y2 - 1) / mmv_y2, 1, 1);
        mul_mat_vec_q5_K_q8_1_2col<scalar_t>
            <<<block_nums, block_dims2, 0, stream>>>(vx, vy, dst, ncols, nrows);
        return;
    }
    mul_mat_vec_q_dispatch_cuda<scalar_t, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q6_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    if (nvecs == 2) {
        const int mmv_y2 = (nrows > 65536) ? 4 : 2;
        const dim3 block_dims2(WARP_SIZE, mmv_y2, 1);
        const dim3 block_nums((nrows + mmv_y2 - 1) / mmv_y2, 1, 1);
        mul_mat_vec_q6_K_q8_1_2col<scalar_t>
            <<<block_nums, block_dims2, 0, stream>>>(vx, vy, dst, ncols, nrows);
        return;
    }
    mul_mat_vec_q_dispatch_cuda<scalar_t, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_iq2_xxs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq2_xs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq2_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq3_xxs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq1_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq1_m_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI1_M, block_iq1_m, 1, vec_dot_iq1_m_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq4_nl_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    if (nvecs == 2) {
        const dim3 block_nums(block_num_y, 1, 1);
        mul_mat_vec_q_ncols<scalar_t, QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, 2, vec_dot_iq4_nl_q8_1>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    } else if (nvecs > 2 && nvecs <= 4) {
        const dim3 block_nums(block_num_y, 1, 1);
        mul_mat_vec_q_ncols<scalar_t, QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, 4, vec_dot_iq4_nl_q8_1>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    } else if (nvecs > 4 && nvecs <= 8) {
        const dim3 block_nums(block_num_y, 1, 1);
        mul_mat_vec_q_ncols<scalar_t, QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, 8, vec_dot_iq4_nl_q8_1>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    } else {
        const dim3 block_nums(block_num_y, nvecs, 1);
        mul_mat_vec_q<scalar_t, QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    }
}

template<typename scalar_t>
static void mul_mat_vec_iq4_xs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    if (nvecs == 2) {
        // The fused 2-col kernel is a single-warp-per-block launch
        // (GGML_CUDA_MMV_Y == 1), which hard-caps resident warps/SM at the
        // Ampere 16-blocks/SM limit (16 warps/SM out of 48 max) regardless
        // of register headroom (measured 40 regs/thread, well under the
        // budget for 16 blocks/SM). For shapes with a large K-dim (many
        // sequential, dependent global-load iterations per warp, e.g.
        // ffn_down 5120x17408 -> 68 blocks_per_row) that low warp count
        // starves latency hiding and the batch-2 cost balloons (measured
        // 1.35x vs 1.06x for ffn_gate/up's 20 blocks_per_row). Packing
        // MMV_Y2=2 rows/block doubles resident warps/SM (32/48) for a pure
        // launch-geometry change -- no change to vec_dot math, so results
        // stay bitwise-identical.
        constexpr int MMV_Y2 = 2;
        const int block_num_y2 = (nrows + MMV_Y2 - 1) / MMV_Y2;
        const dim3 block_dims2(WARP_SIZE, MMV_Y2, 1);
        const dim3 block_nums(block_num_y2, 1, 1);
        mul_mat_vec_iq4_xs_q8_1_2col<scalar_t>
            <<<block_nums, block_dims2, 0, stream>>>(vx, vy, dst, ncols, nrows);
    } else if (nvecs > 2 && nvecs <= 4) {
        const dim3 block_nums(block_num_y, 1, 1);
        mul_mat_vec_q_ncols<scalar_t, QK_K, QI4_XS, block_iq4_xs, 1, 4, vec_dot_iq4_xs_q8_1>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    } else if (nvecs > 4 && nvecs <= 8) {
        const dim3 block_nums(block_num_y, 1, 1);
        mul_mat_vec_q_ncols<scalar_t, QK_K, QI4_XS, block_iq4_xs, 1, 8, vec_dot_iq4_xs_q8_1>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    } else {
        const dim3 block_nums(block_num_y, nvecs, 1);
        mul_mat_vec_q<scalar_t, QK_K, QI4_XS, block_iq4_xs, 1, vec_dot_iq4_xs_q8_1>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    }
}

template<typename scalar_t>
static void mul_mat_vec_iq3_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}
