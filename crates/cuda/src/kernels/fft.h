#ifndef FFT_H
#define FFT_H

// M31 prime modulus
#define P 2147483647 // 2^31 - 1

// Perform modular reduction for M31 field
__device__ __forceinline__ static unsigned int m31_reduce(unsigned long long x)
{
    unsigned int lo = (unsigned int)(x & P);
    unsigned int hi = (unsigned int)(x >> 31);
    unsigned int res = lo + hi;
    if (res >= P)
    {
        res -= P;
    }
    return res;
}

// Add two M31 field elements
__device__ __forceinline__ static unsigned int m31_add(unsigned int a, unsigned int b)
{
    unsigned int res = a + b;
    if (res >= P)
    {
        res -= P;
    }
    return res;
}

// Subtract two M31 field elements
__device__ __forceinline__ static unsigned int m31_sub(unsigned int a, unsigned int b)
{
    return (a >= b) ? (a - b) : (a + P - b);
}

// Multiply two M31 field elements, with one doubled (used for twiddle factors)
__device__ __forceinline__ static unsigned int m31_mul_doubled(unsigned int a, unsigned int twiddle_dbl)
{
    // a * twiddle_dbl / 2 mod P
    return m31_reduce((unsigned long long)a * twiddle_dbl);
}

// Butterfly operation: (a, b) -> (a + b*w, a - b*w)
__device__ __forceinline__ static void butterfly(unsigned int &a, unsigned int &b, unsigned int twiddle_dbl)
{
    unsigned int prod = m31_mul_doubled(b, twiddle_dbl);
    unsigned int sum = m31_add(a, prod);
    unsigned int diff = m31_sub(a, prod);
    a = sum;
    b = diff;
}

// Warp-level butterfly operation using shuffle
__device__ __forceinline__ static void warp_butterfly(unsigned int &val, unsigned int twiddle_dbl, int mask, int offset)
{
    unsigned int paired_val = __shfl_xor_sync(0xffffffff, val, mask);
    unsigned int prod = m31_mul_doubled(paired_val, twiddle_dbl);

    if ((threadIdx.x & mask) == 0)
    {
        // First element in pair: a + b*w
        val = m31_add(val, prod);
    }
    else
    {
        // Second element in pair: a - b*w
        val = m31_sub(__shfl_xor_sync(0xffffffff, val, mask), prod);
    }
}

#endif // FFT_H
