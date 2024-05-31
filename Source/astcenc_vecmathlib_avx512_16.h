// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2019-2024 Arm Limited
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
// ----------------------------------------------------------------------------

/**
 * @brief 16x32-bit vectors, implemented using AVX512.
 *
 * This module implements 16-wide 32-bit float, int, and mask vectors for x86
 * AVX512.
 *
 * There is a baseline level of functionality provided by all vector widths and
 * implementations. This is implemented using identical function signatures,
 * modulo data type, so we can use them as substitutable implementations in VLA
 * code.
 */

#ifndef ASTC_VECMATHLIB_AVX512_16_H_INCLUDED
#define ASTC_VECMATHLIB_AVX512_16_H_INCLUDED

#ifndef ASTCENC_SIMD_INLINE
	#error "Include astcenc_vecmathlib.h, do not include directly"
#endif

#include <cstdio>

// ============================================================================
// vfloat16 data type
// ============================================================================

/**
 * @brief Data type for 16-wide floats.
 */
struct vfloat16
{
	/**
	 * @brief Construct from zero-initialized value.
	 */
	ASTCENC_SIMD_INLINE vfloat16() = default;

	/**
	 * @brief Construct from 16 values loaded from an unaligned address.
	 *
	 * Consider using loada() which is better with vectors if data is aligned
	 * to vector length.
	 */
	ASTCENC_SIMD_INLINE explicit vfloat16(const float *p)
	{
		m = _mm512_loadu_ps(p);
	}

	/**
	 * @brief Construct from 1 scalar value replicated across all lanes.
	 *
	 * Consider using zero() for constexpr zeros.
	 */
	ASTCENC_SIMD_INLINE explicit vfloat16(float a)
	{
		m = _mm512_set1_ps(a);
	}

	/**
	 * @brief Construct from 16 scalar values.
	 *
	 * The value of @c a is stored to lane 0 (LSB) in the SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vfloat16(
		float a, float b, float c, float d,
		float e, float f, float g, float h,
		float i, float j, float k, float l,
		float n, float o, float p, float q)
	{
		m = _mm512_set_ps(q, p, o, n, l, k, j, i, h, g, f, e, d, c, b, a);
	}

	/**
	 * @brief Construct from an existing SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vfloat16(__m512 a)
	{
		m = a;
	}

	/**
	 * @brief Get the scalar value of a single lane.
	 */
	template <int l> ASTCENC_SIMD_INLINE float lane() const
	{
	#if !defined(__clang__) && defined(_MSC_VER)
		return m.m512_f32[l];
	#else
		union { __m512 m; float f[16]; } cvt;
		cvt.m = m;
		return cvt.f[l];
	#endif
	}

	/**
	 * @brief Factory that returns a vector of zeros.
	 */
	static ASTCENC_SIMD_INLINE vfloat16 zero()
	{
		return vfloat16(_mm512_setzero_ps());
	}

	/**
	 * @brief Factory that returns a replicated scalar loaded from memory.
	 */
	static ASTCENC_SIMD_INLINE vfloat16 load1(const float* p)
	{
		return vfloat16(_mm512_set1_ps(*p));
	}

	/**
	 * @brief Factory that returns a vector loaded from 64B aligned memory.
	 */
	static ASTCENC_SIMD_INLINE vfloat16 loada(const float* p)
	{
		return vfloat16(_mm512_load_ps(p));
	}

	/**
	 * @brief Factory that returns a vector containing the lane IDs.
	 */
	static ASTCENC_SIMD_INLINE vfloat16 lane_id()
	{
		return vfloat16(_mm512_set_ps(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
	}

	/**
	 * @brief The vector ...
	 */
	__m512 m;
};

// ============================================================================
// vint16 data type
// ============================================================================

/**
 * @brief Data type for 8-wide ints.
 */
struct vint16
{
	/**
	 * @brief Construct from zero-initialized value.
	 */
	ASTCENC_SIMD_INLINE vint16() = default;

	/**
	 * @brief Construct from 16 values loaded from an unaligned address.
	 *
	 * Consider using loada() which is better with vectors if data is aligned
	 * to vector length.
	 */
	ASTCENC_SIMD_INLINE explicit vint16(const int *p)
	{
		m = _mm512_loadu_epi32(p);
	}

	/**
	 * @brief Construct from 16 uint8_t loaded from an unaligned address.
	 */
	ASTCENC_SIMD_INLINE explicit vint16(const uint8_t *p)
	{
		m = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
	}

	/**
	 * @brief Construct from 1 scalar value replicated across all lanes.
	 *
	 * Consider using vfloat4::zero() for constexpr zeros.
	 */
	ASTCENC_SIMD_INLINE explicit vint16(int a)
	{
		m = _mm512_set1_epi32(a);
	}

	/**
	 * @brief Construct from 16 scalar values.
	 *
	 * The value of @c a is stored to lane 0 (LSB) in the SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vint16(
		int a, int b, int c, int d,
		int e, int f, int g, int h,
		int i, int j, int k, int l,
		int n, int o, int p, int q)
	{
		m = _mm512_set_epi32(q, p, o, n, l, k, j, i, h, g, f, e, d, c, b, a);
	}

	/**
	 * @brief Construct from an existing SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vint16(__m512i a)
	{
		m = a;
	}

	/**
	 * @brief Get the scalar from a single lane.
	 */
	template <int l> ASTCENC_SIMD_INLINE int lane() const
	{
	#if !defined(__clang__) && defined(_MSC_VER)
		return m.m512i_i32[l];
	#else
		union { __m512i m; int f[16]; } cvt;
		cvt.m = m;
		return cvt.f[l];
	#endif
	}

	/**
	 * @brief Factory that returns a vector of zeros.
	 */
	static ASTCENC_SIMD_INLINE vint16 zero()
	{
		return vint16(_mm512_setzero_si512());
	}

	/**
	 * @brief Factory that returns a replicated scalar loaded from memory.
	 */
	static ASTCENC_SIMD_INLINE vint16 load1(const int* p)
	{
		__m128i a = _mm_set1_epi32(*p);
		return vint16(_mm512_broadcastd_epi32(a));
	}

	/**
	 * @brief Factory that returns a vector loaded from unaligned memory.
	 */
	static ASTCENC_SIMD_INLINE vint16 load(const uint8_t* p)
	{
		return vint16(_mm512_loadu_si512(p));
	}

	/**
	 * @brief Factory that returns a vector loaded from 64B aligned memory.
	 */
	static ASTCENC_SIMD_INLINE vint16 loada(const int* p)
	{
		return vint16(_mm512_load_si512(p));
	}

	/**
	 * @brief Factory that returns a vector containing the lane IDs.
	 */
	static ASTCENC_SIMD_INLINE vint16 lane_id()
	{
		return vint16(_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
	}

	/**
	 * @brief The vector ...
	 */
	__m512i m;
};

// ============================================================================
// vmask16 data type
// ============================================================================

/**
 * @brief Data type for 16-wide control plane masks.
 */
struct vmask16
{
	/**
	 * @brief Construct from an existing SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vmask16(__m512 a)
	{
		m = _mm512_cmplt_ps_mask(a, _mm512_setzero_ps());
	}

	/**
	 * @brief Construct from an existing SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vmask16(__m512i a)
	{
		m = _mm512_cmplt_epi32_mask(a, _mm512_setzero_si512());
	}

	/**
	 * @brief Construct from an existing SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vmask16(__mmask16 a)
	{
		m = a;
	}

	/**
	 * @brief Construct from 1 scalar value.
	 */
	ASTCENC_SIMD_INLINE explicit vmask16(bool a)
	{
		m = _mm512_int2mask(a ? -1 : 0);
	}

	/**
	 * @brief The vector ...
	 */
	__mmask16 m;
};

// ============================================================================
// vmask16 operators and functions
// ============================================================================

/**
 * @brief Overload: mask union (or).
 */
ASTCENC_SIMD_INLINE vmask16 operator|(vmask16 a, vmask16 b)
{
	return vmask16(_mm512_kor(a.m, b.m));
}

/**
 * @brief Overload: mask intersect (and).
 */
ASTCENC_SIMD_INLINE vmask16 operator&(vmask16 a, vmask16 b)
{
	return vmask16(_mm512_kand(a.m, b.m));
}

/**
 * @brief Overload: mask difference (xor).
 */
ASTCENC_SIMD_INLINE vmask16 operator^(vmask16 a, vmask16 b)
{
	return vmask16(_mm512_kxor(a.m, b.m));
}

/**
 * @brief Overload: mask invert (not).
 */
ASTCENC_SIMD_INLINE vmask16 operator~(vmask16 a)
{
	return vmask16(_mm512_knot(a.m));
}

/**
 * @brief Return a 16-bit mask code indicating mask status.
 *
 * bit0 = lane 0
 */
ASTCENC_SIMD_INLINE unsigned int mask(vmask16 a)
{
	return _mm512_mask2int(a.m);
}

/**
 * @brief True if any lanes are enabled, false otherwise.
 */
ASTCENC_SIMD_INLINE bool any(vmask16 a)
{
	return mask(a) != 0;
}

/**
 * @brief True if all lanes are enabled, false otherwise.
 */
ASTCENC_SIMD_INLINE bool all(vmask16 a)
{
	return mask(a) == 0xFFFF;
}

// ============================================================================
// vint16 operators and functions
// ============================================================================
/**
 * @brief Overload: vector by vector addition.
 */
ASTCENC_SIMD_INLINE vint16 operator+(vint16 a, vint16 b)
{
	return vint16(_mm512_add_epi32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector incremental addition.
 */
ASTCENC_SIMD_INLINE vint16& operator+=(vint16& a, const vint16& b)
{
	a = a + b;
	return a;
}

/**
 * @brief Overload: vector by vector subtraction.
 */
ASTCENC_SIMD_INLINE vint16 operator-(vint16 a, vint16 b)
{
	return vint16(_mm512_sub_epi32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector multiplication.
 */
ASTCENC_SIMD_INLINE vint16 operator*(vint16 a, vint16 b)
{
	return vint16(_mm512_mullo_epi32(a.m, b.m));
}

/**
 * @brief Overload: vector bit invert.
 */
ASTCENC_SIMD_INLINE vint16 operator~(vint16 a)
{
	return vint16(_mm512_xor_si512(a.m, _mm512_set1_epi32(-1)));
}

/**
 * @brief Overload: vector by vector bitwise or.
 */
ASTCENC_SIMD_INLINE vint16 operator|(vint16 a, vint16 b)
{
	return vint16(_mm512_or_si512(a.m, b.m));
}

/**
 * @brief Overload: vector by vector bitwise and.
 */
ASTCENC_SIMD_INLINE vint16 operator&(vint16 a, vint16 b)
{
	return vint16(_mm512_and_si512(a.m, b.m));
}

/**
 * @brief Overload: vector by vector bitwise xor.
 */
ASTCENC_SIMD_INLINE vint16 operator^(vint16 a, vint16 b)
{
	return vint16(_mm512_xor_si512(a.m, b.m));
}

/**
 * @brief Overload: vector by vector equality.
 */
ASTCENC_SIMD_INLINE vmask16 operator==(vint16 a, vint16 b)
{
	return vmask16(_mm512_cmpeq_epi32_mask(a.m, b.m));
}

/**
 * @brief Overload: vector by vector inequality.
 */
ASTCENC_SIMD_INLINE vmask16 operator!=(vint16 a, vint16 b)
{
	return vmask16(_mm512_cmpneq_epi32_mask(a.m, b.m));
}

/**
 * @brief Overload: vector by vector less than.
 */
ASTCENC_SIMD_INLINE vmask16 operator<(vint16 a, vint16 b)
{
	return vmask16(_mm512_cmplt_epi32_mask(a.m, b.m));
}

/**
 * @brief Overload: vector by vector greater than.
 */
ASTCENC_SIMD_INLINE vmask16 operator>(vint16 a, vint16 b)
{
	return vmask16(_mm512_cmpgt_epi32_mask(a.m, b.m));
}

/**
 * @brief Logical shift left.
 */
template <int s> ASTCENC_SIMD_INLINE vint16 lsl(vint16 a)
{
	return vint16(_mm512_slli_epi32(a.m, s));
}

/**
 * @brief Arithmetic shift right.
 */
template <int s> ASTCENC_SIMD_INLINE vint16 asr(vint16 a)
{
	return vint16(_mm512_srai_epi32(a.m, s));
}

/**
 * @brief Logical shift right.
 */
template <int s> ASTCENC_SIMD_INLINE vint16 lsr(vint16 a)
{
	return vint16(_mm512_srli_epi32(a.m, s));
}

/**
 * @brief Return the min vector of two vectors.
 */
ASTCENC_SIMD_INLINE vint16 min(vint16 a, vint16 b)
{
	return vint16(_mm512_min_epi32(a.m, b.m));
}

/**
 * @brief Return the max vector of two vectors.
 */
ASTCENC_SIMD_INLINE vint16 max(vint16 a, vint16 b)
{
	return vint16(_mm512_max_epi32(a.m, b.m));
}

/**
 * @brief Return the horizontal minimum of a vector.
 */
ASTCENC_SIMD_INLINE int hmin_s(vint16 a)
{
	return _mm512_reduce_min_epi32(a.m);
}

/**
 * @brief Return the horizontal minimum of a vector.
 */
ASTCENC_SIMD_INLINE vint16 hmin(vint16 a)
{
	return vint16(hmin_s(a));
}

/**
 * @brief Return the horizontal maximum of a vector.
 */
ASTCENC_SIMD_INLINE int hmax_s(vint16 a)
{
	return _mm512_reduce_max_epi32(a.m);
}

/**
 * @brief Return the horizontal maximum of a vector.
 */
ASTCENC_SIMD_INLINE vint16 hmax(vint16 a)
{
	return vint16(hmax_s(a));
}

/**
 * @brief Store a vector to a 16B aligned memory address.
 */
ASTCENC_SIMD_INLINE void storea(vint16 a, int* p)
{
	_mm512_store_epi32(p, a.m);
}

/**
 * @brief Store a vector to an unaligned memory address.
 */
ASTCENC_SIMD_INLINE void store(vint16 a, int* p)
{
	_mm512_storeu_epi32(p, a.m);
}

/**
 * @brief Store lowest N (vector width) bytes into an unaligned address.
 */
ASTCENC_SIMD_INLINE void store_nbytes(vint16 a, uint8_t* p)
{
	_mm_storeu_si128(reinterpret_cast<__m128i*>(p), _mm512_extracti32x4_epi32(a.m, 0));
}

/**
 * @brief Gather N (vector width) indices from the array.
 */
ASTCENC_SIMD_INLINE vint16 gatheri(const int* base, vint16 indices)
{
	return vint16(_mm512_i32gather_epi32(indices.m, base, 4));
}

/**
 * @brief Pack low 8 bits of N (vector width) lanes into bottom of vector.
 */
ASTCENC_SIMD_INLINE vint16 pack_low_bytes(vint16 v)
{
	__m512i shuf = _mm512_set_epi8(
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0);
	vint16 a(_mm512_shuffle_epi8(v.m, shuf));
	return vint16(_mm512_set4_epi32(a.lane<12>(), a.lane<8>(), a.lane<4>(), a.lane<0>()));
}

/**
 * @brief Return lanes from @c b if @c cond is set, else @c a.
 */
ASTCENC_SIMD_INLINE vint16 select(vint16 a, vint16 b, vmask16 cond)
{
	return vint16(_mm512_mask_blend_epi32(cond.m, a.m, b.m));
}

// ============================================================================
// vfloat4 operators and functions
// ============================================================================

/**
 * @brief Overload: vector by vector addition.
 */
ASTCENC_SIMD_INLINE vfloat16 operator+(vfloat16 a, vfloat16 b)
{
	return vfloat16(_mm512_add_ps(a.m, b.m));
}

/**
 * @brief Overload: vector by vector incremental addition.
 */
ASTCENC_SIMD_INLINE vfloat16& operator+=(vfloat16& a, const vfloat16& b)
{
	a = a + b;
	return a;
}

/**
 * @brief Overload: vector by vector subtraction.
 */
ASTCENC_SIMD_INLINE vfloat16 operator-(vfloat16 a, vfloat16 b)
{
	return vfloat16(_mm512_sub_ps(a.m, b.m));
}

/**
 * @brief Overload: vector by vector multiplication.
 */
ASTCENC_SIMD_INLINE vfloat16 operator*(vfloat16 a, vfloat16 b)
{
	return vfloat16(_mm512_mul_ps(a.m, b.m));
}

/**
 * @brief Overload: vector by scalar multiplication.
 */
ASTCENC_SIMD_INLINE vfloat16 operator*(vfloat16 a, float b)
{
	return vfloat16(_mm512_mul_ps(a.m, _mm512_set1_ps(b)));
}

/**
 * @brief Overload: scalar by vector multiplication.
 */
ASTCENC_SIMD_INLINE vfloat16 operator*(float a, vfloat16 b)
{
	return vfloat16(_mm512_mul_ps(_mm512_set1_ps(a), b.m));
}

/**
 * @brief Overload: vector by vector division.
 */
ASTCENC_SIMD_INLINE vfloat16 operator/(vfloat16 a, vfloat16 b)
{
	return vfloat16(_mm512_div_ps(a.m, b.m));
}

/**
 * @brief Overload: vector by scalar division.
 */
ASTCENC_SIMD_INLINE vfloat16 operator/(vfloat16 a, float b)
{
	return vfloat16(_mm512_div_ps(a.m, _mm512_set1_ps(b)));
}


/**
 * @brief Overload: scalar by vector division.
 */
ASTCENC_SIMD_INLINE vfloat16 operator/(float a, vfloat16 b)
{
	return vfloat16(_mm512_div_ps(_mm512_set1_ps(a), b.m));
}


/**
 * @brief Overload: vector by vector equality.
 */
ASTCENC_SIMD_INLINE vmask16 operator==(vfloat16 a, vfloat16 b)
{
	return vmask16(_mm512_cmp_ps_mask(a.m, b.m, _CMP_EQ_OQ));
}

/**
 * @brief Overload: vector by vector inequality.
 */
ASTCENC_SIMD_INLINE vmask16 operator!=(vfloat16 a, vfloat16 b)
{
	return vmask16(_mm512_cmp_ps_mask(a.m, b.m, _CMP_NEQ_OQ));
}

/**
 * @brief Overload: vector by vector less than.
 */
ASTCENC_SIMD_INLINE vmask16 operator<(vfloat16 a, vfloat16 b)
{
	return vmask16(_mm512_cmp_ps_mask(a.m, b.m, _CMP_LT_OQ));
}

/**
 * @brief Overload: vector by vector greater than.
 */
ASTCENC_SIMD_INLINE vmask16 operator>(vfloat16 a, vfloat16 b)
{
	return vmask16(_mm512_cmp_ps_mask(a.m, b.m, _CMP_GT_OQ));
}

/**
 * @brief Overload: vector by vector less than or equal.
 */
ASTCENC_SIMD_INLINE vmask16 operator<=(vfloat16 a, vfloat16 b)
{
	return vmask16(_mm512_cmp_ps_mask(a.m, b.m, _CMP_LE_OQ));
}

/**
 * @brief Overload: vector by vector greater than or equal.
 */
ASTCENC_SIMD_INLINE vmask16 operator>=(vfloat16 a, vfloat16 b)
{
	return vmask16(_mm512_cmp_ps_mask(a.m, b.m, _CMP_GE_OQ));
}

/**
 * @brief Return the min vector of two vectors.
 *
 * If either lane value is NaN, @c b will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat16 min(vfloat16 a, vfloat16 b)
{
	return vfloat16(_mm512_min_ps(a.m, b.m));
}

/**
 * @brief Return the min vector of a vector and a scalar.
 *
 * If either lane value is NaN, @c b will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat16 min(vfloat16 a, float b)
{
	return min(a, vfloat16(b));
}

/**
 * @brief Return the max vector of two vectors.
 *
 * If either lane value is NaN, @c b will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat16 max(vfloat16 a, vfloat16 b)
{
	return vfloat16(_mm512_max_ps(a.m, b.m));
}

/**
 * @brief Return the max vector of a vector and a scalar.
 *
 * If either lane value is NaN, @c b will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat16 max(vfloat16 a, float b)
{
	return max(a, vfloat16(b));
}

/**
 * @brief Return the clamped value between min and max.
 *
 * It is assumed that neither @c min nor @c max are NaN values. If @c a is NaN
 * then @c min will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat16 clamp(float min, float max, vfloat16 a)
{
	// Do not reorder - second operand will return if either is NaN
	a.m = _mm512_max_ps(a.m, _mm512_set1_ps(min));
	a.m = _mm512_min_ps(a.m, _mm512_set1_ps(max));
	return a;
}

/**
 * @brief Return a clamped value between 0.0f and max.
 *
 * It is assumed that @c max is not a NaN value. If @c a is NaN then zero will
 * be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat16 clampz(float max, vfloat16 a)
{
	a.m = _mm512_max_ps(a.m, _mm512_setzero_ps());
	a.m = _mm512_min_ps(a.m, _mm512_set1_ps(max));
	return a;
}

/**
 * @brief Return a clamped value between 0.0f and 1.0f.
 *
 * If @c a is NaN then zero will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat16 clampzo(vfloat16 a)
{
	a.m = _mm512_max_ps(a.m, _mm512_setzero_ps());
	a.m = _mm512_min_ps(a.m, _mm512_set1_ps(1.0f));
	return a;
}

/**
 * @brief Return the absolute value of the float vector.
 */
ASTCENC_SIMD_INLINE vfloat16 abs(vfloat16 a)
{
	return vfloat16(_mm512_abs_ps(a.m));
}

/**
 * @brief Return a float rounded to the nearest integer value.
 */
ASTCENC_SIMD_INLINE vfloat16 round(vfloat16 a)
{
	return vfloat16(_mm512_roundscale_ps(a.m, _MM_FROUND_TO_NEAREST_INT));
}

/**
 * @brief Return the horizontal minimum of a vector.
 */
ASTCENC_SIMD_INLINE float hmin_s(vfloat16 a)
{
	return _mm512_reduce_min_ps(a.m);
}

/**
 * @brief Return the horizontal minimum of a vector.
 */
ASTCENC_SIMD_INLINE vfloat16 hmin(vfloat16 a)
{
	return vfloat16(hmin_s(a));
}

/**
 * @brief Return the horizontal maximum of a vector.
 */
ASTCENC_SIMD_INLINE float hmax_s(vfloat16 a)
{
	return _mm512_reduce_max_ps(a.m);
}

/**
 * @brief Return the horizontal maximum of a vector.
 */
ASTCENC_SIMD_INLINE vfloat16 hmax(vfloat16 a)
{
	return vfloat16(hmax_s(a));
}

/**
 * @brief Return the horizontal sum of a vector.
 */
ASTCENC_SIMD_INLINE float hadd_s(vfloat16 a)
{
	return _mm512_reduce_add_ps(a.m);
}

/**
 * @brief Return lanes from @c b if @c cond is set, else @c a.
 */
ASTCENC_SIMD_INLINE vfloat16 select(vfloat16 a, vfloat16 b, vmask16 cond)
{
	return vfloat16(_mm512_mask_blend_ps(cond.m, a.m, b.m));
}

/**
 * @brief Return lanes from @c b if MSB of @c cond is set, else @c a.
 */
ASTCENC_SIMD_INLINE vfloat16 select_msb(vfloat16 a, vfloat16 b, vmask16 cond)
{
	return vfloat16(_mm512_mask_blend_ps(cond.m, a.m, b.m));
}

/**
 * @brief Accumulate lane-wise sums for a vector, folded 4-wide.
 *
 * This is invariant with 4-wide implementations.
 */
ASTCENC_SIMD_INLINE void haccumulate(vfloat4& accum, vfloat16 a)
{
	accum.m = _mm_add_ps(accum.m, _mm512_extractf32x4_ps(a.m, 0));
	accum.m = _mm_add_ps(accum.m, _mm512_extractf32x4_ps(a.m, 1));
	accum.m = _mm_add_ps(accum.m, _mm512_extractf32x4_ps(a.m, 2));
	accum.m = _mm_add_ps(accum.m, _mm512_extractf32x4_ps(a.m, 3));
}

/**
 * @brief Accumulate lane-wise sums for a vector.
 *
 * This is NOT invariant with 4-wide implementations.
 */
ASTCENC_SIMD_INLINE void haccumulate(vfloat16& accum, vfloat16 a)
{
	accum += a;
}

/**
 * @brief Accumulate masked lane-wise sums for a vector, folded 4-wide.
 *
 * This is invariant with 4-wide implementations.
 */
ASTCENC_SIMD_INLINE void haccumulate(vfloat4& accum, vfloat16 a, vmask16 m)
{
	a = select(vfloat16::zero(), a, m);
	haccumulate(accum, a);
}

/**
 * @brief Accumulate masked lane-wise sums for a vector.
 *
 * This is NOT invariant with 4-wide implementations.
 */
ASTCENC_SIMD_INLINE void haccumulate(vfloat16& accum, vfloat16 a, vmask16 m)
{
	accum.m = _mm512_mask_add_ps(accum.m, m.m, accum.m, a.m);
}

/**
 * @brief Return the sqrt of the lanes in the vector.
 */
ASTCENC_SIMD_INLINE vfloat16 sqrt(vfloat16 a)
{
	return vfloat16(_mm512_sqrt_ps(a.m));
}

/**
 * @brief Load a vector of gathered results from an array;
 */
ASTCENC_SIMD_INLINE vfloat16 gatherf(const float* base, vint16 indices)
{
	return vfloat16(_mm512_i32gather_ps(indices.m, base, 4));
}

/**
 * @brief Store a vector to an unaligned memory address.
 */
ASTCENC_SIMD_INLINE void store(vfloat16 a, float* p)
{
	_mm512_storeu_ps(p, a.m);
}

/**
 * @brief Store a vector to a 32B aligned memory address.
 */
ASTCENC_SIMD_INLINE void storea(vfloat16 a, float* p)
{
	_mm512_store_ps(p, a.m);
}

/**
 * @brief Return a integer value for a float vector, using truncation.
 */
ASTCENC_SIMD_INLINE vint16 float_to_int(vfloat16 a)
{
	return vint16(_mm512_cvttps_epi32(a.m));
}

/**
 * @brief Return a integer value for a float vector, using round-to-nearest.
 */
ASTCENC_SIMD_INLINE vint16 float_to_int_rtn(vfloat16 a)
{
	a = a + vfloat16(0.5f);
	return vint16(_mm512_cvttps_epi32(a.m));
}


/**
 * @brief Return a float value for an integer vector.
 */
ASTCENC_SIMD_INLINE vfloat16 int_to_float(vint16 a)
{
	return vfloat16(_mm512_cvtepi32_ps(a.m));
}

/**
 * @brief Return a float value as an integer bit pattern (i.e. no conversion).
 *
 * It is a common trick to convert floats into integer bit patterns, perform
 * some bit hackery based on knowledge they are IEEE 754 layout, and then
 * convert them back again. This is the first half of that flip.
 */
ASTCENC_SIMD_INLINE vint16 float_as_int(vfloat16 a)
{
	return vint16(_mm512_castps_si512(a.m));
}

/**
 * @brief Return a integer value as a float bit pattern (i.e. no conversion).
 *
 * It is a common trick to convert floats into integer bit patterns, perform
 * some bit hackery based on knowledge they are IEEE 754 layout, and then
 * convert them back again. This is the second half of that flip.
 */
ASTCENC_SIMD_INLINE vfloat16 int_as_float(vint16 a)
{
	return vfloat16(_mm512_castsi512_ps(a.m));
}

/**
 * @brief Prepare a vtable lookup table for use with the native SIMD size.
 */
ASTCENC_SIMD_INLINE void vtable_prepare(vint4 t0, vint16& t0p)
{
	t0p.m = _mm512_castsi128_si512(t0.m);
	t0p.m = _mm512_inserti32x4(t0p.m, t0.m, 1);
	t0p.m = _mm512_inserti32x4(t0p.m, t0.m, 2);
	t0p.m = _mm512_inserti32x4(t0p.m, t0.m, 3);
}

/**
 * @brief Prepare a vtable lookup table for use with the native SIMD size.
 */
ASTCENC_SIMD_INLINE void vtable_prepare(vint4 t0, vint4 t1, vint16& t0p, vint16& t1p)
{
	vtable_prepare(t0, t0p);
	vtable_prepare(t0 ^ t1, t1p);
}

/**
 * @brief Prepare a vtable lookup table for use with the native SIMD size.
 */
ASTCENC_SIMD_INLINE void vtable_prepare(
	vint4 t0, vint4 t1, vint4 t2, vint4 t3,
	vint16& t0p, vint16& t1p, vint16& t2p, vint16& t3p)
{
	vtable_prepare(t0, t0p);
	vtable_prepare(t0 ^ t1, t1p);
	vtable_prepare(t1 ^ t2, t2p);
	vtable_prepare(t2 ^ t3, t3p);
}

/**
 * @brief Perform an 8-bit 16-entry table lookup, with 32-bit indexes.
 */
ASTCENC_SIMD_INLINE vint16 vtable_8bt_32bi(vint16 t0, vint16 idx)
{
	// Set index byte MSB to 1 for unused bytes so shuffle returns zero
	__m512i idxx = _mm512_or_si512(idx.m, _mm512_set1_epi32(0xFFFFFF00));

	__m512i result = _mm512_shuffle_epi8(t0.m, idxx);
	return vint16(result);
}

/**
 * @brief Perform an 8-bit 32-entry table lookup, with 32-bit indexes.
 */
ASTCENC_SIMD_INLINE vint16 vtable_8bt_32bi(vint16 t0, vint16 t1, vint16 idx)
{
	// Set index byte MSB to 1 for unused bytes so shuffle returns zero
	__m512i idxx = _mm512_or_si512(idx.m, _mm512_set1_epi32(0xFFFFFF00));

	__m512i result = _mm512_shuffle_epi8(t0.m, idxx);
	idxx = _mm512_sub_epi8(idxx, _mm512_set1_epi8(16));

	__m512i result2 = _mm512_shuffle_epi8(t1.m, idxx);
	result = _mm512_xor_si512(result, result2);
	return vint16(result);
}

/**
 * @brief Perform an 8-bit 64-entry table lookup, with 32-bit indexes.
 */
ASTCENC_SIMD_INLINE vint16 vtable_8bt_32bi(vint16 t0, vint16 t1, vint16 t2, vint16 t3, vint16 idx)
{
	// Set index byte MSB to 1 for unused bytes so shuffle returns zero
	__m512i idxx = _mm512_or_si512(idx.m, _mm512_set1_epi32(0xFFFFFF00));

	__m512i result = _mm512_shuffle_epi8(t0.m, idxx);
	idxx = _mm512_sub_epi8(idxx, _mm512_set1_epi8(16));

	__m512i result2 = _mm512_shuffle_epi8(t1.m, idxx);
	result = _mm512_xor_si512(result, result2);
	idxx = _mm512_sub_epi8(idxx, _mm512_set1_epi8(16));

	result2 = _mm512_shuffle_epi8(t2.m, idxx);
	result = _mm512_xor_si512(result, result2);
	idxx = _mm512_sub_epi8(idxx, _mm512_set1_epi8(16));

	result2 = _mm512_shuffle_epi8(t3.m, idxx);
	result = _mm512_xor_si512(result, result2);

	return vint16(result);
}

/**
 * @brief Return a vector of interleaved RGBA data.
 *
 * Input vectors have the value stored in the bottom 8 bits of each lane,
 * with high  bits set to zero.
 *
 * Output vector stores a single RGBA texel packed in each lane.
 */
ASTCENC_SIMD_INLINE vint16 interleave_rgba8(vint16 r, vint16 g, vint16 b, vint16 a)
{
	return r + lsl<8>(g) + lsl<16>(b) + lsl<24>(a);
}

/**
 * @brief Store a vector, skipping masked lanes.
 *
 * All masked lanes must be at the end of vector, after all non-masked lanes.
 */
ASTCENC_SIMD_INLINE void store_lanes_masked(uint8_t* base, vint16 data, vmask16 mask)
{
	_mm512_mask_store_epi32(base, mask.m, data.m);
}

/**
 * @brief Debug function to print a vector of ints.
 */
ASTCENC_SIMD_INLINE void print(vint16 a)
{
	alignas(64) int v[16];
	storea(a, v);
	printf("v16_i32:\n  %8d %8d %8d %8d %8d %8d %8d %8d %8d %8d %8d %8d %8d %8d %8d %8d\n",
	    v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
		v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
}

/**
 * @brief Debug function to print a vector of ints.
 */
ASTCENC_SIMD_INLINE void printx(vint16 a)
{
	alignas(64) int v[16];
	storea(a, v);
	printf("v16_i32:\n  %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x\n",
	    v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
		v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
}

/**
 * @brief Debug function to print a vector of floats.
 */
ASTCENC_SIMD_INLINE void print(vfloat16 a)
{
	alignas(64) float v[16];
	storea(a, v);
	printf("v16_f32:\n  %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f\n",
		v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
		v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
}

/**
 * @brief Debug function to print a vector of masks.
 */
ASTCENC_SIMD_INLINE void print(vmask16 a)
{
	print(select(vint16(0), vint16(1), a));
}

#endif // #ifndef ASTC_VECMATHLIB_AVX512_16_H_INCLUDED
