/**
 * @file types.hpp
 * @brief Common integer, floating-point, and memory type aliases for the Kribu engine.
 * @details This file defines type shorthands used across the performance-critical Sholo Guti engine.
 *          It standardizes fixed-width and architecture-dependent types to ensure portability and
 *          compact memory layouts.
 */

#pragma once

#include <cstddef>
#include <cstdint>

/**
 * @typedef i8
 * @brief 8-bit signed integer.
 * @details Range: -128 to 127. Typically used for small counts, coordinates, or indices.
 */
using i8 = std::int8_t;

/**
 * @typedef i16
 * @brief 16-bit signed integer.
 * @details Range: -32,768 to 32,767. Typically used for move IDs, flags, or local offsets.
 */
using i16 = std::int16_t;

/**
 * @typedef i32
 * @brief 32-bit signed integer.
 * @details Range: -2,147,483,648 to 2,147,483,647. Used for general math, game results, and loop bounds.
 */
using i32 = std::int32_t;

/**
 * @typedef i64
 * @brief 64-bit signed integer.
 * @details Range: -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807. Used for very large integer metrics or time
 * stamps.
 */
using i64 = std::int64_t;

/**
 * @typedef u8
 * @brief 8-bit unsigned integer.
 * @details Range: 0 to 255. Often used for small positive constants and compact array indices.
 */
using u8 = std::uint8_t;

/**
 * @typedef u16
 * @brief 16-bit unsigned integer.
 * @details Range: 0 to 65,535. Used for unsigned move lists or counts.
 */
using u16 = std::uint16_t;

/**
 * @typedef u32
 * @brief 32-bit unsigned integer.
 * @details Range: 0 to 4,294,967,295. Used for positive identifiers, sizes, and indices.
 */
using u32 = std::uint32_t;

/**
 * @typedef u64
 * @brief 64-bit unsigned integer.
 * @details Range: 0 to 18,446,744,073,709,551,615. Used heavily for bitboards/bitmasks (e.g., me/opp sets).
 */
using u64 = std::uint64_t;

/**
 * @typedef usize
 * @brief Architecture-dependent unsigned size type.
 * @details Equivalent to std::size_t. Used for array/vector sizes and indexing operations.
 */
using usize = std::size_t;

/**
 * @typedef isize
 * @brief Architecture-dependent signed offset/size type.
 * @details Equivalent to std::ptrdiff_t. Used for pointer arithmetic and pointer offsets.
 */
using isize = std::ptrdiff_t;

/**
 * @typedef uptr
 * @brief Unsigned integer guaranteed to be large enough to hold a pointer.
 * @details Range: 0 to UINTPTR_MAX. Used for low-level memory addressing operations.
 */
using uptr = std::uintptr_t;

/**
 * @typedef f32
 * @brief Single-precision floating-point number.
 * @details Conforms to IEEE 754. Used for non-precision critical floats.
 */
using f32 = float;

/**
 * @typedef f64
 * @brief Double-precision floating-point number.
 * @details Conforms to IEEE 754. Used for high-precision evaluations or weights.
 */
using f64 = double;

/**
 * @typedef byte
 * @brief Represents raw, untyped memory bytes.
 * @details Equivalent to std::byte. Used for low-level memory buffers.
 */
using byte = std::byte;
