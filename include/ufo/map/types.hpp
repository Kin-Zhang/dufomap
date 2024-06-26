/*!
 * UFOMap: An Efficient Probabilistic 3D Mapping Framework That Embraces the Unknown
 *
 * @author Daniel Duberg (dduberg@kth.se)
 * @see https://github.com/UnknownFreeOccupied/ufomap
 * @version 1.0
 * @date 2022-05-13
 *
 * @copyright Copyright (c) 2022, Daniel Duberg, KTH Royal Institute of Technology
 *
 * BSD 3-Clause License
 *
 * Copyright (c) 2022, Daniel Duberg, KTH Royal Institute of Technology
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef UFO_MAP_TYPES_HPP
#define UFO_MAP_TYPES_HPP

// STL
#include <array>
#include <concepts>
#include <cstdint>
#include <limits>
#include <vector>

namespace ufo
{
enum class OccupancyState { UNKNOWN, FREE, OCCUPIED };
enum class PropagationCriteria { MIN, MAX, MEAN, FIRST, NONE };
enum class Device { CPU, GPU };

using coord_t         = float;
using node_size_t     = double;
using pos_t           = std::uint32_t;
using offset_t        = std::uint_fast8_t;
using depth_t         = std::uint8_t;
using key_t           = std::uint_fast32_t;
using code_t          = std::uint64_t;
using occupancy_t     = float;
using logit_t         = std::int8_t;
using time_t          = float;
using color_t         = std::uint8_t;
using label_t         = std::uint32_t;
using value_t         = float;
using intensity_t     = float;
using count_t         = std::int32_t;
using reflection_t    = double;
using distance_t      = float;
using surfel_scalar_t = float;
using freedom_t       = std::uint8_t;

template <class T, std::size_t N>
using DataBlock = std::array<T, N>;

template <class T>
using Container = std::vector<T>;

//
// Null pos
//

static constexpr pos_t const NULL_POS = std::numeric_limits<pos_t>::max();

//
// Map types
//

using mt_t = std::uint64_t;

enum MapType : mt_t {
	NONE           = 0,
	ALL            = ~mt_t(0),
	OCCUPANCY      = mt_t(1),
	TIME           = mt_t(1) << 1,
	COLOR          = mt_t(1) << 2,
	LABEL          = mt_t(1) << 3,
	LABELS         = mt_t(1) << 4,
	SEMANTIC       = mt_t(1) << 5,
	SEMANTICS      = mt_t(1) << 6,
	SURFEL         = mt_t(1) << 7,
	DISTANCE       = mt_t(1) << 8,
	INTENSITY      = mt_t(1) << 9,
	COUNT          = mt_t(1) << 10,
	REFLECTION     = mt_t(1) << 11,
	POINT          = mt_t(1) << 12,
	POINTS8        = mt_t(1) << 13,
	POINTS16       = mt_t(1) << 14,
	POINTS32       = mt_t(1) << 15,
	POINTS_COLOR8  = mt_t(1) << 16,
	POINTS_COLOR16 = mt_t(1) << 17,
	POINTS_COLOR32 = mt_t(1) << 18,
	FREE           = mt_t(1) << 19,
	SEEN_FREE      = mt_t(1) << 20,
	VALUE          = mt_t(1) << 21,
};

//
// Concepts
//

template <class Map, MapType MT>
concept IsMapType = MT == (Map::mapType() & MT);
}  // namespace ufo

#endif  // UFO_MAP_TYPES_HPP