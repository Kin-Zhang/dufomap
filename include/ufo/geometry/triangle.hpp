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

#ifndef UFO_GEOMETRY_TRIANGLE_HPP
#define UFO_GEOMETRY_TRIANGLE_HPP

#include <ufo/geometry/point.hpp>

namespace ufo
{
struct Triangle {
	std::array<Point, 3> points;

	constexpr Triangle() noexcept = default;

	constexpr Triangle(Point point_1, Point point_2, Point point_3) noexcept
	    : points{point_1, point_2, point_3}
	{
	}

	bool operator==(Triangle const& rhs) const noexcept { return rhs.points == points; }

	bool operator!=(Triangle const& rhs) const noexcept { return !(*this == rhs); }

	constexpr Point min() const noexcept
	{
		return Point(std::min({points[0].x, points[1].x, points[2].x}),
		             std::min({points[0].y, points[1].y, points[2].y}),
		             std::min({points[0].z, points[1].z, points[2].z}));
	}

	constexpr Point max() const noexcept
	{
		return Point(std::max({points[0].x, points[1].x, points[2].x}),
		             std::max({points[0].y, points[1].y, points[2].y}),
		             std::max({points[0].z, points[1].z, points[2].z}));
	}
};
}  // namespace ufo

#endif  // UFO_GEOMETRY_TRIANGLE_HPP