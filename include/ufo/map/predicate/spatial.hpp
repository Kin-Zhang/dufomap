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

#ifndef UFO_MAP_PREDICATE_SPATIAL_HPP
#define UFO_MAP_PREDICATE_SPATIAL_HPP

// UFO
#include <bits/ranges_algo.h>

#include <ufo/geometry/contains.hpp>
#include <ufo/geometry/intersects.hpp>
#include <ufo/map/predicate/predicate.hpp>

// STL
#include <type_traits>

namespace ufo::pred
{
//
// Spatial tag
//

enum class SpatialTag {
	CONTAINS,    // Geometry has to be inside node
	DISJOINT,    // Node and geometry are disjoint
	INTERSECTS,  // Node and geometry intersects
	WITHIN       // Node has to be inside geometry
};

template <SpatialTag Tag>
constexpr char const* spatialTagToString()
{
	if constexpr (SpatialTag::CONTAINS == Tag) {
		return "contains";
	} else if constexpr (SpatialTag::DISJOINT == Tag) {
		return "disjoint";
	} else if constexpr (SpatialTag::INTERSECTS == Tag) {
		return "intersects";
	} else if constexpr (SpatialTag::WITHIN == Tag) {
		return "within";
	} else {
		return "";
	}
}

//
// Spatial
//

template <class Geometry, SpatialTag Tag, bool Negated>
struct Spatial {
	// constexpr Spatial(Geometry geometry) : geometries(geometry)
	// {
	// }

	template <std::input_iterator I, std::sentinel_for<I> S>
	constexpr Spatial(I first, S last) : geometries(first, last)
	{
	}

	constexpr Spatial(std::initializer_list<Geometry> geometries) : geometries(geometries)
	{
	}

	std::vector<Geometry> geometries;
};

// template <class Geometry>
// constexpr auto Contains(Geometry geometry)
// {
// 	return Spatial<Geometry, SpatialTag::CONTAINS, false>(geometry);
// }

template <std::input_iterator I, std::sentinel_for<I> S>
constexpr auto Contains(I first, S last)
{
	return Spatial<typename std::iterator_traits<I>::value_type, SpatialTag::CONTAINS,
	               false>(first, last);
}

template <std::ranges::input_range R>
constexpr auto Contains(R&& r)
{
	return Contains(std::ranges::begin(r), std::ranges::end(r));
}

template <class Geometry>
constexpr auto Contains(std::initializer_list<Geometry> geometries)
{
	return Spatial<Geometry, SpatialTag::CONTAINS, false>(geometries);
}

// template <class Geometry>
// constexpr auto Disjoint(Geometry geometry)
// {
// 	return Spatial<Geometry, SpatialTag::DISJOINT, false>(geometry);
// }

template <std::input_iterator I, std::sentinel_for<I> S>
constexpr auto Disjoint(I first, S last)
{
	return Spatial<typename std::iterator_traits<I>::value_type, SpatialTag::DISJOINT,
	               false>(first, last);
}

template <std::ranges::input_range R>
constexpr auto Disjoint(R&& r)
{
	return Disjoint(std::ranges::begin(r), std::ranges::end(r));
}

template <class Geometry>
constexpr auto Disjoint(std::initializer_list<Geometry> geometries)
{
	return Spatial<Geometry, SpatialTag::DISJOINT, false>(geometries);
}

// template <class Geometry>
// constexpr auto Intersects(Geometry geometry)
// {
// 	return Spatial<Geometry, SpatialTag::INTERSECTS, false>(geometry);
// }

template <std::input_iterator I, std::sentinel_for<I> S>
constexpr auto Intersects(I first, S last)
{
	return Spatial<typename std::iterator_traits<I>::value_type, SpatialTag::INTERSECTS,
	               false>(first, last);
}

template <std::ranges::input_range R>
constexpr auto Intersects(R&& r)
{
	return Intersects(std::ranges::begin(r), std::ranges::end(r));
}

template <class Geometry>
constexpr auto Intersects(std::initializer_list<Geometry> geometries)
{
	return Spatial<Geometry, SpatialTag::INTERSECTS, false>(geometries);
}

// template <class Geometry>
// constexpr auto Within(Geometry geometry)
// {
// 	return Spatial<Geometry, SpatialTag::WITHIN, false>(geometry);
// }

template <std::input_iterator I, std::sentinel_for<I> S>
constexpr auto Within(I first, S last)
{
	return Spatial<typename std::iterator_traits<I>::value_type, SpatialTag::WITHIN, false>(
	    first, last);
}

template <std::ranges::input_range R>
constexpr auto Within(R&& r)
{
	return Within(std::ranges::begin(r), std::ranges::end(r));
}

template <class Geometry>
constexpr auto Within(std::initializer_list<Geometry> geometries)
{
	return Spatial<Geometry, SpatialTag::WITHIN, false>(geometries);
}

//
// Is spatial predicate
//

// FIXME: These are probably wrong

template <class>
struct is_spatial_pred_impl : std::false_type {
};

template <class Geometry, SpatialTag Tag, bool Negated>
struct is_spatial_pred_impl<Spatial<Geometry, Tag, Negated>> : std::true_type {
};

template <class Geometry, SpatialTag Tag, bool Negated>
struct is_spatial_pred
    : is_spatial_pred_impl<std::remove_cv_t<Spatial<Geometry, Tag, Negated>>> {
};

// Helper variable template
// template <class T>
// inline constexpr bool is_spatial_pred_v = is_spatial_pred<T>::value;

//
// Spatial negate
//

template <class Geometry, SpatialTag Tag, bool Negated>
constexpr Spatial<Geometry, Tag, !Negated> operator!(
    Spatial<Geometry, Tag, Negated> const& p)
{
	return Spatial<Geometry, Tag, !Negated>(p.geometries);
}

//
// Spatial call
//

template <bool Check, SpatialTag Tag>
struct static_assert_check_spatial : std::bool_constant<Check> {
};

template <SpatialTag Tag>
struct SpatialCall {
	static_assert(static_assert_check_spatial<false, Tag>::value,
	              "Not implemented for the Tag.");
};

template <>
struct SpatialCall<SpatialTag::CONTAINS> {
	template <class G1, class G2>
	static inline bool apply(G1 const& g1, std::vector<G2> const& g2)
	{
		// TODO: Implement correct
		return std::ranges::all_of(g2, [&g1](auto const& g) { return contains(g1, g); });
		// return contains(g1, g2);
	}
};

template <>
struct SpatialCall<SpatialTag::DISJOINT> {
	template <class G1, class G2>
	static inline bool apply(G1 const& g1, std::vector<G2> const& g2)
	{
		// TODO: Implement correct
		return std::ranges::all_of(g2, [&g1](auto const& g) { return !intersects(g1, g); });
		// return !intersects(g1, g2);
	}
};

template <>
struct SpatialCall<SpatialTag::INTERSECTS> {
	template <class G1, class G2>
	static inline bool apply(G1 const& g1, std::vector<G2> const& g2)
	{
		// FIXME: Make sure that it is correct
		return std::ranges::any_of(g2, [&g1](auto const& g) { return intersects(g1, g); });
		// return intersects(g1, g2);
	}
};

template <>
struct SpatialCall<SpatialTag::WITHIN> {
	template <class G1, class G2>
	static inline bool apply(G1 const& g1, std::vector<G2> const& g2)
	{
		// TODO: Implement correct
		return std::ranges::any_of(g2, [&g1](auto const& g) { return contains(g, g1); });
		// return contains(g2, g1);
	}
};

//
// Predicate value check
//

template <class Geometry, SpatialTag Tag, bool Negated>
struct ValueCheck<Spatial<Geometry, Tag, Negated>> {
	using Pred = Spatial<Geometry, Tag, Negated>;

	template <class Tree, class Node>
	static inline bool apply(Pred const& p, Tree const& m, Node const& n)
	{
		if constexpr (Negated) {
			return !SpatialCall<Tag>::apply(m.boundingVolume(n), p.geometries);
		} else {
			return SpatialCall<Tag>::apply(m.boundingVolume(n), p.geometries);
		}
	}
};

//
// Predicate inner check
//

// Default
template <class Geometry, SpatialTag Tag>
struct InnerCheck<Spatial<Geometry, Tag, false>> {
	using Pred = Spatial<Geometry, Tag, false>;

	template <class Tree, class Node>
	static inline bool apply(Pred const& p, Tree const& m, Node const& n)
	{
		return SpatialCall<SpatialTag::INTERSECTS>::apply(m.boundingVolume(n), p.geometries);
	}
};

// Contains
template <class Geometry>
struct InnerCheck<Spatial<Geometry, SpatialTag::CONTAINS, false>> {
	using Pred = Spatial<Geometry, SpatialTag::CONTAINS, false>;

	template <class Tree, class Node>
	static inline bool apply(Pred const& p, Tree const& m, Node const& n)
	{
		return SpatialCall<SpatialTag::CONTAINS>::apply(m.boundingVolume(n), p.geometries);
	}
};

// Disjoint
template <class Geometry>
struct InnerCheck<Spatial<Geometry, SpatialTag::DISJOINT, false>> {
	using Pred = Spatial<Geometry, SpatialTag::DISJOINT, false>;

	template <class Tree, class Node>
	static inline bool apply(Pred const& p, Tree const& m, Node const& n)
	{
		return !SpatialCall<SpatialTag::WITHIN>::apply(m.boundingVolume(n), p.geometries);
	}
};

//
// Negated
//

// Default
template <class Geometry, SpatialTag Tag>
struct InnerCheck<Spatial<Geometry, Tag, true>> {
	using Pred = Spatial<Geometry, Tag, true>;

	template <class Tree, class Node>
	static inline bool apply(Pred const& p, Tree const& m, Node const& n)
	{
		return !SpatialCall<SpatialTag::WITHIN>::apply(m.boundingVolume(n), p.geometries);
	}
};

// Contains
template <class Geometry>
struct InnerCheck<Spatial<Geometry, SpatialTag::CONTAINS, true>> {
	using Pred = Spatial<Geometry, SpatialTag::CONTAINS, true>;

	template <class Tree, class Node>
	static inline constexpr bool apply(Pred const&, Tree const&, Node const&)
	{
		return true;
	}
};

// Disjoint
template <class Geometry>
struct InnerCheck<Spatial<Geometry, SpatialTag::DISJOINT, true>> {
	using Pred = Spatial<Geometry, SpatialTag::DISJOINT, true>;

	template <class Tree, class Node>
	static inline bool apply(Pred const& p, Tree const& m, Node const& n)
	{
		return SpatialCall<SpatialTag::INTERSECTS>::apply(m.boundingVolume(n), p.geometries);
	}
};

//
// Contains spatial predicate
//

template <typename>
struct has_spatial_type : std::false_type {
};

template <class Geometry, SpatialTag Tag, bool Negated>
struct has_spatial_type<Spatial<Geometry, Tag, Negated>> : std::true_type {
};

template <typename L, typename R>
struct has_spatial_type<OR<L, R>>
    : std::conditional_t<bool(has_spatial_type<L>::value), std::true_type,
                         has_spatial_type<R>> {
};

template <typename L, typename R>
struct has_spatial_type<THEN<L, R>>
    : std::conditional_t<bool(has_spatial_type<L>::value), std::true_type,
                         has_spatial_type<R>> {
};

template <typename... Ts>
struct has_spatial_type<std::tuple<Ts...>> : std::disjunction<has_spatial_type<Ts>...> {
};

template <typename Predicates>
using contains_spatial_predicate = has_spatial_type<Predicates>;

template <typename Predicates>
inline constexpr bool contains_spatial_predicate_v =
    contains_spatial_predicate<Predicates>::value;

//
// Contains always spatial predicate
//

template <typename>
struct has_always_spatial_type : std::false_type {
};

template <class Geometry, SpatialTag Tag, bool Negated>
struct has_always_spatial_type<Spatial<Geometry, Tag, Negated>> : std::true_type {
};

template <typename L, typename R>
struct has_always_spatial_type<OR<L, R>> : std::false_type {
};

// FIXME: Should this be false if L is Spatial predicate?
template <typename L, typename R>
struct has_always_spatial_type<THEN<L, R>> : std::false_type {
};

template <typename... Ts>
struct has_always_spatial_type<std::tuple<Ts...>>
    : std::disjunction<has_always_spatial_type<Ts>...> {
};

template <typename Predicates>
using contains_always_spatial_predicate = has_always_spatial_type<Predicates>;

template <typename Predicates>
inline constexpr bool contains_always_spatial_predicate_v =
    contains_always_spatial_predicate<Predicates>::value;

}  // namespace ufo::pred

#endif  // UFO_MAP_PREDICATE_SPATIAL_HPP