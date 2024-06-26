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

#ifndef UFO_ALGORITHM_HPP
#define UFO_ALGORITHM_HPP

// STL
#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

// STL parallel
#ifdef UFO_PARALLEL
#include <execution>
#endif

namespace ufo
{
using Permutation = std::vector<std::size_t>;

/*!
 * Retrieve the permutation to sort the elements in the range [first, last) in
 * non-descending order. The order of equal elements is not guaranteed.
 *
 * @param first,last The range of elements to sort.
 * @return The permutation.
 */
[[nodiscard]] Permutation sortPermutation(std::random_access_iterator auto first,
                                          std::random_access_iterator auto last)
{
	Permutation p(std::distance(first, last));
	std::iota(std::begin(p), std::end(p), std::size_t{});
	std::sort(std::begin(p), std::end(p), [first](std::size_t i, std::size_t j) {
		return *(first + i) < *(first + j);
	});
	return p;
}

/*!
 * Retrieve the permutation to sort the elements in the range [first, last) in
 * non-descending order. The order of equal elements is not guaranteed.
 *
 * @param first,last The range of elements to sort.
 * @param comp Comparison function object which returns true if the first argument is less
 * than (i.e., is ordered before) the second.
 * @return The permutation.
 */
template <class Compare>
[[nodiscard]] Permutation sortPermutation(std::random_access_iterator auto first,
                                          std::random_access_iterator auto last,
                                          Compare                          comp)
{
	Permutation p(std::distance(first, last));
	std::iota(std::begin(p), std::end(p), std::size_t{});
	std::sort(std::begin(p), std::end(p), [first, comp](std::size_t i, std::size_t j) {
		return comp(*(first + i), *(first + j));
	});
	return p;
}

// TODO: Add comment
template <class Container>
[[nodiscard]] Permutation sortPermutation(Container const& c)
{
	return sortPermutation(std::cbegin(c), std::cend(c));
}

// TODO: Add comment
template <class Container, class Compare>
[[nodiscard]] Permutation sortPermutation(Container const& c, Compare comp)
{
	return sortPermutation(std::cbegin(c), std::cend(c), comp);
}

/*!
 * Apply permutation to the elements in the range [first, last).
 *
 * @code
 * auto perm = sortPermutation(a, b);
 * applyPermutation(a, b, perm);
 * @endcode
 *
 * @param first,last The range of elements to permutate.
 * @param perm The permutation.
 */
void applyPermutation(std::random_access_iterator auto first,
                      std::random_access_iterator auto last, Permutation const& perm)
{
	auto const size = perm.size();

	assert(std::distance(first, last) == size);

	std::vector<bool> done(size, false);
	for (std::size_t i{}; i != size; ++i) {
		if (std::as_const(done)[i]) {
			continue;
		}

		for (std::size_t prev_j{i}, j{perm[i]}; i != j; prev_j = j, j = perm[j]) {
			std::iter_swap(first + prev_j, first + j);
			done[j] = true;
		}
	}
}

// TODO: Add comment
template <class Container>
void applyPermutation(Container& c, Permutation const& perm)
{
	applyPermutation(std::begin(c), std::end(c), perm);
}
}  // namespace ufo

#endif  // UFO_ALGORITHM_HPP