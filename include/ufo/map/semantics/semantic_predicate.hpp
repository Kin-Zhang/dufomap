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

#ifndef UFO_MAP_PREDICATE_SEMANTICS_HPP
#define UFO_MAP_PREDICATE_SEMANTICS_HPP

// UFO
#include <ufo/map/predicate/predicate.hpp>
#include <ufo/map/semantic/semantic.hpp>

// STL
#include <initializer_list>
#include <unordered_set>

namespace ufo::pred
{
//
// Predicates
//

struct AnyLabel {
	AnyLabel(std::initializer_list<std::string> ilist) : strings(ilist), need_fetch(true) {}

	AnyLabel(std::initializer_list<label_t> ilist) : ranges(ilist), need_fetch(false) {}

	AnyLabel(std::initializer_list<SemanticRange> range) : ranges(range), need_fetch(false)
	{
	}

	template <class... Args>
	AnyLabel(Args&&... args)
	{
		add(std::forward<Args>(args)...);
	}

	void add(std::string const& tag)
	{
		need_fetch = strings.insert(arg).second || need_fetch;
	}

	void add(label_t label) { ranges.insert(label); }

	void add(SemanticRange range) { ranges.insert(range); }

	template <class T, class... Args,
	          typename std::enable_if<0 < sizeof...(Args), int>::type = 0>
	void add(T const& arg, Args&&... args)
	{
		add(arg);
		add(std::forward<Args>(args)...);
	}

	void clear()
	{
		strings.clear();
		ranges.clear();
		need_fetch = false;
	}

 private:
	std::unordered_set<std::string> tags;
	SemanticRangeSet                labels;
	bool                            need_fetch;
};

struct AllLabels {
};

struct NotLabel {
};

struct AnyLabelValue {
};

struct AnyLabelValueInterval {
};

struct AllLabelsValue {
};

struct AllLabelsValueInterval {
};

struct LabelValue {
};

struct LabelValueInterval {
};

//
// Predicate value/return check
//

//
// Predicate inner check
//

}  // namespace ufo::pred

#endif  // UFO_MAP_PREDICATE_SEMANTICS_HPP