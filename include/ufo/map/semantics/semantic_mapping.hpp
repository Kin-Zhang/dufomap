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

#ifndef UFO_MAP_SEMANTIC_LABEL_MAPPING_HPP
#define UFO_MAP_SEMANTIC_LABEL_MAPPING_HPP

// UFO
#include <ufo/container/range.hpp>
#include <ufo/map/color/color.hpp>
#include <ufo/map/semantic/semantic.hpp>

// STL
#include <algorithm>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace ufo
{
class SemanticMapping
{
 public:
	void add(std::string const& name, label_t label) { add(name, SemanticRange(label)); }

	void add(std::string const& name, SemanticRange labels)
	{
		add(name, SemanticRangeSet(labels));
	}

	void add(std::string const& name, SemanticRangeSet const& labels)
	{
		if (consumer_mapping_[name].ranges.insert(std::begin(labels), std::end(labels))) {
			if (mapping_[name].ranges.insert(std::begin(labels), std::end(labels))) {
				changed_ = true;
			}
		}
	}

	void add(std::string const& name, std::string const& other_name)
	{
		if (consumer_mapping_[name].strings.insert(other_name).second) {
			if (mapping_[name].strings.insert(other_name).second) {
				changed_ = true;
			}
		}
	}

	void setColor(std::string const& name, Color color)
	{
		if (0 == consumer_mapping_.count(name) || consumer_mapping_[name].color != color) {
			consumer_mapping_[name].color = color;
			if (0 == mapping_.count(name) || mapping_[name].color != color) {
				mapping_[name].color = color;
				changed_             = true;
			}
		}
	}

	void clear()
	{
		clearLabel();
		clearString();
		clearColor();
	}

	void clear(std::string const& name)
	{
		clearLabel(name);
		clearString(name);
		clearColor(name);
	}

	void clearLabel()
	{
		// TODO: Implement
		std::for_each(std::begin(mapping_), std::end(mapping_),
		              [](auto const& m) { m.ranges.clear(); });
	}

	void clearLabel(std::string const& name)
	{
		// TODO: Implement
		mapping_[name].ranges.clear();
	}

	void clearString()
	{
		// TODO: Implement
		std::for_each(std::begin(mapping_), std::end(mapping_),
		              [](auto const& m) { m.strings.clear(); });
	}

	void clearString(std::string const& name)
	{
		// TODO: Implement
		mapping_[name].strings.clear();
	}

	void clearColor()
	{
		// TODO: Implement
		std::for_each(std::begin(mapping_), std::end(mapping_),
		              [](auto const& m) { m.color.clear(); });
	}

	void clearColor(std::string const& name)
	{
		// TODO: Implement
		mapping_[name].color.clear();
	}

	void erase(std::string const& name, label_t label)
	{
		// TODO: Implement
		mapping_[name].ranges.erase(label);
	}

	void erase(std::string const& name, SemanticRange labels)
	{
		// TODO: Implement
		mapping_[name].ranges.erase(labels);
	}

	void erase(std::string const& name, SemanticRange labels)
	{
		// TODO: Implement
		for (auto const& label : labels) {
			mapping_[name].ranges.erase(label);
		}
	}

	void erase(std::string const& name, std::string const& other_name)
	{
		// TODO: Implement
		mapping_[name].strings.erase(other_name);
	}

	SemanticRangeSet ranges(std::string const& name, bool recursive = true) const
	{
		if (!recursive) {
			if (auto it = mapping_.find(name); mapping_.end() != it) {
				return it->second.ranges;
			}
			return SemanticRangeSet();
		}

		SemanticRangeSet labels;
		for (auto const& str : strings(name, true)) {
			if (auto it = mapping_.find(str); mapping_.end() != it) {
				labels.insert(std::cbegin(it->second.ranges), std::cend(it->second.ranges));
			}
		}
		return labels;
	}

	std::unordered_set<std::string> strings(std::string const& name,
	                                        bool               recursive = true) const
	{
		if (!recursive) {
			if (auto it = mapping_.find(name); mapping_.end() != it) {
				return it->second.strings;
			}
			return std::unordered_set<std::string>();
		}

		std::unordered_set<std::string> names;
		names.insert(name);
		std::queue<std::string> queue;
		queue.push(name);
		while (!queue.empty()) {
			std::string cur = queue.front();
			queue.pop();
			auto it = mapping_.find(cur);
			if (mapping_.end() == it) {
				continue;
			}
			for (auto const& str : it->second.strings) {
				if (auto it = mapping_.find(str); mapping_.end() != it) {
					if (names.insert(str).second) {
						queue.push(str);
					}
				}
			}
		}

		return names;
	}

	Color color(std::string const& name) const
	{
		if (auto it = mapping_.find(name); mapping_.end() != it) {
			return it->second.color;
		}
		return Color();
	}

	void read(std::istream& in)
	{
		std::uint64_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		std::vector<char> producer_serialized(size);
		in.read(producer_serialized.data(), size);

		if (producer_serialized == producer_serialized_) {
			return;
		}

		producer_serialized_.swap(producer_serialized);

		producer_mapping_.clear();
		char*       first = producer_serialized_.data();
		char*       last  = first + producer_serialized_.size();
		std::string str;
		while (first != last) {
			std::memcpy(&size, first, sizeof(size));
			first += sizeof(size);
			str.resize(size);
			std::memcpy(str.data(), first, size);
			first += size;
			first = producer_mapping_[str].read(first);
		}

		mapping_ = producer_mapping_;

		// NOTE: Important that consumer is after producer so it overwrites
		for (auto const& [name, data] : consumer_mapping_) {
			auto& mapping_data = mapping_[name];

			mapping_data.ranges.insert(std::begin(data.ranges), std::end(data.ranges));
			mapping_data.strings.insert(std::begin(data.strings), std::end(data.strings));
			if (data.color.set()) {
				mapping_data.color = data.color;
			}
		}

		changed_ = true;
	}

	void write(std::ostream& out) const
	{
		if (changed_) {
			std::uint64_t size = 0;
			for (auto const& [name, data] : mapping_) {
				size += sizeof(size) + name.length() + data.serializedSize();
			}

			mapping_serialized_.resize(size);
			char* ptr = mapping_serialized_.data();
			for (auto const& [name, data] : mapping_) {
				size = name.length();
				std::memcpy(ptr, &size, sizeof(size));
				ptr += sizeof(size);
				std::memcpy(ptr, name.data(), size);
				ptr += size;
				ptr = data.write(ptr);
			}
		}

		std::uint64_t size = mapping_serialized_.size();
		out.write(reinterpret_cast<char const*>(&size), sizeof(size));
		out.write(mapping_serialized_.data(), mapping_serialized_.size());
		changed_ = false;
	}

	friend std::ostream& operator<<(std::ostream& os, SemanticMapping const& mapping)
	{
		// TODO: Look at
		for (auto const& [str, data] : mapping.mapping_) {
			os << str << ': ' << data.ranges << '\n';
			if (!data.strings.empty()) {
				std::copy(std::cbegin(data.strings), std::prev(std::cend(data.strings)),
				          std::ostream_iterator<std::string>(os, ", "));
				os << *std::prev(std::end(data.strings));
			}
		}
	}

 private:
	struct Data {
		SemanticRangeSet                ranges;
		std::unordered_set<std::string> strings;
		Color                           color;

		void clear()
		{
			ranges.clear();
			strings.clear();
			color.clear();
		}

		std::uint64_t serializedSize()
		{
			std::uint64_t size = sizeof(color);
			size += sizeof(size) + ranges.size() * sizeof(SemanticRangeSet::value_type);
			size += sizeof(size);
			for (auto const& str : strings) {
				size += sizeof(size) + str.length();
			}
			return size;
		}

		char const* read(char const* in)
		{
			// Ranges
			ranges.clear();
			std::uint64_t size;
			std::memcpy(&size, in, sizeof(size));
			in += sizeof(size);
			for (std::uint64_t i = 0; i != size; ++i) {
				Range r;
				std::memcpy(&r, in, sizeof(r));
				in += sizeof(r);
				ranges.insert(r);
			}

			// Strings
			strings.clear();
			std::memcpy(&size, in, sizeof(size));
			in += sizeof(size);
			for (std::uint64_t i = 0; i != size; ++i) {
				std::uint64_t length;
				std::memcpy(&length, in, sizeof(length));
				in += sizeof(length);
				auto str = std::make_unique<char[]>(length);
				std::memcpy(str.get(), in, length);
				in += length;
				strings.emplace(str.get(), length);
			}

			// Color
			std::memcpy(&color, in, sizeof(color));
			in += sizeof(color);

			return in;
		}

		char* write(char* out) const
		{
			// Ranges
			std::uint64_t size = ranges.size();
			std::memcpy(out, &size, sizeof(size));
			out += sizeof(size);
			for (Range r : ranges) {
				std::memcpy(out, &r, sizeof(r));
				out += sizeof(r);
			}

			// Strings
			size = strings.size();
			std::memcpy(out, &size, sizeof(size));
			out += sizeof(size);
			for (auto const& str : strings) {
				size = str.length();
				std::memcpy(out, &size, sizeof(size));
				out += sizeof(size);
				std::memcpy(out, str.data(), size);
				out += size;
			}

			// Color
			std::memcpy(out, &color, sizeof(color));
			out += sizeof(color);

			return out;
		}
	};

 private:
	// Updated by consumer
	std::unordered_map<std::string, Data> consumer_mapping_;
	// Updated by producer
	std::unordered_map<std::string, Data> producer_mapping_;
	// Combined mapping
	std::unordered_map<std::string, Data> mapping_;

	// Producer mapping serialized form
	std::vector<char> producer_serialized_;
	// Combined mapping serialized form
	std::vector<char> mapping_serialized_;

	// Change detection
	bool changed_ = false;
};
}  // namespace ufo

#endif  // UFO_MAP_SEMANTIC_LABEL_MAPPING_HPP