// UFO
#include <ufo/map/ufomap.hpp>
#include <ufo/map/point_cloud.hpp>
#include <ufo/map/predicate/spatial.hpp>
#include <ufo/map/predicate/satisfies.hpp>
#include <ufo/map/types.hpp>

// STL
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

void printUsage(const char* program_name)
{
	std::cout << "Usage: " << program_name << " <input.ufomap> [output.pcd] [options]\n";
	std::cout << "Options:\n";
	std::cout << "  --depth <N>        Extract voxels at depth level N (default: 0 = leaf level)\n";
	std::cout << "  --min-hits <N>     Only include voxels with at least N hits (default: 1)\n";
	std::cout << "  --seen-free-only   Only export voxels marked as not seen-free\n";
	std::cout << "\nExamples:\n";
	std::cout << "  " << program_name << " exist_map.ufomap output.pcd --seen-free-only\n";
}

int main(int argc, char* argv[])
{
	if (argc < 2) {
		printUsage(argv[0]);
		return 1;
	}

	std::filesystem::path input_path(argv[1]);
	std::filesystem::path output_path;
	
	// Default parameters
	ufo::depth_t depth = 0;  // Leaf level
	std::size_t min_hits = 1;
	bool seen_free_only = false;
	bool export_all = false;

	// Parse arguments
	if (argc >= 3 && argv[2][0] != '-') {
		output_path = argv[2];
	} else {
		// Auto-generate output filename
		output_path = input_path;
		output_path.replace_extension(".pcd");
	}

	// Parse options
	for (int i = 2; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--depth" && i + 1 < argc) {
			depth = std::stoi(argv[++i]);
		} else if (arg == "--min-hits" && i + 1 < argc) {
			min_hits = std::stoul(argv[++i]);
		} else if (arg == "--seen-free-only") {
			seen_free_only = true;
		}
	}

	// Check if input file exists
	if (!std::filesystem::exists(input_path)) {
		std::cerr << "[ERROR] Input file does not exist: " << input_path << std::endl;
		return 1;
	}

	std::cout << "[LOG] Reading UFOMap from: " << input_path << std::endl;

    ufo::Map<ufo::MapType::SEEN_FREE | ufo::MapType::REFLECTION | ufo::MapType::LABEL> map(input_path);

    std::cout << "[LOG] Map loaded successfully!" << std::endl;
    std::cout << "      Depth levels: " << +map.depthLevels() << std::endl;
    std::cout << "      Total nodes: " << map.size() << std::endl;
    
    // Debug: Check root node state
    auto root_idx = map.rootIndex();

    // Debug: Count nodes with different predicates
    std::size_t total_leaf = 0, with_hits = 0, seen_free_count = 0, not_seen_free = 0;
    std::size_t total_all_nodes = 0;
    

    // Extract voxel centers as point cloud
    ufo::PointCloudColor cloud;
    std::cout << "[LOG] Extracting voxels at depth " << +depth 
                << " with min_hits >= " << min_hits << std::endl;

    std::size_t count = 0;
    
    if (seen_free_only) {
        // Export only non-seen-free voxels (obstacles)
        std::cout << "[LOG] Mode: Exporting only non-seen-free voxels (obstacles)" << std::endl;
        for (auto node : map.query(ufo::pred::Leaf(depth) && !ufo::pred::SeenFree() && 
                                    ufo::pred::HitsMin(min_hits))) {
            auto center = map.center(node);
            
            // Color by label if available
            auto label = map.label(node.index());
            ufo::Color color;
            if (label > 0) {
                // Generate color from label ID
                ufo::color_t r = static_cast<ufo::color_t>((label * 123) % 255);
                ufo::color_t g = static_cast<ufo::color_t>((label * 456) % 255);
                ufo::color_t b = static_cast<ufo::color_t>((label * 789) % 255);
                color = ufo::Color(r, g, b);
            } else {
                color = ufo::Color(255, 255, 255);
            }
            
            cloud.emplace_back(center, color);
            ++count;
        }
    }

    std::cout << "[LOG] Extracted " << count << " voxel centers" << std::endl;

    if (cloud.empty()) {
        std::cout << "[WARNING] No voxels found matching the criteria!" << std::endl;
        std::cout << "          Try different parameters (--depth, --min-hits, --all)" << std::endl;
        return 0;
    }

    // Save point cloud
    std::cout << "[LOG] Saving point cloud to: " << output_path << std::endl;
    ufo::writePointCloudPCD(output_path, cloud);

    std::cout << "[LOG] Successfully exported " << cloud.size() << " points!" << std::endl;
    std::cout << "[LOG] Done! ^v^" << std::endl;

	return 0;
}
