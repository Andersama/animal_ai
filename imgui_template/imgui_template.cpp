// imgui_template.cpp : Defines the entry point for the application.
//

#include "imgui_template.h"
#include "file-cpp/file.h"

// #include "darknet/darknet.h"
// #include "darknet/yolo_v2_class.hpp"

#include "darknet.h"
#include "yolo_v2_class.hpp"
// #include "C:/Users/ajgam/Desktop/Github/darknet/out/install/x64-Release/include/darknet/darknet.h"
// #include "C:/Users/ajgam/Desktop/Github/darknet/out/install/x64-Release/include/darknet/yolo_v2_class.hpp"

// going to use this for the training thread!
#include <thread>

// IMGUI and glfw includes
#define IMGUI_IMPLEMENTATION
#include <gl/glew.h>
#include <GLFW/glfw3.h>
#include "imgui/imgui.h"
#include "imgui/imgui_stdlib.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
// we need this to change tesselation tolerance
#include "imgui/imgui_internal.h"
// #include "zpp_bits/zpp_bits.h"
#define STB_IMAGE_IMPLEMENTATION
#include "nothings/stb_image.h"

#include "utf8-cpp/source/utf8.h"

#include "ankerl/unordered_dense.h"
#include "yolo_preconfig/yolo7_template.h"

#include "nfd.h"
// #include "nfd.hpp"

#include <filesystem>
#include <chrono>
#include <charconv>

#include <span>
#include <cstdio>

#include "stack_vector/stack_vector.h"
// #include "../../darknet/out/install/x64-Debug/include/darknet/darknet.h"
#if 0
char*   basecfg(char* cfgfile);
network parse_network_cfg(char* filename);
void    load_weights(network* net, char* filename);
int     get_current_batch(network net);
int64_t get_current_iteration(network net);
void    save_weights(network net, char* filename);
float   train_network(network net, data d);

void train_yolo(char* cfgfile, char* weightfile, std::vector<char*>& cxx_paths)
{
	// const char* train_images     = "data/voc/train.txt";
	const char* backup_directory = "backup/";
	srand(time(0));
	char* base = basecfg(cfgfile);
	printf("%s\n", base);
	float   avg_loss = -1;
	network net      = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
	int  imgs = net.batch * net.subdivisions;
	int  i    = *net.seen / imgs;
	data train, buffer;

	layer l = net.layers[net.n - 1];

	int   side    = l.side;
	int   classes = l.classes;
	float jitter  = l.jitter;

	// list* plist = get_paths(train_images);
	//  int N = plist->size;
	// char** paths = (char**)list_to_array(plist);

	load_args args = {0};
	args.w         = net.w;
	args.h         = net.h;
	args.paths     = &cxx_paths[0]; // paths
	args.n         = imgs;
	args.m         = cxx_paths.size(); // plist->size;
	args.classes   = classes;
	args.jitter    = jitter;
	args.num_boxes = side;
	args.d         = &buffer;
	args.type      = REGION_DATA;

	args.angle      = net.angle;
	args.exposure   = net.exposure;
	args.saturation = net.saturation;
	args.hue        = net.hue;

	pthread_t load_thread = load_data_in_thread(args);
	clock_t   time;
	// while(i*imgs < N*120){
	while (get_current_batch(net) < net.max_batches) {
		i += 1;
		time = clock();
		pthread_join(load_thread, 0);
		train       = buffer;
		load_thread = load_data_in_thread(args);

		// printf("Loaded: %lf seconds\n", sec(clock() - time));

		time       = clock();
		float loss = train_network(net, train);
		if (avg_loss < 0)
			avg_loss = loss;
		avg_loss = avg_loss * .9 + loss * .1;

		// printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net),
		//				sec(clock() - time), i * imgs);
		if (i % 1000 == 0 || (i < 1000 && i % 100 == 0)) {
			char buff[256];
			sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
			save_weights(net, buff);
		}
		free_data(train);
	}
	char buff[256];
	sprintf(buff, "%s/%s_final.weights", backup_directory, base);
	save_weights(net, buff);
}
#endif
// #include "fswatch/include/fswatch.hpp"

static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

struct glfw3_setup_t {
	int         err_code;
	GLFWwindow* window;
};

glfw3_setup_t glfw3_setup(uint32_t default_window_width, uint32_t default_window_height, bool fullscreen = false)
{
	// Setup window
	std::cout << "glfwinit\n";
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
		return {1, nullptr};
	std::cout << "glfwinit complete\n";
	// vg::Context svg_ctx;
	//  Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
	// GL ES 2.0 + GLSL 100
	const char* glsl_version = "#version 100";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
	// GL 3.2 + GLSL 150
	const char* glsl_version = "#version 150";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);           // Required on Mac
#else
	// GL 3.0 + GLSL 130
	const char* glsl_version = "#version 130";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	// glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	// glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif
	// GLFWmonitor* monitor = fullscreen ? glfwGetPrimaryMonitor() : NULL;
	GLFWmonitor* monitor = NULL;
	// Create window with graphics context
	GLFWwindow* window = glfwCreateWindow(
					default_window_width, default_window_height, "Dear ImGui GLFW+OpenGL3 example", monitor, NULL);
	if (window == NULL)
		return {1, nullptr};
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1); // Enable vsync

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	(void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
	// io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	// ImGui::StyleColorsLight();

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	return {0, window};
}

// Simple helper function to load an image into a OpenGL texture with common settings
bool LoadTextureFromFile(const char* filename, GLuint* out_texture, int* out_width, int* out_height)
{
	// Load from file
	int            image_width  = 0;
	int            image_height = 0;
	unsigned char* image_data   = stbi_load(filename, &image_width, &image_height, NULL, 4);
	if (image_data == NULL)
		return false;

	// Create a OpenGL texture identifier
	GLuint image_texture;
	glGenTextures(1, &image_texture);
	glBindTexture(GL_TEXTURE_2D, image_texture);

	// Setup filtering parameters for display
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
					GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

	// Upload pixels into texture
#if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
#endif
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data);
	stbi_image_free(image_data);

	*out_texture = image_texture;
	*out_width   = image_width;
	*out_height  = image_height;

	return true;
}

bool ReleaseTexture(GLuint* texture)
{
	glDeleteTextures(1, texture);
	return true;
}

void label_path_old(std::string& out, std::string& buf, const std::string& image_path)
{
	out.clear();
	buf.clear();

	buf.assign(image_path.data(), image_path.size());
	buf.replace(buf.find(std::string_view{"/images/train2017/"}), std::string_view{"/images/train2017/"}.size(),
					std::string_view{"/labels/train2017/"});
	buf.replace(buf.find(std::string_view{"/images/val2017/"}), std::string_view{"/images/val2017/"}.size(),
					std::string_view{"/labels/train2017/"});

	buf.replace(buf.find(std::string_view{"\\images\\train2017\\"}), std::string_view{"\\images\\train2017\\"}.size(),
					std::string_view{"\\labels\\train2017/"});
	buf.replace(buf.find(std::string_view{"\\images\\val2017\\"}), std::string_view{"\\images\\val2017\\"}.size(),
					std::string_view{"\\labels\\train2017\\"});

	buf.replace(buf.find(std::string_view{"/images/train2014/"}), std::string_view{"/images/train2014/"}.size(),
					std::string_view{"/labels/train2014/"});
	buf.replace(buf.find(std::string_view{"/images/val2014/"}), std::string_view{"/images/val2014/"}.size(),
					std::string_view{"/labels/train2014/"});

	buf.replace(buf.find(std::string_view{"\\images\\train2014\\"}), std::string_view{"\\images\\train2014\\"}.size(),
					std::string_view{"\\labels\\train2014/"});
	buf.replace(buf.find(std::string_view{"\\images\\val2014\\"}), std::string_view{"\\images\\val2014\\"}.size(),
					std::string_view{"\\labels\\train2014\\"});

	buf.replace(buf.find(std::string_view{"/JPEGImages/"}), std::string_view{"/JPEGImages/"}.size(),
					std::string_view{"/labels/"});
	buf.replace(buf.find(std::string_view{"\\JPEGImages\\"}), std::string_view{"\\JPEGImages\\"}.size(),
					std::string_view{"\\labels\\"});

	fmt::format_to(std::back_inserter(out), "{}.txt", util::utf8::filename(buf));
}

//
void label_path(std::string& out, std::string& buf, const std::string& image_path)
{
	out.clear();

	buf.clear();
	buf.assign(image_path.data(), image_path.size());

	fmt::format_to(std::back_inserter(out), "{}_lbl", buf);
	// find_replace(input_path, "/images/train2017/", "/labels/train2017/", output_path);        // COCO
	// find_replace(output_path, "/images/val2017/", "/labels/val2017/", output_path);           // COCO
	// find_replace(output_path, "/JPEGImages/", "/labels/", output_path);                       // PascalVOC
	// find_replace(output_path, "\\images\\train2017\\", "\\labels\\train2017\\", output_path); // COCO
	// find_replace(output_path, "\\images\\val2017\\", "\\labels\\val2017\\", output_path);     // COCO
	//
	// find_replace(output_path, "\\images\\train2014\\", "\\labels\\train2014\\", output_path); // COCO
	// find_replace(output_path, "\\images\\val2014\\", "\\labels\\val2014\\", output_path);     // COCO
	// find_replace(output_path, "/images/train2014/", "/labels/train2014/", output_path);       // COCO
	// find_replace(output_path, "/images/val2014/", "/labels/val2014/", output_path);           // COCO
	//
	// find_replace(output_path, "\\JPEGImages\\", "\\labels\\", output_path); // PascalVOC
	//  find_replace(output_path, "/images/", "/labels/", output_path);    // COCO
	//  find_replace(output_path, "/VOC2007/JPEGImages/", "/VOC2007/labels/", output_path);        // PascalVOC
	//  find_replace(output_path, "/VOC2012/JPEGImages/", "/VOC2012/labels/", output_path);        // PascalVOC

	// find_replace(output_path, "/raw/", "/labels/", output_path);
	// trim(output_path);

	// replace only ext of files
	// find_replace_extension(output_path, ".jpg", ".jpg_lbl", output_path);
	// find_replace_extension(output_path, ".JPG", ".JPG_lbl", output_path); // error
	// find_replace_extension(output_path, ".jpeg", ".jpeg_lbl", output_path);
	// find_replace_extension(output_path, ".JPEG", ".JPEG_lbl", output_path);
	// find_replace_extension(output_path, ".png", ".png_lbl", output_path);
	// find_replace_extension(output_path, ".PNG", ".PNG_lbl", output_path);
	// find_replace_extension(output_path, ".bmp", ".bmp_lbl", output_path);
	// find_replace_extension(output_path, ".BMP", ".BMP_lbl", output_path);
	// find_replace_extension(output_path, ".ppm", ".ppm_lbl", output_path);
	// find_replace_extension(output_path, ".PPM", ".PPM_lbl", output_path);
	// find_replace_extension(output_path, ".tiff", ".tiff_lbl", output_path);
	// find_replace_extension(output_path, ".TIFF", ".TIFF_lbl", output_path);

	// Check file ends with _lbl:
	// if (strlen(output_path) > 4) {
	//	char* output_path_ext = output_path + strlen(output_path) - 4;
	//	if (strcmp("_lbl", output_path_ext) != 0) {
	//		fprintf(stderr, "Failed to infer label file name (check image extension is supported): %s \n", output_path);
	//	}
	//} else {
	//	fprintf(stderr, "Label file name is too short: %s \n", output_path);
	//}
}

// improve on this
void strings_from_file(std::vector<std::string>& strings, nfdu8char_t* filename)
{
	std::ifstream ifile(filename);
	if (!ifile.is_open())
		return;

	strings.clear();
	for (std::string line; std::getline(ifile, line);)
		strings.push_back(line);

	ifile.close();
	return;
}

struct labelled_bounding_box {
	int object_class;
	// within 0.0f to 1.0f
	float x_center;
	float y_center;
	float width;
	float height;
};

struct simple_bounding_box {
	// within 0.0f to 1.0f
	float x_center;
	float y_center;
	float width;
	float height;
	//
	std::string label;
	//
	float prob;

	std::errc from(std::string_view line)
	{
		size_t                 f0 = line.find_first_not_of("0123456789.e-", 0);
		std::from_chars_result r0 = std::from_chars(line.data(), line.data() + f0, x_center);
		if (r0.ec != std::errc{})
			return r0.ec;
		line.remove_prefix(f0);
		line.remove_prefix(line.find_first_not_of(" \t"));

		f0 = line.find_first_not_of("0123456789.e-", 0);
		r0 = std::from_chars(line.data(), line.data() + f0, y_center);
		if (r0.ec != std::errc{})
			return r0.ec;
		line.remove_prefix(f0);
		line.remove_prefix(line.find_first_not_of(" \t"));

		f0 = line.find_first_not_of("0123456789.e-", 0);
		r0 = std::from_chars(line.data(), line.data() + f0, width);
		if (r0.ec != std::errc{})
			return r0.ec;
		line.remove_prefix(f0);
		line.remove_prefix(line.find_first_not_of(" \t"));

		f0 = line.find_first_not_of("0123456789.e-", 0);
		r0 = std::from_chars(line.data(), line.data() + f0, height);
		if (r0.ec != std::errc{})
			return r0.ec;
		line.remove_prefix(f0);
		line.remove_prefix(line.find_first_not_of(" \t"));

		std::string_view end_label = line;
		// end_label.remove_prefix(end_label.find_first_not_of(" \t"));
		label = end_label;
		return std::errc{};
	}

	std::errc from_old(std::string_view line)
	{
		size_t id0 = line.find_first_not_of("0123456789+-", 0);
		label.clear();
		label.assign(line.data(), id0);

		line.remove_prefix(id0);
		line.remove_prefix(line.find_first_not_of(" \t"));

		size_t                 f0 = line.find_first_not_of("0123456789.e-", 0);
		std::from_chars_result r0 = std::from_chars(line.data(), line.data() + f0, x_center);
		if (r0.ec != std::errc{})
			return r0.ec;
		line.remove_prefix(f0);
		line.remove_prefix(line.find_first_not_of(" \t"));

		f0 = line.find_first_not_of("0123456789.e-", 0);
		r0 = std::from_chars(line.data(), line.data() + f0, y_center);
		if (r0.ec != std::errc{})
			return r0.ec;
		line.remove_prefix(f0);
		line.remove_prefix(line.find_first_not_of(" \t"));

		f0 = line.find_first_not_of("0123456789.e-", 0);
		r0 = std::from_chars(line.data(), line.data() + f0, width);
		if (r0.ec != std::errc{})
			return r0.ec;
		line.remove_prefix(f0);
		line.remove_prefix(line.find_first_not_of(" \t"));

		f0 = line.find_first_not_of("0123456789.e-", 0);
		r0 = std::from_chars(line.data(), line.data() + f0, height);
		if (r0.ec != std::errc{})
			return r0.ec;
		line.remove_prefix(f0);
		line.remove_prefix(line.find_first_not_of(" \t"));

		return std::errc{};
	}

	std::errc from(bbox_t& box)
	{
		x_center = box.x;
		y_center = box.y;
		width    = box.w;
		height   = box.h;
		return std::errc{};
	}

	std::errc from(simple_bounding_box& box)
	{
		x_center = box.x_center;
		y_center = box.y_center;
		width    = box.width;
		height   = box.height;
		label.clear();
		label.assign(box.label.data(), box.label.size());
		return std::errc{};
	}
	#if 0
	static std::errc from_file(std::vector<simple_bounding_box>& boxes, std::string_view in, std::string& tmp_buf)
	{
		std::fstream ifile_stream;
		ifile_stream.open(in, std::ios::in);
		if (ifile_stream.is_open()) {
			for (; std::getline(ifile_stream, tmp_buf);) {
				//
			}
			ifile_stream.close();
		}
	}

	static std::errc from_file_old(std::vector<simple_bounding_box>& boxes, std::string_view in, std::string& tmp_buf)
	{
		std::fstream ifile_stream;
		ifile_stream.open(in, std::ios::in);
		if (ifile_stream.is_open()) {
			for (; std::getline(ifile_stream, tmp_buf);) {
				//
			}
			ifile_stream.close();
		}
	}

	bool write_old(std::string_view out)
	{
		std::fstream ofile_stream;
		ofile_stream.open(out, std::ios::out | std::ios::trunc);
		if (ofile_stream.is_open()) {
			for (; std::getline(ifile_stream, tmp_line);) {
			}
			ofile_stream.close();
		}
	}

	bool write(std::string_view out)
	{
	}
	#endif
};

struct path_data {
	std::string                  path      = {};
	std::string_view             parent    = {};
	std::string_view             child     = {};
	std::string_view             extension = {};
	std::filesystem::file_status status    = {};

	void from(std::string_view path_view, std::filesystem::file_status path_status = {})
	{
		path = path_view;

		parent = util::utf8::parent_path(path);

		child = std::string_view{path.data() + parent.size(), path.size() - parent.size()};

		// remove slashes
		child.remove_prefix(std::min(child.find_first_not_of("/\\"), child.size()));

		extension = util::utf8::extension(path);

		status = path_status;
	}

	void from(std::string&& path_str, std::filesystem::file_status path_status = {})
	{
		path = path_str;

		parent = util::utf8::parent_path(path);

		child = std::string_view{path.data() + parent.size(), path.size() - parent.size()};

		// remove slashes
		child.remove_prefix(std::min(child.find_first_not_of("/\\"), child.size()));

		extension = util::utf8::extension(path);

		status = path_status;
	}
	/*
	path_data& operator=(const path_data &other) {
		path = other.path;
		parent = other.parent;
		child  = other.child;
		extension = other.extension;
		status    = other.status;
		return *this;
	}
	path_data& operator=(path_data&& other)
	{
		path      = other.path;
		parent    = other.parent;
		child     = other.child;
		extension = other.extension;
		status    = other.status;
		return *this;
	}
	*/
};

struct nonrecursive_directory_data {
	// std::string parent;
	std::vector<path_data> paths;
	size_t                 last_update = 0ull;
	path_data              current;
	// std::chrono::steady_clock::time_point last_update;

	// minute delay default
	bool should_update_now(size_t                         update_rate_in_ns = 60000000000ul,
					std::chrono::steady_clock::time_point current_time      = std::chrono::steady_clock::now())
	{
		size_t current_ns = current_time.time_since_epoch().count();
		return (current_ns - last_update) >= update_rate_in_ns;
	}

	void update()
	{
		std::error_code                     iterator_error = {};
		std::filesystem::directory_iterator parent_iterator{current.path, iterator_error};

		if (iterator_error == std::error_code{}) {
			paths.clear();
			for (const std::filesystem::directory_entry& entry : parent_iterator) {
				std::error_code status_error = {};
				// std::filesystem::file_status entry_status = std::filesystem::status(entry.path(), status_error);
				std::filesystem::file_status entry_status = entry.status(status_error);
				if (status_error == std::error_code{}) {
					path_data& data = paths.emplace_back();
					data.from(entry.path().string(), entry_status);
				}
			}
			// keep the relative order
			std::stable_partition(paths.begin(), paths.end(),
							[](path_data& lhs) { return std::filesystem::is_directory(lhs.status); });

			last_update = std::chrono::steady_clock::now().time_since_epoch().count();
		}
	}

	path_data& operator[](size_t idx)
	{
		return paths[idx];
	}
	const path_data& operator[](size_t idx) const
	{
		return paths[idx];
	}

	path_data& get_child(size_t idx)
	{
		return operator[](idx);
	}

	const path_data& get_child(size_t idx) const
	{
		return operator[](idx);
	}
};

size_t linuxlike_directory_view(nonrecursive_directory_data& folder, size_t selected_file = ~0ull)
{
	// update every minute
	if (folder.should_update_now(60000000000ul)) {
		folder.update();
	}

	size_t selected = ~size_t{0};

	bool             change_directory = false;
	std::string_view next_directory   = folder.current.path;

	// ImGui::SetNextItemOpen(true, ImGuiCond_::ImGuiCond_Once);
	// if (ImGui::CollapsingHeader("Folder")) {
	//  if used recursively we'd need to do this anyway
	ImGui::PushID(folder.current.path.c_str(), folder.current.path.c_str() + folder.current.path.size());
	// use ImGuiTreeNodeFlags_NoTreePushOnOpen so we don't need to call ImGui::TreePop()
	bool parent_node_open =
					ImGui::TreeNodeEx("..", ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_NoTreePushOnOpen);

	if (ImGui::IsItemClicked()) {
		next_directory   = util::utf8::parent_path(folder.current.path);
		change_directory = true;
	}

	for (size_t i = 0; i < folder.paths.size(); i++) {
		if (std::filesystem::is_directory(folder.paths[i].status)) {
			bool tree_open = ImGui::TreeNodeEx(folder.paths[i].child.data(),
							ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_NoTreePushOnOpen |
											((i == selected_file) * ImGuiTreeNodeFlags_Selected));

			if (ImGui::IsItemClicked()) {
				next_directory   = folder[i].path;
				change_directory = true;
			}

		} else {
			bool tree_open = ImGui::TreeNodeEx(folder.paths[i].child.data(),
							ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_Leaf |
											ImGuiTreeNodeFlags_SpanFullWidth |
											((i == selected_file) * ImGuiTreeNodeFlags_Selected));
			// handle clicking a file name
			if (ImGui::IsItemClicked()) {
				selected = i;
			}
		}
	}

	ImGui::PopID();
	// ImGui::EndChild();
	//}
	// we have to wait for the next loop to update
	if (change_directory) {
		folder.current.from(next_directory);
		// folder.parent = next_directory;
		// we can do this by resetting the time
		folder.last_update = 0ull;
	}

	return selected;
}

struct recursive_directory_data {
	std::vector<std::unique_ptr<recursive_directory_data>> children;
	size_t                                                 last_update = 0ull;
	path_data                                              current;

	// don't necessarily need this, can do upates on clicks
	bool should_update_now(size_t                         update_rate_in_ns = 60000000000ul,
					std::chrono::steady_clock::time_point current_time      = std::chrono::steady_clock::now())
	{
		size_t current_ns = current_time.time_since_epoch().count();
		return (current_ns - last_update) >= update_rate_in_ns;
	}

	void update()
	{
		std::error_code                     iterator_error = {};
		std::filesystem::directory_iterator parent_iterator{current.parent, iterator_error};

		if (iterator_error == std::error_code{}) {
			children.clear();
			for (const std::filesystem::directory_entry& entry : parent_iterator) {
				std::error_code status_error = {};
				// std::filesystem::file_status entry_status = std::filesystem::status(entry.path(), status_error);
				std::filesystem::file_status entry_status = entry.status(status_error);
				if (status_error == std::error_code{}) {
					// path_data& data = paths.emplace_back();
					std::unique_ptr<recursive_directory_data>& ptr = children.emplace_back();
					// ptr = std::make_unique<recursive_directory_data>();
					path_data& data = ptr->current;
					data.from(entry.path().string(), entry_status);
				}
			}
			// keep the relative order
			std::stable_partition(children.begin(), children.end(), [](std::unique_ptr<recursive_directory_data>& lhs) {
				return std::filesystem::is_directory(lhs->current.status);
			});

			last_update = std::chrono::steady_clock::now().time_since_epoch().count();
		}
	}

	path_data& operator[](size_t idx)
	{
		return children[idx]->current;
	}
	const path_data& operator[](size_t idx) const
	{
		return children[idx]->current;
	}

	path_data& get_child(size_t idx)
	{
		return operator[](idx);
	}

	const path_data& get_child(size_t idx) const
	{
		return operator[](idx);
	}
};

struct translate_scale2d {
	ImVec2 translate;
	ImVec2 scale;
};

struct image_data {
	path_data   file;
	int         width;
	int         height;
	ImTextureID texture_id;

	std::vector<simple_bounding_box> manual_bounding_boxes;
	std::vector<simple_bounding_box> detect_bounding_boxes;
	std::vector<bbox_t>              detect_bounding_boxes_tmp;
	// TODO: trained bounding boxes

	translate_scale2d transform;

	bool texture_loaded = false;
	// TODO: make into flags
	bool dirty      = false;
	bool show       = true;
	bool first_show = true;
	bool detected   = false;
};

void generate_yolov7_template(std::string& buf, yolo_template_opts opts, size_t training_images_count)
{
	size_t max_batches   = std::max(std::max(opts.classes * 2000ull, 6000ull), training_images_count);
	size_t line_steps_80 = (max_batches * 8) / 10;
	size_t line_steps_90 = (max_batches * 9) / 10;
	size_t filters       = (opts.classes + 5) * 3;

	// yolov7_template (tiny)
	fmt::format_to(std::back_inserter(buf), yolov7_tiny_template, fmt::arg("width", opts.width),
					fmt::arg("height", opts.height), fmt::arg("batch", opts.batch),
					fmt::arg("subdivisions", opts.subdivisions), fmt::arg("channels", opts.channels),
					fmt::arg("max_batches", max_batches), fmt::arg("line_steps_80", line_steps_80),
					fmt::arg("line_steps_90", line_steps_90), fmt::arg("filters", filters),
					fmt::arg("classes", opts.classes), fmt::arg("mixup", opts.mixup));
}

void generate_yolov7_large_template(std::string& buf, yolo_template_opts opts, size_t training_images_count)
{
	size_t max_batches   = std::max(std::max(opts.classes * 2000ull, 6000ull), training_images_count);
	size_t line_steps_80 = (max_batches * 8) / 10;
	size_t line_steps_90 = (max_batches * 9) / 10;
	size_t filters       = (opts.classes + 5) * 3;

	// yolov7_template (large)
	fmt::format_to(std::back_inserter(buf), yolov7_template, fmt::arg("width", opts.width),
					fmt::arg("height", opts.height), fmt::arg("batch", opts.batch),
					fmt::arg("subdivisions", opts.subdivisions), fmt::arg("channels", opts.channels),
					fmt::arg("max_batches", max_batches), fmt::arg("line_steps_80", line_steps_80),
					fmt::arg("line_steps_90", line_steps_90), fmt::arg("filters", filters),
					fmt::arg("classes", opts.classes), fmt::arg("mixup", opts.mixup));
}

bool save_cstrings(const char* out, std::span<char*> cstrs)
{
	std::fstream labelfile;
	labelfile.open(out, std::ios::out | std::ios::binary);
	if (!labelfile.is_open())
		return false;

	char                                buf[4096];
	std::pmr::monotonic_buffer_resource mbr{buf, sizeof(buf)};
	std::pmr::string                    tmp_buffer(&mbr);
	tmp_buffer.reserve(sizeof(buf));

	for (size_t i = 0; i < cstrs.size(); i++) {
		std::string_view vw = cstrs[i];
		if ((tmp_buffer.size() + vw.size()) > tmp_buffer.capacity()) {
			if (tmp_buffer.size()) {
				labelfile.write(tmp_buffer.data(), tmp_buffer.size());
				tmp_buffer.clear();
			}

			// just write the thing out move on to the next
			labelfile.write(vw.data(), vw.size());
			labelfile.write("\n", 1);
			continue;
		}

		tmp_buffer.append(vw.data(), vw.size());
		tmp_buffer.append("\n", 1);
	}

	if (tmp_buffer.size()) {
		labelfile.write(tmp_buffer.data(), tmp_buffer.size());
		tmp_buffer.clear();
	}

	labelfile.close();
}

void write_yolo7_cfg_file(
				nfdu8char_t* cfg_out_filename, std::string& buf, yolo_template_opts opts, size_t training_images_count)
{
	buf.clear();
	// buf.reserve(yolov7_template.size() + 32 * 16);
	generate_yolov7_template(buf, opts, training_images_count);

	// write the file
	std::fstream cfgfile;
	cfgfile.open(cfg_out_filename, std::ios::out);
	if (!cfgfile.is_open())
		return;
	cfgfile.write(buf.data(), buf.size());
	cfgfile.close();
}

void write_yolo_data_file(nfdu8char_t* datafile, std::string& buf, std::string_view train_txt,
				std::span<char*> train_paths, std::string_view valid_txt, std::span<char*> valid_paths,
				std::string_view names_path, std::span<char*> labels, std::string_view weights_folder)
{
	// write out the job data...in the event we fail
	save_cstrings(train_txt.data(), train_paths);
	save_cstrings(valid_txt.data(), valid_paths);
	save_cstrings(names_path.data(), labels);

	buf.clear();
	fmt::format_to(std::back_inserter(buf), "classses = {}\ntrain = {}\nvalid = {}\nnames = {}\nbackup = {}",
					labels.size(), train_txt, valid_txt, names_path, weights_folder);

	std::fstream cfgfile;
	cfgfile.open(datafile, std::ios::out);
	if (!cfgfile.is_open())
		return;
	cfgfile.write(buf.data(), buf.size());
	cfgfile.close();
}

void generate_yolo_obj_data_file(std::string& buf, std::string_view data_set_name, size_t classes)
{
	fmt::format_to(std::back_inserter(buf),
					R"(classes = {}
train  = data/{data_set_name}_train.txt
valid  = data/{data_set_name}_test.txt
names = data/{data_set_name}_obj.names
backup = backup/
)",
					classes, fmt::arg("data_set_name", data_set_name));
}

void train_ai(char* cfgfile, char* weightfile)
{
	// network* net = load_network(cfgfile, weightfile, 0);
	// data train;
	// data buffer;
	// float loss;
	// load_data_in_thread
	char* datacfg;
	// char* cfgfile;
	// char* weightfile;
	int*  gpus;
	int   ngpus;
	int   clear;
	int   dont_show;
	int   calc_map;
	float thresh     = .25;
	float iou_thresh = .5;
	int   mjpeg_port;
	int   show_imgs;
	int   benchmark_layers;
	char* chart_path;
	train_detector(datacfg, cfgfile, weightfile, gpus, ngpus, clear, dont_show, calc_map, thresh, iou_thresh,
					mjpeg_port, show_imgs, benchmark_layers, chart_path);
}

void open_directory(nonrecursive_directory_data& label_folder, std::string_view working_directory)
{
	nfdu8char_t* out    = nullptr;
	nfdresult_t  result = NFD_PickFolderU8(&out, working_directory.data());

	if (result == NFD_OKAY) {
		std::error_code              status_error_code = {};
		std::filesystem::file_status folder_status     = std::filesystem::status(out, status_error_code);
		if (status_error_code == std::error_code{}) {
			label_folder.current.from(std::string_view{out}, folder_status);
			label_folder.update();
		}
		free(out);
	}
}

void open_labels(std::vector<std::string>& labels)
{
	nfdu8char_t* out    = nullptr;
	nfdresult_t  result = NFD_OpenDialogU8(&out, nullptr, 0, nullptr);

	if (result == NFD_OKAY) {
		strings_from_file(labels, out);
		free(out);
	}
#if 0
	else if (result == NFD_CANCEL) {
		// fmt::print("{}", "User pressed cancel");
		// puts("User pressed cancel.");
	} else {
		// fmt::print("Error: {}\n", NFD_GetError());
	}
#endif
};

bool save_labels(std::vector<std::string>& labels)
{
	nfdu8char_t* out    = nullptr;
	nfdresult_t  result = NFD_SaveDialogU8(&out, nullptr, 0, nullptr, nullptr);

	if (result == NFD_OKAY) {
		// zpp_out(labels_map);
		std::fstream labelfile;
		labelfile.open(out, std::ios::out | std::ios::binary);
		if (!labelfile.is_open())
			return false;

		char                                buf[4096];
		std::pmr::monotonic_buffer_resource mbr{buf, sizeof(buf)};
		std::pmr::string                    tmp_buffer(&mbr);
		tmp_buffer.reserve(sizeof(buf));

		for (size_t i = 0; i < labels.size(); i++) {
			if ((tmp_buffer.size() + labels[i].size()) > tmp_buffer.capacity()) {
				if (tmp_buffer.size()) {
					labelfile.write(tmp_buffer.data(), tmp_buffer.size());
					tmp_buffer.clear();
				}

				// just write the thing out move on to the next
				labelfile.write(labels[i].data(), labels[i].size());
				labelfile.write("\n", 1);
				continue;
			}

			tmp_buffer.append(labels[i].data(), labels[i].size());
			tmp_buffer.append("\n", 1);
			// tmp_buffer.append();
			/*
			labelfile.write(labels[i].data(), labels[i].size());
			labelfile.write("\n", 1);
			*/
		}

		if (tmp_buffer.size()) {
			labelfile.write(tmp_buffer.data(), tmp_buffer.size());
			tmp_buffer.clear();
		}

		labelfile.close();
		free(out);

		return true;
	}

	return false;
};

bool save_bounding_boxes_to_buffer(std::string& buf, std::vector<simple_bounding_box>& boxes)
{
	buf.clear();
	for (size_t box_idx = 0; box_idx < boxes.size(); box_idx++) {
		simple_bounding_box& current_box = boxes[box_idx];
		fmt::format_to(std::back_inserter(buf), "{} {} {} {} {}\n", current_box.x_center, current_box.y_center,
						current_box.width, current_box.height, current_box.label);
	}
}

bool save_bounding_boxes(std::string& buf, image_data& image)
{
	// temporarily use buffer to write out the target filepath
	buf.clear();
	fmt::format_to(std::back_inserter(buf), "{}_lbl", image.file.path);

	std::fstream boxfile;
	boxfile.open(buf.c_str(), std::ios::out);
	if (!boxfile.is_open()) // failed to write
		return false;

	save_bounding_boxes_to_buffer(buf, image.manual_bounding_boxes);
	boxfile.write(buf.data(), buf.size());
	boxfile.close();

	return true;
}

bool is_string_in(std::string& str, std::span<std::string_view> strs)
{
	for (size_t i = 0; i < strs.size(); i++) {
		if (str.compare(strs[i]) == 0) {
			return true;
		}
	}
	return false;
}

void lowercase_str(std::string& buf)
{
	for (size_t i = 0; i < buf.size(); i++)
		buf[i] = buf[i] <= 0x7f ? std::tolower(buf[i]) : buf[i]; // gaurd against manipulating values that aren't ascii
}

constexpr size_t lru_size = 64;
void lazy_load_image(std::vector<image_data>& open_images, stack_vector::stack_vector<size_t, lru_size>& lru_idxs,
				size_t idx)
{
	if (idx < open_images.size()) {
		auto idx_it = std::find(lru_idxs.begin(), lru_idxs.end(), idx);
		if (idx_it != lru_idxs.end()) {
			for (size_t i = std::distance(lru_idxs.begin(), idx_it); i < (lru_idxs.size() - 1); i++) {
				lru_idxs[i] = lru_idxs[i + 1];
			}
			lru_idxs[lru_idxs.size() - 1] = idx;
		} else {
			if (lru_idxs.size() < lru_idxs.capacity()) {
				lru_idxs.emplace_back(idx);
			} else {
				// eject oldest
				size_t oldest_idx = lru_idxs[0];
				for (size_t i = 0; i < (lru_idxs.size() - 1); i++) {
					lru_idxs[i] = lru_idxs[i + 1];
				}
				image_data& old_image = open_images[oldest_idx];
				ReleaseTexture((GLuint*)&(old_image.texture_id));
				old_image.texture_loaded = false;
			}
		}
		// load "newest" if needed
		image_data& new_image = open_images[idx];
		if (!new_image.texture_loaded) {
			new_image.texture_loaded = LoadTextureFromFile(new_image.file.path.c_str(),
							(GLuint*)&(new_image.texture_id), &(new_image.width), &(new_image.height));
		}
	}
}

void open_image_unchecked(nonrecursive_directory_data& label_folder, std::vector<image_data>& open_images,
				std::string& lowercase_extension, std::string& label_file_path,
				std::span<std::string_view> image_extensions, std::vector<std::string>& known_labels,
				size_t selected_file)
{
	path_data& selected_path = label_folder[selected_file];

	if (!std::filesystem::is_regular_file(selected_path.status)) {
		return;
	}

	lowercase_extension.clear();
	lowercase_extension.append(selected_path.extension);
	lowercase_str(lowercase_extension);

	for (size_t i = 0; i < image_extensions.size(); i++) {
		if (lowercase_extension.compare(image_extensions[i]) == 0) {
			// clicked on what appears to be an image file

			auto it = std::find_if(open_images.begin(), open_images.end(), [&selected_path](const image_data& image) {
				return image.file.path.compare(selected_path.path) == 0;
			});

			image_data* selected_image = nullptr;
			if (it != open_images.end()) {
				selected_image       = &(*it);
				selected_image->show = true;
				size_t idx           = std::distance(open_images.begin(), it);

			} else {
				size_t idx     = open_images.size();
				selected_image = &(open_images.emplace_back());
				// label_folder[selected_file]
				//*selected_image      = ;
				selected_image->file = selected_path;
				// load the image for the first time
				// selected_image->texture_loaded = LoadTextureFromFile(selected_path.path.c_str(),
				// (GLuint*)&(selected_image->texture_id), &(selected_image->width), &(selected_image->height));

				// selected_image->transform.translate = ImVec2{-(float)selected_image->width, 0.0f};
				// selected_image->transform.scale     = ImVec2{1.0f, 1.0f};

				// open at top left corner
				selected_image->transform.translate = ImVec2{0.0f, 0.0f};
				selected_image->transform.scale     = ImVec2{1.0f, 1.0f};
				selected_image->show                = true;
				selected_image->first_show          = true;

				// std::vector<simple_bounding_box>

				label_file_path.clear();
				fmt::format_to(std::back_inserter(label_file_path), "{}_lbl", selected_path.path);
				if (std::filesystem::exists(label_file_path)) {
					std::ifstream ifile(label_file_path);
					if (!ifile.is_open())
						break;

					selected_image->manual_bounding_boxes.clear();
					size_t box_idx = 0;
					for (std::string line; std::getline(ifile, line);) {
						selected_image->manual_bounding_boxes.emplace_back();
						// only keep bounding boxes we can parse correctly
						std::errc ec = selected_image->manual_bounding_boxes[box_idx].from(line);
						box_idx += ec == std::errc{};
					}
					// remove unnecessary
					selected_image->manual_bounding_boxes.erase(selected_image->manual_bounding_boxes.begin() + box_idx,
									selected_image->manual_bounding_boxes.begin() +
													selected_image->manual_bounding_boxes.size());
					ifile.close();

					for (size_t box_idx = 0; box_idx < selected_image->manual_bounding_boxes.size(); box_idx++) {
						simple_bounding_box& box           = selected_image->manual_bounding_boxes[box_idx];
						bool                 label_matched = false;
						for (const std::string& label : known_labels) {
							if (box.label.compare(label) == 0) {
								label_matched = true;
								break;
							}
						}

						if (!label_matched) {
							known_labels.emplace_back(box.label);
						}

						// skip matching consecutive labels
						for (size_t tmp = box_idx + 1; tmp < selected_image->manual_bounding_boxes.size(); tmp++) {
							simple_bounding_box& tmp_box = selected_image->manual_bounding_boxes[tmp];
							if (tmp_box.label.compare(box.label) == 0) {
								box_idx = tmp;
							} else {
								break;
							}
						}
					}
				}
			}

			break;
		}
	}
}

struct label_mapping {
	std::string              from;
	std::vector<std::string> to;
};

void string_from(std::string& buf, const std::filesystem::path& path)
{
	// buf.clear();

	if constexpr (std::is_same_v<std::filesystem::path::string_type, std::string>) {
		// buf.insert(buf.end(), path.native().data(), path.native().data() + path.native().size());
		// char_star(buf, path.native());

		buf.assign(path.native().data(), path.native().data() + path.native().size());
	} else if (std::is_same_v<std::filesystem::path::string_type, std::wstring>) {
		// buf = path.string();//char_star(buf, path.native());
#if _WIN32
		buf.clear();
		utf8::unchecked::utf16to8(
						path.native().data(), path.native().data() + path.native().size(), std::back_inserter(buf));
#else
		buf = path.string();
#endif
	} else if (std::is_same_v<std::filesystem::path::string_type, std::u8string>) {
		buf.assign(path.native().data(), path.native().data() + path.native().size());
	} else if (std::is_same_v<std::filesystem::path::string_type, std::u16string>) {
		buf.clear();
		utf8::unchecked::utf16to8(
						path.native().data(), path.native().data() + path.native().size(), std::back_inserter(buf));
	} else if (std::is_same_v<std::filesystem::path::string_type, std::u32string>) {
		buf.clear();
		utf8::unchecked::utf32to8(
						path.native().data(), path.native().data() + path.native().size(), std::back_inserter(buf));
	} else {
		buf = path.string();
	}
}

int util_max_index(float* a, int n)
{
	if (n <= 0)
		return -1;
	int   i, max_i = 0;
	float max = a[0];
	for (i = 1; i < n; ++i) {
		if (a[i] > max) {
			max   = a[i];
			max_i = i;
		}
	}
	return max_i;
}

image util_load_image_stb(char* filename, int channels)
{
	int            w, h, c;
	unsigned char* data = stbi_load(filename, &w, &h, &c, channels);
	if (!data)
		throw std::runtime_error("file not found");
	if (channels)
		c = channels;
	int   i, j, k;
	image im = make_image(w, h, c);
	for (k = 0; k < c; ++k) {
		for (j = 0; j < h; ++j) {
			for (i = 0; i < w; ++i) {
				int dst_index      = i + w * j + w * h * k;
				int src_index      = k + c * i + c * w * j;
				im.data[dst_index] = (float)data[src_index] / 255.;
			}
		}
	}
	free(data);
	return im;
}

void util_load_image(image_t& img, const std::string& image_filename)
{
	char* input = const_cast<char*>(image_filename.c_str());
	image im    = util_load_image_stb(input, 3);

	img.c    = im.c;
	img.data = im.data;
	img.h    = im.h;
	img.w    = im.w;
}

void detect(std::vector<bbox_t>& bbox_vec, network& net, image_t img, float thresh, bool use_mean, int gpu_id = 0)
{
	// detector_gpu_t& detector_gpu = *static_cast<detector_gpu_t*>(detector_gpu_ptr.get());
	// network&        net          = detector_gpu.net;
	// #ifdef GPU
	//	int old_gpu_index;
	//	cudaGetDevice(&old_gpu_index);
	//	if (gpu_id != old_gpu_index)
	//		cudaSetDevice(gpu_id);
	//
	//		// net.wait_stream = wait_stream; // 1 - wait CUDA-stream, 0 - not to wait
	// #endif
	//  std::cout << "net.gpu_index = " << net.gpu_index << std::endl;

	image im;
	im.c    = img.c;
	im.data = img.data;
	im.h    = img.h;
	im.w    = img.w;

	image sized;

	if (net.w == im.w && net.h == im.h) {
		sized = make_image(im.w, im.h, im.c);
		memcpy(sized.data, im.data, im.w * im.h * im.c * sizeof(float));
	} else {
		sized = resize_image(im, net.w, net.h);
	}

	layer l = net.layers[net.n - 1];

	float* X = sized.data;

	float* prediction = network_predict(net, X);

	// if (use_mean) {
	//	memcpy(detector_gpu.predictions[detector_gpu.demo_index], prediction, l.outputs * sizeof(float));
	//	mean_arrays(detector_gpu.predictions, NFRAMES, l.outputs, detector_gpu.avg);
	//	l.output                = detector_gpu.avg;
	//	detector_gpu.demo_index = (detector_gpu.demo_index + 1) % NFRAMES;
	// }
	//  get_region_boxes(l, 1, 1, thresh, detector_gpu.probs, detector_gpu.boxes, 0, 0);
	//  if (nms) do_nms_sort(detector_gpu.boxes, detector_gpu.probs, l.w*l.h*l.n, l.classes, nms);

	int        nboxes      = 0;
	int        letterbox   = 0;
	float      hier_thresh = 0.5;
	detection* dets        = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);
	//
	float nms = .4f;
	if (nms)
		do_nms_sort(dets, nboxes, l.classes, nms);

	// std::vector<bbox_t> bbox_vec;
	size_t nreal_count = 0;
	for (int i = 0; i < nboxes; ++i) {
		box b = dets[i].bbox;

		int const   obj_id = util_max_index(dets[i].prob, l.classes);
		float const prob   = dets[i].prob[obj_id];

		if (prob > thresh) {
			nreal_count++;
		}
	}

	bbox_vec.clear();
	bbox_vec.reserve(nreal_count);

	if (nreal_count > 0) {
		for (int i = 0; i < nboxes; ++i) {
			box         b      = dets[i].bbox;
			int const   obj_id = util_max_index(dets[i].prob, l.classes);
			float const prob   = dets[i].prob[obj_id];

			if (prob > thresh) {
				bbox_t bbox;
				bbox.x              = std::max((double)0, (b.x - b.w / 2.) * im.w);
				bbox.y              = std::max((double)0, (b.y - b.h / 2.) * im.h);
				bbox.w              = b.w * im.w;
				bbox.h              = b.h * im.h;
				bbox.obj_id         = obj_id;
				bbox.prob           = prob;
				bbox.track_id       = 0;
				bbox.frames_counter = 0;
				bbox.x_3d           = NAN;
				bbox.y_3d           = NAN;
				bbox.z_3d           = NAN;

				bbox_vec.push_back(bbox);
			}
		}
	}

	free_detections(dets, nboxes);
	if (sized.data)
		free(sized.data);

	// #ifdef GPU
	//	if (gpu_id != old_gpu_index)
	//		cudaSetDevice(old_gpu_index);
	// #endif
}

void mapped_labels(std::vector<char*>& detector_labels, const std::vector<label_mapping>& from_label_to_label)
{
	detector_labels.clear();
	for (size_t rule = 0; rule < from_label_to_label.size(); rule++) {
		for (size_t t = 0; t < from_label_to_label[rule].to.size(); t++) {

			bool found_ptr = false;
			for (size_t l = 0; l < detector_labels.size(); l++) {
				if (from_label_to_label[rule].to[t].compare(detector_labels[l]) == 0) {
					found_ptr = true;
					break;
				}
			}

			if (!found_ptr) {
				detector_labels.emplace_back();
				detector_labels.back() = (char*)from_label_to_label[rule].to[t].data();
			}
		}
	}

	std::sort(detector_labels.begin(), detector_labels.end(), [](const char* lhs, const char* rhs) {
		std::string_view lv{lhs};
		std::string_view rv{rhs};
		return lv.compare(rv) < 0;
	});
}

int main(int argc, char** argv)
{
	uint32_t window_width       = 1920;
	uint32_t window_height      = 1080;
	uint32_t network_width      = 320;
	uint32_t network_height     = 320;
	uint32_t tmp_network_width  = network_width;
	uint32_t tmp_network_height = network_height;

	network* custom_detector = nullptr;

	stack_vector::stack_vector<size_t, lru_size> lru_idxs;
	// Detector d;

	glfw3_setup_t r = glfw3_setup(window_width, window_height);

	NFD_Init();

	// ankerl::unordered_dense::map<std::string, int> labels_map;
	std::vector<label_mapping> from_label_to_label;
	// ankerl::unordered_dense::map<std::string, std::string>

	std::string imgui_id;
	imgui_id.reserve(64);

	std::string filepath_tmp;
	filepath_tmp.reserve(512); // ...probably sane assumption filepaths won't be this long

	// network net;

	// yolo_template_opts yolo_opts;
	// std::string template_out;
	// generate_yolov7_template(template_out, yolo_opts, 4000);

	// std::vector<char> raw_bytes;
	// zpp::bits::in     zpp_in(raw_bytes);
	// zpp::bits::out    zpp_out(raw_bytes);

	std::string built_with;
	if (built_with_cuda()) {
		fmt::format_to(std::back_inserter(built_with), "Built with CUDA");
	} else if (built_with_cudnn()) {
		fmt::format_to(std::back_inserter(built_with), "Built with CUDNN");
	} else if (built_with_opencv()) {
		fmt::format_to(std::back_inserter(built_with), "Built with OPENCV");
	} else {
		fmt::format_to(std::back_inserter(built_with), "Not Built with CUDA/CUDNN/OPENCV");
	}

	std::string_view working_directory = util::utf8::parent_path(argv[0]);

	std::vector<std::string>                       labels;
	std::vector<std::string>                       labels_tmp;
	ankerl::unordered_dense::map<std::string, int> marks;
	size_t                                         largest_label_size = 0;

	// simple_bounding_box bbox;
	// bbox.from(std::string_view{"0.5 0.5 0.25 0.25 dog"});

	// should be lowercase and contain leading .
	std::array<std::string_view, 6> image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tiff"};

	// ankerl::unordered_dense_map<std::string, int> marks;
	//  std::string label_folder;

	// we might end up opening a lot of images*
	std::vector<image_data> open_images;
	open_images.reserve(1024);

	nonrecursive_directory_data  label_folder;
	std::error_code              status_error_code = {};
	std::filesystem::file_status folder_status     = std::filesystem::status(working_directory, status_error_code);
	label_folder.current.from(working_directory, folder_status);
	// label_folder.update();

	// GLuint last_texture = 1;
	// Detector detector(cfg_file, weights_file);
	// Detector yolo_detector;
	std::string label_input;
	std::string lowercase_extension;
	std::string label_file_path;

	std::string from_combo_label_dropdown;
	size_t      from_combo_idx;
	std::string to_combo_label_dropdown;
	size_t      to_combo_idx;

	std::vector<ImVec2> uv_coordinates;
	size_t              active_window = 0;

	size_t selected_label_idx = 0ull;

	// bool show_image_tooltips = false;

	size_t selected_file_idx = ~0ull;

	bool adding_bounding_box = false;
	bool labels_dirty        = false;
	bool label_map_dirty     = false;

	bool show_label_map_window = false;
	bool show_demo_window      = true;

	std::vector<char*> training_labels;

	std::string cfgfile_str;
	std::string weightfile_str;
	std::string backupdir_str;

	std::vector<char*> train_paths;
	std::vector<char*> valid_paths;
	std::vector<char*> difficult_paths;

	std::vector<char*> detector_labels;
	std::string        bbox_txt;
	std::string        bbox_lbl;

	float detector_thresh         = 0.5f;
	float detector_visible_thresh = 0.7f;
	/*
		if (!ImGui::IsPopupOpen("Save?"))
			ImGui::OpenPopup("Save?");
		if (ImGui::BeginPopupModal("Save?", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {

		}
	*/
	std::jthread train_thread;
	std::jthread detect_thread;

	while (!glfwWindowShouldClose(r.window)) {
		// Poll and handle events (inputs, window resize, etc.)
		// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your
		// inputs.
		// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or
		// clear/overwrite your copy of the mouse data.
		// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or
		// clear/overwrite your copy of the keyboard data. Generally you may always pass all inputs to dear imgui, and
		// hide them from your application based on those two flags.
		glfwPollEvents();

		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// if (ImGui::)

		// ImGui::Text
		// ImGui::ShowDemoWindow(&show_demo_window);
		// ImGui::TextUnformatted(built_with.data(), built_with.data() + built_with.size());
		/*
		int width;
		int height;
		glfwGetFramebufferSize(r.window, &width, &height);
		ImGui::SetNextWindowSize(ImVec2(width, height)); // ensures ImGui fits the GLFW window
		ImGui::SetNextWindowPos(ImVec2(0, 0));
		if (ImGui::Begin("Image GUI", nullptr, ImGuiWindowFlags_::ImGuiWindowFlags_NoTitleBar |
		ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse |
			ImGuiWindowFlags_::ImGuiWindowFlags_NoResize)) {


		}
		ImGui::End();
		*/
		ImGuiIO& imio = ImGui::GetIO();
		/*
		ImGuiID  dockspace_id = ImGui::GetID("MyDockspace");
		if (ImGui::DockBuilderGetNode(dockspace_id) == NULL) {
			ImGui::DockBuilderRemoveNode(dockspace_id);                            // Clear out existing layout
			ImGui::DockBuilderAddNode(
							dockspace_id, ImGuiDockNodeFlagsPrivate_::ImGuiDockNodeFlags_DockSpace); // Add empty node
			ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

			ImGuiID dock_main_id = dockspace_id; // This variable will track the document node, however we are not using
												 // it here as we aren't docking anything into it.
			ImGuiID dock_id_prop = ImGui::DockBuilderSplitNode(
							dock_main_id, ImGuiDir_::ImGuiDir_Left, 0.20f, NULL, &dock_main_id);
			ImGuiID dock_id_bottom =
							ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Down, 0.20f, NULL, &dock_main_id);

			ImGui::DockBuilderDockWindow("Log", dock_id_bottom);
			ImGui::DockBuilderDockWindow("Properties", dock_id_prop);
			ImGui::DockBuilderDockWindow("Mesh", dock_id_prop);
			ImGui::DockBuilderDockWindow("Extra", dock_id_prop);
			ImGui::DockBuilderFinish(dockspace_id);
		}
		ImGui::DockSpace(dockspace_id);
		*/
		bool home_pressed = ImGui::IsKeyPressed(ImGuiKey_Home);
		if (home_pressed)
			ImGui::SetNextWindowPos(ImVec2{0, 0});
		// ImGui::SetNextWindowSize();

		if (ImGui::Begin("Labels", nullptr,
							ImGuiWindowFlags_MenuBar | (labels_dirty * ImGuiWindowFlags_UnsavedDocument))) {
			if (ImGui::BeginMenuBar()) {
				if (ImGui::BeginMenu("File")) {
					if (ImGui::MenuItem("Open", "Ctrl+O")) {
						open_labels(labels);
						labels_dirty = false;
					}
					if (ImGui::MenuItem("Save", "Ctrl+S")) {
						bool labels_saved = save_labels(labels);
						labels_dirty      = labels_saved ? false : labels_dirty;
					}

					ImGui::EndMenu();
				}

				if (ImGui::BeginMenu("Help")) {
					if (ImGui::MenuItem("Controls")) {
						// just for show
					}
					if (ImGui::IsItemHovered()) {
						ImGui::BeginTooltip();
						ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
						ImGui::TextUnformatted("add labels with the (+) button or press enter after typing\nCtrl+S to "
											   "save the label file for later\nCtrl+O to load a label file");
						ImGui::PopTextWrapPos();
						ImGui::EndTooltip();
					}

					// show_image_tooltips
					ImGui::EndMenu();
				}

				if (ImGui::BeginMenu("AI")) {
					if (ImGui::MenuItem("Configure")) {
						show_label_map_window = true;
					}

					ImGui::EndMenu();
				}

				ImGui::EndMenuBar();
			}

			if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_O)) {
				open_labels(labels);
				labels_dirty = false;
			}

			if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_S)) {
				bool labels_saved = save_labels(labels);
				labels_dirty      = labels_saved ? false : labels_dirty;
			}

			bool imio_previous                  = imio.ConfigInputTextEnterKeepActive;
			imio.ConfigInputTextEnterKeepActive = true;

			bool hit_enter = ImGui::InputText(
							"##NewLabel", &label_input, ImGuiInputTextFlags_::ImGuiInputTextFlags_EnterReturnsTrue);

			imio.ConfigInputTextEnterKeepActive = imio_previous;

			ImGui::SameLine();
			if ((ImGui::Button("(+) Label") || (hit_enter)) && label_input.size()) {
				bool found_label = false;

				for (size_t label_idx = 0; label_idx < labels.size(); label_idx++) {
					if (labels[label_idx].compare(label_input) == 0) {
						found_label = true;
						break;
					}
				}

				if (!found_label) {
					labels.emplace_back(label_input);
					labels_dirty = true;
				}
			}

			// bool new_label_active = ImGui::IsItemActive() || ImGui::IsItemActivated();
			// fmt::print("l:{}\te:{}\n", new_label_active, hit_enter);

			/*
			* 	bool hit_enter = false;
			ImGui::InputText("##NewLabel", &label_input, ImGuiInputTextFlags_::ImGuiInputTextFlags_None,
				(ImGuiInputTextCallback)[](ImGuiInputTextCallbackData* data) {
					*(bool*)data->UserData = data->EventKey == ImGuiKey::ImGuiKey_Enter;
					return 0;
				}, (void*)&hit_enter);
				*/

			largest_label_size = 0ull; // reset the worst case label size when we render these
			for (size_t i = 0; i < labels.size(); i++) {
				// ImGui::TextUnformatted(built_with.data(), built_with.data() + built_with.size());
				// ImGui::SameLine();
				ImGui::PushID((int)i);

				// imgui_id.clear();
				// fmt::format_to(std::back_inserter(imgui_id), "##select{}", i);
				bool label_is_selected = selected_label_idx == i;

				if (ImGui::RadioButton("##select", label_is_selected))
					selected_label_idx = i;

				ImGui::SameLine();

				// imgui_id.clear();
				// fmt::format_to(std::back_inserter(imgui_id), "##label{}", i);
				ImGui::InputText("##label", &labels[i]);

				ImGui::SameLine();

				if (ImGui::Button("(-) Label")) {
					labels.erase(labels.begin() + i);
					i--;
					labels_dirty = true;
					ImGui::PopID();
					continue;
				}
				ImGui::PopID();
				largest_label_size = std::max(largest_label_size, labels[i].size());
				// ImGui::TextUnformatted(labels[i].data(), labels[i].data() + labels[i].size());
			}
			/*
			if (ImGui::BeginListBox("Selection")) {
				for (int n = 0; n < labels.size(); n++) {
					const bool is_selected = (item_current_idx == n);
					if (ImGui::Selectable(items[n], is_selected))
						item_current_idx = n;

					// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndListBox();
			}
			*/
			// label file
			// <label> 0
			// <label> 1
			// <label> 2

			// for each <name>.jpg / <name>.png in a folder there is a <name>.txt
			// where each line looks like this:
			// <object-class:int> <x_center:float> <y_center:float> <width:float> <height:float>
		}
		ImGui::End();

		if (home_pressed)
			ImGui::SetNextWindowPos(ImVec2{10, 10});
		if (ImGui::Begin("File Navigation", nullptr, ImGuiWindowFlags_MenuBar)) {
			// TODO: make a better directory navigator
			if (ImGui::BeginMenuBar()) {
				if (ImGui::BeginMenu("Folder")) {
					if (ImGui::MenuItem("Open", "Ctrl+O"))
						open_directory(label_folder, working_directory);

					ImGui::EndMenu();
				}

				if (ImGui::BeginMenu("Help")) {
					if (ImGui::MenuItem("Controls")) {
						// just for show
					}
					if (ImGui::IsItemHovered()) {
						ImGui::BeginTooltip();
						ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
						ImGui::TextUnformatted("left click on any image file to open the label editor\nleft click on "
											   ".. to travel up a directory\nleft click on a folder to travel into a "
											   "directory\nCtrl+O to select a folder using your operating "
											   "system\nCtrl+Up/Down to immediately open image files in order");
						ImGui::PopTextWrapPos();
						ImGui::EndTooltip();
					}

					// show_image_tooltips
					ImGui::EndMenu();
				}

				ImGui::EndMenuBar();
			}

			if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_O)) {
				open_directory(label_folder, working_directory);
			}

			// careful! folder.paths must live to the next frame
			size_t selected_file = linuxlike_directory_view(label_folder, selected_file_idx);

			if (selected_file < label_folder.paths.size()) {
				selected_file_idx = selected_file;
				open_image_unchecked(label_folder, open_images, lowercase_extension, label_file_path, image_extensions,
								labels, selected_file_idx);
			}
		}
		ImGui::End();

		if (ImGui::IsKeyDown(ImGuiMod_Ctrl) && ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
			selected_file_idx = selected_file_idx < label_folder.paths.size()
											  ? (selected_file_idx - 1) % label_folder.paths.size()
											  : 0ull;
			open_image_unchecked(label_folder, open_images, lowercase_extension, label_file_path, image_extensions,
							labels, selected_file_idx);
		} else if (ImGui::IsKeyDown(ImGuiMod_Ctrl) && ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
			selected_file_idx = selected_file_idx < label_folder.paths.size()
											  ? (selected_file_idx + 1) % label_folder.paths.size()
											  : 0ull;
			open_image_unchecked(label_folder, open_images, lowercase_extension, label_file_path, image_extensions,
							labels, selected_file_idx);
		}

		// if (ImGui::Begin("Label Viewer")) {

		ImGuiIO& io = ImGui::GetIO();
		// ImVec2 window_size = ImGui::GetWindowSize();
		// window_size.x      = std::max((float)1280, window_size.x);
		// window_size.y      = std::max((float)720, window_size.y);
		// ImGui::SetNextWindowSize(window_size, ImGuiCond_::ImGuiCond_Once);
		if (home_pressed)
			ImGui::SetNextWindowPos(ImVec2{20, 20});
		if (ImGui::Begin("Editor", nullptr, ImGuiWindowFlags_MenuBar)) {
			bool save_current  = false;
			bool save_all      = false;
			bool run_dectector = false;
			if (ImGui::BeginMenuBar()) {
				if (ImGui::BeginMenu("File")) {
					if (ImGui::MenuItem("Save Current", "Ctrl+S")) {
						save_current = true;
						// bool saved           = save_bounding_boxes(imgui_id, open_images[i]);
						// open_images[i].dirty = saved ? false : open_images[i].dirty;
					}
					if (ImGui::MenuItem("Save All")) {
						save_all = true;
					}

					ImGui::EndMenu();
				}

				if (ImGui::BeginMenu("AI")) {
					if (ImGui::MenuItem("Configure")) {
						show_label_map_window = true;
					}

					ImGui::EndMenu();
				}

				if (ImGui::BeginMenu("Detect")) {
					if (ImGui::MenuItem("Run Detector", "Ctrl+D")) {
						run_dectector = true;
					}
					ImGui::EndMenu();
				}

				if (ImGui::BeginMenu("Help")) {
					if (ImGui::MenuItem("Controls")) {
						// just for show
					}
					if (ImGui::IsItemHovered()) {
						ImGui::BeginTooltip();
						ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
						ImGui::TextUnformatted("right click and drag to add labels\nleft click and drag to move "
											   "around\nmouse wheel to zoom in and out\nclick and hold on a label "
											   "and "
											   "press delete to remove a label\nCtrl+S to save the label "
											   "file\n\twill "
											   "be "
											   "found at <filepath>.<extension>_lbl");
						ImGui::PopTextWrapPos();
						ImGui::EndTooltip();
					}

					if (ImGui::MenuItem("Label Files")) {
					}
					if (ImGui::IsItemHovered()) {
						ImGui::BeginTooltip();
						ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
						ImGui::TextUnformatted("label files are a text file containing lines in the following "
											   "format:\n"
											   "<x_center> <y_center> <width> <height> <label>\n"
											   "where x_center, y_center, with and height are floating point "
											   "values. "
											   "x_center and y_center representing their unit vector distance "
											   "from "
											   "the "
											   "top left corner of the image and width and height representing "
											   "the "
											   "normalized width and height. For example the line \n0.0 0.0 0.5 "
											   "0.5 "
											   "dog\nrepresents a bounding box in the lop left corner to the "
											   "center "
											   "of "
											   "the image with the label \"dog\"");
						ImGui::PopTextWrapPos();
						ImGui::EndTooltip();
					}

					// show_image_tooltips
					ImGui::EndMenu();
				}

				ImGui::EndMenuBar();
			}

			if (ImGui::BeginTabBar("##openfiles")) {
				for (size_t i = 0; i < open_images.size(); i++) {
					imgui_id.clear();
					fmt::format_to(std::back_inserter(imgui_id), "{0}", open_images[i].file.path);

					// ImGui::SetNextWindowSize(window_size);
					// ImGuiWindowFlags_MenuBar |
					if (save_all) { // ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_S)
						bool saved           = save_bounding_boxes(imgui_id, open_images[i]);
						open_images[i].dirty = saved ? false : open_images[i].dirty;
					}

					if (open_images[i].show &&
									ImGui::BeginTabItem(imgui_id.c_str(), &(open_images[i].show),

													(open_images[i].dirty * ImGuiTabItemFlags_UnsavedDocument))) {
#if 0
						if (ImGui::BeginMenuBar()) {
							if (ImGui::BeginMenu("File")) {
								if (ImGui::MenuItem("Save", "Ctrl+S")) {
									bool saved           = save_bounding_boxes(imgui_id, open_images[i]);
									open_images[i].dirty = saved ? false : open_images[i].dirty;
								}

								ImGui::EndMenu();
							}

							if (ImGui::BeginMenu("Help")) {
								if (ImGui::MenuItem("Controls")) {
									// just for show
								}
								if (ImGui::IsItemHovered()) {
									ImGui::BeginTooltip();
									ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
									ImGui::TextUnformatted(
													"right click and drag to add labels\nleft click and drag to move "
													"around\nmouse wheel to zoom in and out\nclick and hold on a label "
													"and "
													"press delete to remove a label\nCtrl+S to save the label "
													"file\n\twill "
													"be "
													"found at <filepath>.<extension>_lbl");
									ImGui::PopTextWrapPos();
									ImGui::EndTooltip();
								}

								if (ImGui::MenuItem("Label Files")) {
								}
								if (ImGui::IsItemHovered()) {
									ImGui::BeginTooltip();
									ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
									ImGui::TextUnformatted(
													"label files are a text file containing lines in the following "
													"format:\n"
													"<x_center> <y_center> <width> <height> <label>\n"
													"where x_center, y_center, with and height are floating point "
													"values. "
													"x_center and y_center representing their unit vector distance "
													"from "
													"the "
													"top left corner of the image and width and height representing "
													"the "
													"normalized width and height. For example the line \n0.0 0.0 0.5 "
													"0.5 "
													"dog\nrepresents a bounding box in the lop left corner to the "
													"center "
													"of "
													"the image with the label \"dog\"");
									ImGui::PopTextWrapPos();
									ImGui::EndTooltip();
								}

								// show_image_tooltips
								ImGui::EndMenu();
							}

							ImGui::EndMenuBar();
						}
#endif
						// if not loaded, attempt to load
						lazy_load_image(open_images, lru_idxs, i);
						if (save_current || ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_S)) {
							bool saved           = save_bounding_boxes(imgui_id, open_images[i]);
							open_images[i].dirty = saved ? false : open_images[i].dirty;
						}
						// if not loaded at this point? BAIL!
						if (!open_images[i].texture_loaded) {
							continue;
						}

						if (run_dectector || ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_D)) {
							detect_thread = std::jthread{[&open_images, i, &detector_labels, from_label_to_label,
																		 cfgfile_str, weightfile_str,
																		 detector_thresh]() {
								network* net = nullptr;

								if (cfgfile_str.size() > 0 && weightfile_str.size() > 0) {
									net = load_network((char*)cfgfile_str.data(), (char*)weightfile_str.data(), 0);
								} else {
									return;
								}

								if (net == nullptr)
									return;

								open_images[i].detect_bounding_boxes_tmp.clear();
								image_t img;
								util_load_image(img, open_images[i].file.path);

								try {
									detect(open_images[i].detect_bounding_boxes_tmp, *net, img, detector_thresh, false);
									open_images[i].detected = true;
								} catch (...) {
									open_images[i].detect_bounding_boxes_tmp.clear();
								}

								if (net) {
									free_network_ptr(net);
									free(net);
								}
								if (img.data)
									free(img.data);
							}};
						}

						if (open_images[i].detected && open_images[i].detect_bounding_boxes_tmp.size()) {
							/*
							open_images[i].detect_bounding_boxes.assign(open_images[i].detect_bounding_boxes_tmp.data(),
											open_images[i].detect_bounding_boxes_tmp.data() + open_images[i]
															.detect_bounding_boxes_tmp.size());
															*/
							mapped_labels(detector_labels, from_label_to_label);

							open_images[i].detect_bounding_boxes.clear();

							for (size_t box_idx = 0; box_idx < open_images[i].detect_bounding_boxes_tmp.size();
											box_idx++) {
								bbox_t&              box  = open_images[i].detect_bounding_boxes_tmp[box_idx];
								simple_bounding_box& sbox = open_images[i].detect_bounding_boxes.emplace_back();
								sbox.x_center             = box.x / (float)open_images[i].width;
								sbox.y_center             = box.y / (float)open_images[i].height;
								sbox.width                = box.w / (float)open_images[i].width;
								sbox.height               = box.h / (float)open_images[i].height;
								sbox.label.clear();
								sbox.prob = box.prob;

								if (box.obj_id < detector_labels.size()) {
									fmt::format_to(std::back_inserter(sbox.label), "{} ({}%)",
													detector_labels[box.obj_id], box.prob * 100.0f);
								} else {
									fmt::format_to(std::back_inserter(sbox.label), "{} ({}%)", box.obj_id,
													box.prob * 100.0f);
								}
							}

							open_images[i].detect_bounding_boxes_tmp.clear();
							open_images[i].detected = false;
						}

						ImDrawList* draw_list = ImGui::GetWindowDrawList();

						// Get the current ImGui cursor position
						ImVec2 canvas_p0   = ImGui::GetCursorScreenPos();    // ImDrawList API uses screen coordinates!
						ImVec2 canvas_size = ImGui::GetContentRegionAvail(); // Resize canvas to what's available

						// guarantee a minimum canvas size
						canvas_size.x = std::max(canvas_size.x, 256.0f);
						canvas_size.y = std::max(canvas_size.y, 250.0f);

						if (open_images[i].first_show) {
							float scale_x = canvas_size.x / (float)open_images[i].width;
							float scale_y = canvas_size.y / (float)open_images[i].height;

							float min_scale = std::min(scale_x, scale_y);

							open_images[i].transform.scale.x = min_scale;
							open_images[i].transform.scale.y = min_scale;
							/*
							ImVec2 pixel_size = {open_images[i].width * open_images[i].transform.scale.x,
											open_images[i].height * open_images[i].transform.scale.y};
											*/
							// snap to center
							open_images[i].first_show = false;
						}

						ImVec2 canvas_p1 = ImVec2{canvas_p0.x + canvas_size.x, canvas_p0.y + canvas_size.y};

						imgui_id.clear();
						fmt::format_to(std::back_inserter(imgui_id), "##canvas_{}", i);
						ImGui::InvisibleButton(imgui_id.c_str(), canvas_size,
										ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight |
														ImGuiButtonFlags_MouseButtonMiddle);

						const bool canvas_hovered         = ImGui::IsItemHovered(); // Hovered
						const bool canvas_active          = ImGui::IsItemActive();  // Held
						const bool canvas_hovered_tooltip = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort);

						// const ImVec2 canvas_p0 = ImGui::GetItemRectMin(); // alternatively we can get the rectangle
						// like this const ImVec2 canvas_p1 = ImGui::GetItemRectMax();

						// Draw border and background color
						draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
						draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

						// TODO: shift or ctrl to slow zoom movement
						// if (canvas_active) {
						float zoom_rate   = 0.1f;
						float zoom_mouse  = io.MouseWheel * zoom_rate; //-0.1f 0.0f 0.1f
						float hzoom_mouse = zoom_mouse * 0.5f;
						float zoom_delta  = zoom_mouse *
										   open_images[i].transform.scale.x; // each step grows or shrinks image by 10%

						// on screen (top left of image)
						ImVec2 old_origin = {canvas_p0.x + open_images[i].transform.translate.x,
										canvas_p0.y + open_images[i].transform.translate.y};
						// on screen (bottom right of image)
						ImVec2 old_p1 = {old_origin.x + (open_images[i].width * open_images[i].transform.scale.x),
										old_origin.y + (open_images[i].height * open_images[i].transform.scale.y)};
						// on screen (center of what we get to see), when adjusting scale this doesn't change!
						ImVec2 old_and_new_canvas_center = {
										canvas_p0.x + canvas_size.x * 0.5f, canvas_p0.y + canvas_size.y * 0.5f};
						// in image coordinate offset of the center
						ImVec2 image_center = {old_and_new_canvas_center.x - old_origin.x,
										old_and_new_canvas_center.y - old_origin.y};

						ImVec2 old_uv_image_center = {
										image_center.x / (open_images[i].width * open_images[i].transform.scale.x),
										image_center.y / (open_images[i].height * open_images[i].transform.scale.y)};

						open_images[i].transform.scale.x += canvas_hovered ? zoom_delta : 0.0f;
						open_images[i].transform.scale.y += canvas_hovered ? zoom_delta : 0.0f;

						// 2.0f -> 2x zoom in
						// 1.0f -> normal
						// 0.5f -> 2x zoom out
						// TODO: clamp based on image size, do we go pixel level?
						open_images[i].transform.scale.x = std::clamp(open_images[i].transform.scale.x, 0.01f, 100.0f);
						open_images[i].transform.scale.y = std::clamp(open_images[i].transform.scale.y, 0.01f, 100.0f);

						// on screen new target center
						ImVec2 new_image_center = {(open_images[i].width * open_images[i].transform.scale.x *
																   old_uv_image_center.x),
										(open_images[i].height * open_images[i].transform.scale.y *
														old_uv_image_center.y)};

						// readjust to center
						open_images[i].transform.translate.x -= new_image_center.x - image_center.x;
						open_images[i].transform.translate.y -= new_image_center.y - image_center.y;

						float drag_delta_x = io.MouseDelta.x;
						float drag_delta_y = io.MouseDelta.y;

						// 0 out second parameter if a context menu is open
						if (canvas_active && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 1.0f)) {
							open_images[i].transform.translate.x += drag_delta_x;
							open_images[i].transform.translate.y += drag_delta_y;
						}

						const ImVec2 origin(canvas_p0.x + open_images[i].transform.translate.x,
										canvas_p0.y + open_images[i].transform.translate.y); // Lock scrolled origin

						// we need to control the rectangle we're going to draw and the uv coordinates
						const ImVec2 image_p1 = {origin.x + (open_images[i].transform.scale.x * open_images[i].width),
										origin.y + (open_images[i].transform.scale.x * open_images[i].height)};

						draw_list->PushClipRect(ImVec2{canvas_p0.x + 2.0f, canvas_p0.y + 2.0f},
										ImVec2{canvas_p1.x - 2.0f, canvas_p1.y - 2.0f}, true);
						draw_list->AddImage(open_images[i].texture_id, origin, image_p1);

						const ImVec2 mouse_pos_in_canvas = {imio.MousePos.x - origin.x, imio.MousePos.y - origin.y};
						const ImVec2 mouse_uv_in_image   = {
                                        mouse_pos_in_canvas.x /
                                                        (open_images[i].width * open_images[i].transform.scale.x),
                                        mouse_pos_in_canvas.y /
                                                        (open_images[i].height * open_images[i].transform.scale.y)};

						if (canvas_hovered && !adding_bounding_box && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
							active_window = i; // remember which window we were interacting with
							uv_coordinates.emplace_back(mouse_uv_in_image);
							uv_coordinates.emplace_back(mouse_uv_in_image);
							adding_bounding_box = true;
						}
						// we have to gaurd against right click and dragging off the window
						if (canvas_hovered && adding_bounding_box && active_window == i) {
							uv_coordinates.back() = mouse_uv_in_image;

							ImVec2 top_left_rect = {
											origin.x + (open_images[i].width * open_images[i].transform.scale.x *
																	   (uv_coordinates[0].x)),
											origin.y + (open_images[i].height * open_images[i].transform.scale.y *
																	   (uv_coordinates[0].y))};

							ImVec2 btm_right_rect = {
											origin.x + (open_images[i].width * open_images[i].transform.scale.x *
																	   (uv_coordinates[1].x)),
											origin.y + (open_images[i].height * open_images[i].transform.scale.y *
																	   (uv_coordinates[1].y))};

							ImU32 rect_color = ImU32{0xffff0000};
							// transparent rectangle background
							draw_list->AddRectFilled(top_left_rect, btm_right_rect, rect_color & ImU32{0x1fffffff});
							// solid border
							draw_list->AddRect(top_left_rect, btm_right_rect, rect_color);

							if (!ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
								float box_w = std::abs(uv_coordinates[0].x - uv_coordinates[1].x);
								float box_h = std::abs(uv_coordinates[0].y - uv_coordinates[1].y);

								// presumably unlikely, don't label anything thinner than a pixel's size
								if (box_w < (1.0f / (float)(open_images[i].width)) ||
												box_h < (1.0f / (float)(open_images[i].width))) {
									adding_bounding_box = false;
									uv_coordinates.clear();
								} else {
									simple_bounding_box& box = open_images[i].manual_bounding_boxes.emplace_back();

									box.x_center = (uv_coordinates[0].x + uv_coordinates[1].x) * 0.5f;
									box.y_center = (uv_coordinates[0].y + uv_coordinates[1].y) * 0.5f;

									box.width  = box_w;
									box.height = box_h;

									box.label.clear();
									// box.label.reserve(largest_label_size);
									if (selected_label_idx < labels.size()) {
										box.label.append(labels[selected_label_idx]);
									}

									adding_bounding_box = false;
									uv_coordinates.clear();
									open_images[i].dirty = true;
								}
								// dirty flag
								// adding_bounding_box_label = true;
							}
						}

						// if (adding_bounding_box_label) {
						// }

						// prevent deleting multiple rectangles with one keypress
						// size_t erase_this = ~size_t{0};
						bool erased_one = false;
						for (size_t box_idx = 0; box_idx < open_images[i].manual_bounding_boxes.size(); box_idx++) {
							simple_bounding_box& box   = open_images[i].manual_bounding_boxes[box_idx];
							ImVec2               hsize = {box.width * 0.5f, box.height * 0.5f};
							//
							ImVec2 top_left_rect = {
											origin.x + (open_images[i].width * open_images[i].transform.scale.x *
																	   (box.x_center - hsize.x)),
											origin.y + (open_images[i].height * open_images[i].transform.scale.y *
																	   (box.y_center - hsize.y))};
							ImVec2 btm_right_rect = {
											origin.x + (open_images[i].width * open_images[i].transform.scale.x *
																	   (box.x_center + hsize.x)),
											origin.y + (open_images[i].height * open_images[i].transform.scale.y *
																	   (box.y_center + hsize.y))};

							const char* label_start = box.label.c_str();
							const char* label_end   = label_start + box.label.size();

							ImVec2 text_size = ImGui::CalcTextSize(label_start, label_end);

							ImVec2 text_btm_right = {top_left_rect.x + text_size.x, top_left_rect.y + text_size.y};

							bool rectangle_hovered = ImGui::IsMouseHoveringRect(top_left_rect, btm_right_rect);
							bool text_hovered      = ImGui::IsMouseHoveringRect(top_left_rect, text_btm_right);

							// erase in place, keep this struct stable
							if (!erased_one && canvas_active && (rectangle_hovered || text_hovered) &&
											ImGui::IsKeyPressed(ImGuiKey::ImGuiKey_Delete)) {

								open_images[i].manual_bounding_boxes.erase(
												open_images[i].manual_bounding_boxes.begin() + box_idx);
								box_idx--;
								erased_one = true;
								// erase_this = box_idx;
								open_images[i].dirty = true;
								continue; // skip drawing this box
							}

							imgui_id.clear();
							fmt::format_to(std::back_inserter(imgui_id), "##edit_label_{}_{}", i, box_idx);
							if (canvas_active && (rectangle_hovered || text_hovered) &&
											ImGui::IsKeyPressed(ImGuiKey::ImGuiKey_Enter)) {
								// ImGui::DropDown()
								ImGui::OpenPopup(imgui_id.data());
								label_input.clear();
								label_input.append(box.label.data(),
												box.label.size()); // copy the text into the input form
							}

							// pop open a gui to edit the label's text
							if (ImGui::BeginPopup(imgui_id.data())) {
								imgui_id.clear();
								fmt::format_to(std::back_inserter(imgui_id), "##new_label_{}_{}", i, box_idx);

								bool imio_previous                  = imio.ConfigInputTextEnterKeepActive;
								imio.ConfigInputTextEnterKeepActive = true;

								bool hit_enter = ImGui::InputText(imgui_id.data(), &label_input,
												ImGuiInputTextFlags_::ImGuiInputTextFlags_EnterReturnsTrue);

								imio.ConfigInputTextEnterKeepActive = imio_previous;

								imgui_id.clear();
								fmt::format_to(std::back_inserter(imgui_id), "(+) Label##popup_label_add_{}_{}", i,
												box_idx);

								ImGui::SameLine();
								if ((ImGui::Button(imgui_id.data()) || (hit_enter)) && label_input.size()) {
									bool found_label = false;

									for (size_t label_idx = 0; label_idx < labels.size(); label_idx++) {
										if (labels[label_idx].compare(label_input) == 0) {
											found_label = true;
											break;
										}
									}

									if (!found_label) {
										labels.emplace_back(label_input);
										labels_dirty = true;
									}

									box.label.clear();
									box.label.append(label_input.data(), label_input.size());
								}

								ImGui::EndPopup();
							}

							ImU32 rect_color =
											(rectangle_hovered || text_hovered) ? ImU32{0xff00ff00} : ImU32{0xffff0000};
							// transparent rectangle background
							draw_list->AddRectFilled(top_left_rect, btm_right_rect, rect_color & ImU32{0x1fffffff});
							// solid border
							draw_list->AddRect(top_left_rect, btm_right_rect, rect_color);

							// solid color behind text
							draw_list->AddRectFilled(top_left_rect, text_btm_right, rect_color);

							draw_list->AddText(top_left_rect, ImU32{0xffffffff}, label_start, label_end);
						}

						for (size_t box_idx = 0; box_idx < open_images[i].detect_bounding_boxes.size(); box_idx++) {
							simple_bounding_box& box = open_images[i].detect_bounding_boxes[box_idx];
							if (box.prob < detector_visible_thresh)
								continue;

							ImVec2 hsize = {box.width * 0.5f, box.height * 0.5f};
							//
							ImVec2 top_left_rect = {
											origin.x + (open_images[i].width * open_images[i].transform.scale.x *
																	   (box.x_center - hsize.x)),
											origin.y + (open_images[i].height * open_images[i].transform.scale.y *
																	   (box.y_center - hsize.y))};
							ImVec2 btm_right_rect = {
											origin.x + (open_images[i].width * open_images[i].transform.scale.x *
																	   (box.x_center + hsize.x)),
											origin.y + (open_images[i].height * open_images[i].transform.scale.y *
																	   (box.y_center + hsize.y))};
							bbox_txt.clear();
							/*
							if (box.obj_id < detector_labels.size()) {
								fmt::format_to(std::back_inserter(bbox_txt), "{} ({}%)", detector_labels[box.obj_id],
												box.prob * 100.0f);
							} else {
								fmt::format_to(std::back_inserter(bbox_txt), "{} ({}%)", box.obj_id, box.prob * 100.0f);
							}
							*/
							const char* label_start = bbox_txt.c_str();
							const char* label_end   = label_start + bbox_txt.size();

							ImVec2 text_size = ImGui::CalcTextSize(label_start, label_end);

							ImVec2 text_btm_right = {top_left_rect.x + text_size.x, top_left_rect.y + text_size.y};

							bool rectangle_hovered = ImGui::IsMouseHoveringRect(top_left_rect, btm_right_rect);
							bool text_hovered      = ImGui::IsMouseHoveringRect(top_left_rect, text_btm_right);

							// erase in place, keep this struct stable
							if (!erased_one && canvas_active && (rectangle_hovered || text_hovered) &&
											ImGui::IsKeyPressed(ImGuiKey::ImGuiKey_Delete)) {

								open_images[i].detect_bounding_boxes.erase(
												open_images[i].detect_bounding_boxes.begin() + box_idx);
								box_idx--;
								erased_one = true;
								// erase_this = box_idx;
								// open_images[i].dirty = true; // don't dirty when removing detections
								continue; // skip drawing this box
							}

							imgui_id.clear();
							fmt::format_to(std::back_inserter(imgui_id), "##edit_label_detect_{}_{}", i, box_idx);
							if (canvas_active && (rectangle_hovered || text_hovered) &&
											ImGui::IsKeyPressed(ImGuiKey::ImGuiKey_Enter)) {
								// ImGui::DropDown()
								ImGui::OpenPopup(imgui_id.data());
								label_input.clear();
								/*
								if (box.obj_id < detector_labels.size()) {
									label_input.append(
													detector_labels[box.obj_id]); // copy the text into the input form
								}
								*/
							}

							// pop open a gui to edit the label's text
							if (ImGui::BeginPopup(imgui_id.data())) {
								imgui_id.clear();
								fmt::format_to(std::back_inserter(imgui_id), "##new_label_detect_{}_{}", i, box_idx);

								bool imio_previous                  = imio.ConfigInputTextEnterKeepActive;
								imio.ConfigInputTextEnterKeepActive = true;

								bool hit_enter = ImGui::InputText(imgui_id.data(), &label_input,
												ImGuiInputTextFlags_::ImGuiInputTextFlags_EnterReturnsTrue);

								imio.ConfigInputTextEnterKeepActive = imio_previous;

								imgui_id.clear();
								fmt::format_to(std::back_inserter(imgui_id), "(+) Label##popup_label_detect_add_{}_{}",
												i, box_idx);

								ImGui::SameLine();
								if ((ImGui::Button(imgui_id.data()) || (hit_enter)) && label_input.size()) {
									bool found_label = false;

									for (size_t label_idx = 0; label_idx < labels.size(); label_idx++) {
										if (labels[label_idx].compare(label_input) == 0) {
											found_label = true;
											break;
										}
									}

									if (!found_label) {
										labels.emplace_back(label_input);
										labels_dirty = true;
									}

									// add a new box
									simple_bounding_box& back = open_images[i].manual_bounding_boxes.emplace_back();
									back.from(box);
									back.label.append(label_input.data(), label_input.size());
									// remove this box
									open_images[i].detect_bounding_boxes.erase(
													open_images[i].detect_bounding_boxes.begin() + box_idx);
									box_idx--;
								}

								ImGui::EndPopup();
							}

							ImU32 rect_color =
											(rectangle_hovered || text_hovered) ? ImU32{0xff0000ff} : ImU32{0xff0000f0};
							// transparent rectangle background
							draw_list->AddRectFilled(top_left_rect, btm_right_rect, rect_color & ImU32{0x1fffffff});
							// solid border
							draw_list->AddRect(top_left_rect, btm_right_rect, rect_color);

							// solid color behind text
							draw_list->AddRectFilled(top_left_rect, text_btm_right, rect_color);

							draw_list->AddText(top_left_rect, ImU32{0xffffffff}, label_start, label_end);
						}

						/*
						if (erase_this < open_images[i].manual_bounding_boxes.size()) {
							// remove the targetted idx
							open_images[i].manual_bounding_boxes.erase(open_images[i].manual_bounding_boxes.begin() +
						erase_this);
						}
						*/
						/*
							ImGui::IsMouseHovering();
							if (opt_enable_context_menu && drag_delta_x == 0.0f && drag_delta_y == 0.0f)
								ImGui::OpenPopupOnItemClick("context", ImGuiPopupFlags_MouseButtonRight);
							if (ImGui::BeginPopup("context")) {
								if (ImGui::MenuItem("Remove one", NULL, false, points.Size > 0)) {
									points.resize(points.size() - 2);
								}
								if (ImGui::MenuItem("Remove all", NULL, false, points.Size > 0)) {
									points.clear();
								}

							ImGui::EndPopup();
						*/

						draw_list->PopClipRect();
						ImGui::EndTabItem();
					}
					// ImGui::End();
				}
				ImGui::EndTabBar();
			}
		}
		ImGui::End();

		if (show_label_map_window) {
			if (home_pressed)
				ImGui::SetNextWindowPos(ImVec2{30, 30});
			if (ImGui::Begin("Training", &show_label_map_window,
								ImGuiWindowFlags_MenuBar | (label_map_dirty * ImGuiWindowFlags_UnsavedDocument))) {
				/*
				ImGui::Button("Generate Config File")
				{
					nfdu8char_t* out    = nullptr;
					nfdresult_t  result = NFD_SaveDialogU8(&out, nullptr, 0, nullptr, nullptr);

					if (result == NFD_OKAY) {
						yolo_template_opts opts;
						opts.
						write_yolo7_cfg_file(out, imgui_id, );
						free(out);
					}
				}
				*/
				bool cfg_enter = ImGui::InputText("config file##cfgfilepath", &cfgfile_str,
								ImGuiInputTextFlags_::ImGuiInputTextFlags_EnterReturnsTrue);
				ImGui::SameLine();
				if (ImGui::Button("browse##cfgfile") || cfg_enter) {
					nfdu8char_t* out = nullptr;
					// nfdresult_t  result = NFD_OpenDialogU8(&out, nullptr, 0, nullptr);
					nfdresult_t result = NFD_SaveDialogU8(&out, nullptr, 0, nullptr, nullptr);

					if (result == NFD_OKAY) {
						cfgfile_str.clear();
						cfgfile_str.append(out);
						free(out);
#if 0
						if (cfgfile_str.size() > 0 && weightfile_str.size() > 0) {
							network* new_network = load_network(cfgfile_str.data(), weightfile_str.data(), 0);
							if (new_network && custom_detector) {
								free_network(*custom_detector);
								free_network_ptr(custom_detector);
								custom_detector = new_network;
							} else if (!custom_detector) {
								custom_detector = new_network;
							}
						}
#endif
					}
				}

				bool weight_enter = ImGui::InputText("weight file##weightfilepath", &weightfile_str,
								ImGuiInputTextFlags_::ImGuiInputTextFlags_EnterReturnsTrue);
				ImGui::SameLine();
				if (ImGui::Button("browse##weightfile") || weight_enter) {
					nfdu8char_t* out    = nullptr;
					nfdresult_t  result = NFD_OpenDialogU8(&out, nullptr, 0, nullptr);

					if (result == NFD_OKAY) {
						weightfile_str.clear();
						weightfile_str.append(out);
						free(out);
#if 0
						if (cfgfile_str.size() > 0 && weightfile_str.size() > 0) {
							network* new_network = load_network(cfgfile_str.data(), weightfile_str.data(), 0);
							if (new_network && custom_detector) {
								free_network(*custom_detector);
								free_network_ptr(custom_detector);
								custom_detector = new_network;
							} else if (!custom_detector) {
								custom_detector = new_network;
							}
						}
#endif
					}
				}

				bool backupdir_enter = ImGui::InputText("weight directory##backupdirpath", &backupdir_str,
								ImGuiInputTextFlags_::ImGuiInputTextFlags_EnterReturnsTrue);
				ImGui::SameLine();
				if (ImGui::Button("browse##backupdirectory") || backupdir_enter) {
					nfdu8char_t* out    = nullptr;
					nfdresult_t  result = NFD_PickFolderU8(&out, nullptr);

					if (result == NFD_OKAY) {
						backupdir_str.clear();
						backupdir_str.append(out);
						free(out);
					}
				}

				if (ImGui::BeginTabBar("##trainfiles_tab")) {
					if (ImGui::BeginTabItem("Training Paths")) {
						if (ImGui::Button("Add File##add_training_file")) {
							nfdu8char_t* out    = nullptr;
							nfdresult_t  result = NFD_OpenDialogU8(&out, nullptr, 0, nullptr);

							if (result == NFD_OKAY) {
								train_paths.emplace_back(out);
								// free(out);
							}
						}

						if (ImGui::Button("Add Folder##add_training_folder")) {
							nfdu8char_t* out    = nullptr;
							nfdresult_t  result = NFD_PickFolderU8(&out, nullptr);

							if (result == NFD_OKAY) {
								std::error_code                     iterator_error = {};
								std::filesystem::directory_iterator parent_iterator{out, iterator_error};

								if (iterator_error == std::error_code{}) {
									// paths.clear();
									for (const std::filesystem::directory_entry& entry : parent_iterator) {
										std::error_code status_error = {};
										// std::filesystem::file_status entry_status =
										// std::filesystem::status(entry.path(), status_error);
										std::filesystem::file_status entry_status = entry.status(status_error);
										if (status_error == std::error_code{} &&
														std::filesystem::is_regular_file(entry_status)) {
											string_from(filepath_tmp, entry.path());

											auto pathit = std::find_if(train_paths.data(),
															train_paths.data() + train_paths.size(),
															[&filepath_tmp](char* ptr) {
																return filepath_tmp.compare(ptr) == 0;
															});

											if (pathit == (train_paths.data() + train_paths.size())) {
												// allocations galore...fix this
												// std::string p = entry.path().string(); // BAD!
												char* tmp = train_paths.emplace_back(
																(char*)malloc(filepath_tmp.size() + 1));
												std::memcpy(tmp, filepath_tmp.data(), filepath_tmp.size() + 1);
												tmp[filepath_tmp.size()] = 0;
											}
										}
									}
								}
								free(out);
							}
						}

						for (size_t i = 0; i < train_paths.size(); i++) {
							ImGui::TextUnformatted(train_paths[i], nullptr);
						}

						ImGui::EndTabItem();
					}
					if (ImGui::BeginTabItem("Validation Paths")) {
						if (ImGui::Button("Add File##add_validation_file")) {
							nfdu8char_t* out    = nullptr;
							nfdresult_t  result = NFD_OpenDialogU8(&out, nullptr, 0, nullptr);

							if (result == NFD_OKAY) {
								valid_paths.emplace_back(out);
								// free(out);
							}
						}

						if (ImGui::Button("Add Folder##add_validation_folder")) {
							nfdu8char_t* out    = nullptr;
							nfdresult_t  result = NFD_PickFolderU8(&out, nullptr);

							if (result == NFD_OKAY) {
								std::error_code                     iterator_error = {};
								std::filesystem::directory_iterator parent_iterator{out, iterator_error};

								if (iterator_error == std::error_code{}) {
									// paths.clear();
									for (const std::filesystem::directory_entry& entry : parent_iterator) {
										std::error_code status_error = {};
										// std::filesystem::file_status entry_status =
										// std::filesystem::status(entry.path(), status_error);
										std::filesystem::file_status entry_status = entry.status(status_error);
										if (status_error == std::error_code{} &&
														std::filesystem::is_regular_file(entry_status)) {
											string_from(filepath_tmp, entry.path());

											auto pathit = std::find_if(valid_paths.data(),
															valid_paths.data() + valid_paths.size(),
															[&filepath_tmp](char* ptr) {
																return filepath_tmp.compare(ptr) == 0;
															});

											if (pathit == (valid_paths.data() + valid_paths.size())) {
												// allocations galore...fix this
												// std::string p = entry.path().string(); // BAD!
												char* tmp = valid_paths.emplace_back(
																(char*)malloc(filepath_tmp.size() + 1));
												std::memcpy(tmp, filepath_tmp.data(), filepath_tmp.size() + 1);
												tmp[filepath_tmp.size()] = 0;
											}
										}
									}
								}
								free(out);
							}
						}

						for (size_t i = 0; i < valid_paths.size(); i++) {
							ImGui::TextUnformatted(valid_paths[i], nullptr);
						}

						ImGui::EndTabItem();
					}
					if (ImGui::BeginTabItem("Difficult Paths")) {
						if (ImGui::Button("Add File##add_hard_file")) {
							nfdu8char_t* out    = nullptr;
							nfdresult_t  result = NFD_OpenDialogU8(&out, nullptr, 0, nullptr);

							if (result == NFD_OKAY) {
								difficult_paths.emplace_back(out);
								// free(out);
							}
						}

						if (ImGui::Button("Add Folder##add_hard_folder")) {
							nfdu8char_t* out    = nullptr;
							nfdresult_t  result = NFD_PickFolderU8(&out, nullptr);

							if (result == NFD_OKAY) {
								std::error_code                     iterator_error = {};
								std::filesystem::directory_iterator parent_iterator{out, iterator_error};

								if (iterator_error == std::error_code{}) {
									// paths.clear();
									for (const std::filesystem::directory_entry& entry : parent_iterator) {
										std::error_code status_error = {};
										// std::filesystem::file_status entry_status =
										// std::filesystem::status(entry.path(), status_error);
										std::filesystem::file_status entry_status = entry.status(status_error);
										if (status_error == std::error_code{} &&
														std::filesystem::is_regular_file(entry_status)) {
											string_from(filepath_tmp, entry.path());

											auto pathit = std::find_if(difficult_paths.data(),
															difficult_paths.data() + difficult_paths.size(),
															[&filepath_tmp](char* ptr) {
																return filepath_tmp.compare(ptr) == 0;
															});

											if (pathit == (difficult_paths.data() + difficult_paths.size())) {
												// allocations galore...fix this
												// std::string p = entry.path().string(); // BAD!
												char* tmp = difficult_paths.emplace_back(
																(char*)malloc(filepath_tmp.size() + 1));
												std::memcpy(tmp, filepath_tmp.data(), filepath_tmp.size() + 1);
												tmp[filepath_tmp.size()] = 0;
											}
										}
									}
								}
								free(out);
							}
						}

						for (size_t i = 0; i < difficult_paths.size(); i++) {
							ImGui::TextUnformatted(difficult_paths[i], nullptr);
						}

						ImGui::EndTabItem();
					}
					if (ImGui::BeginTabItem("Training")) {
						if (ImGui::Button("Train")) {
							// copy capture as opposed to reference for thread safety's sake
							train_thread = std::jthread{[cfgfile_str, weightfile_str, backupdir_str, train_paths,
																		valid_paths, difficult_paths,
																		from_label_to_label, network_width,
																		network_height, &image_extensions]() {
								char* cfgfile          = (char*)cfgfile_str.data();
								char* weightfile       = weightfile_str.size() ? (char*)weightfile_str.data()
																			   : nullptr; // no training data to start
								char* backup_directory = (char*)backupdir_str.data();
								if (backupdir_str.size() <= 0) {
									return; // bad times...not saving weights, we need to specify a backup directory
								}

								std::vector<char*> t_paths;
								std::vector<char*> v_paths;
								std::vector<char*> d_paths;

								t_paths.reserve(train_paths.size());
								v_paths.reserve(valid_paths.size());
								d_paths.reserve(difficult_paths.size());

								// remove weird extensions
								std::string ext_tmp;
								size_t      ok_count = 0;
								for (size_t i = 0; i < train_paths.size(); i++) {
									std::string_view ext = util::utf8::extension(train_paths[i]);
									ext_tmp.clear();
									ext_tmp.append(ext.data(), ext.size());
									lowercase_str(ext_tmp);

									bool good_extension = is_string_in(ext_tmp, image_extensions);

									t_paths.emplace_back();
									t_paths[ok_count] = train_paths[i];
									ok_count += good_extension;
								}
								t_paths.erase(t_paths.begin() + ok_count, t_paths.begin() + t_paths.size());

								ok_count = 0;
								for (size_t i = 0; i < valid_paths.size(); i++) {
									std::string_view ext = util::utf8::extension(valid_paths[i]);
									ext_tmp.clear();
									ext_tmp.append(ext.data(), ext.size());
									lowercase_str(ext_tmp);

									bool good_extension = is_string_in(ext_tmp, image_extensions);

									v_paths.emplace_back();
									v_paths[ok_count] = valid_paths[i];
									ok_count += good_extension;
								}
								v_paths.erase(v_paths.begin() + ok_count, v_paths.begin() + v_paths.size());

								ok_count = 0;
								for (size_t i = 0; i < difficult_paths.size(); i++) {
									std::string_view ext = util::utf8::extension(difficult_paths[i]);
									ext_tmp.clear();
									ext_tmp.append(ext.data(), ext.size());
									lowercase_str(ext_tmp);

									bool good_extension = is_string_in(ext_tmp, image_extensions);

									d_paths.emplace_back();
									d_paths[ok_count] = difficult_paths[i];
									ok_count += good_extension;
								}
								d_paths.erase(d_paths.begin() + ok_count, d_paths.begin() + d_paths.size());

								std::string  tmp_buf;
								std::string  tmp_out;
								std::string  tmp_line;
								std::fstream ifile_stream;
								std::fstream ofile_stream;

								std::vector<simple_bounding_box> boxes;
								for (size_t i = 0; i < train_paths.size(); i++) {
									ext_tmp.clear();
									ext_tmp.assign(train_paths[i]);

									label_path(tmp_out, tmp_buf, ext_tmp);
									if (std::filesystem::exists(tmp_out)) {
										// has a label file
										continue;
									}

									label_path_old(tmp_out, tmp_buf, ext_tmp);
									if (std::filesystem::exists(tmp_out)) {
										// no "new" label file
										ifile_stream.open(tmp_out, std::ios::in);
										size_t count = 0;
										if (ifile_stream.is_open()) {
											for (; std::getline(ifile_stream, tmp_line);) {
												boxes.emplace_back();
												count += (int)(boxes[count].from_old(tmp_line)) != 0;
											}
											ifile_stream.close();
										}

										ofile_stream.open(tmp_out, std::ios::out | std::ios::trunc);
										if (ofile_stream.is_open()) {
											save_bounding_boxes_to_buffer(tmp_buf, boxes);
											ofile_stream.write(tmp_buf.data(), tmp_buf.size());
											ofile_stream.close();
										}
									}
								}
								/*
								for (std::string line; std::getline(ifile, line);) {
									selected_image->manual_bounding_boxes.emplace_back();
									// only keep bounding boxes we can parse correctly
									std::errc ec = selected_image->manual_bounding_boxes[box_idx].from(line);
									box_idx += ec == std::errc{};
								}
								*/
								char** paths  = (char**)t_paths.data();
								int    npaths = t_paths.size();

								char** validate_paths = v_paths.size() ? (char**)v_paths.data() : (char**)nullptr;
								int    nvalidatepaths = v_paths.size();

								if (validate_paths == nullptr || nvalidatepaths == 0) {
									validate_paths = paths;
									nvalidatepaths = npaths;
								}

								char** hard_paths  = d_paths.size() ? (char**)d_paths.data() : (char**)nullptr;
								int    n_hardpaths = d_paths.size();

								float  iou_thresh       = 0.5f;
								float  thresh           = 0.25f;
								char** labels           = nullptr;
								int    classes          = 0;
								int*   gpus             = 0;
								int    ngpus            = 1;
								int    clear            = 0; // don't start fresh ever?
								int    calc_map         = 0;
								int    benchmark_layers = 0;
								int    dont_show        = 1;
								int    show_imgs        = 0;

								// create new mapping to train against
								std::vector<char*> training_labels;
								for (size_t rule = 0; rule < from_label_to_label.size(); rule++) {
									for (size_t t = 0; t < from_label_to_label[rule].to.size(); t++) {

										bool found_ptr = false;
										for (size_t l = 0; l < training_labels.size(); l++) {
											if (from_label_to_label[rule].to[t].compare(training_labels[l]) == 0) {
												found_ptr = true;
												break;
											}
										}

										if (!found_ptr) {
											training_labels.emplace_back();
											training_labels.back() = (char*)from_label_to_label[rule].to[t].data();
										}
									}
								}

								std::sort(training_labels.begin(), training_labels.end(),
												[](const char* lhs, const char* rhs) {
													std::string_view lv{lhs};
													std::string_view rv{rhs};
													return lv.compare(rv) < 0;
												});

								// pass along label info
								labels  = training_labels.data();
								classes = training_labels.size();

								// use only 1 gpu...don't trust multiple
								int gpu = 0;
								gpus    = &gpu;
								ngpus   = 1;

								yolo_template_opts opts;
								opts.classes = classes;
								// higher reduces memory load
								opts.subdivisions = 64;
								// ?
								opts.batch = 64;
								// lower dimensions reduces memory load
								opts.width  = network_width;  // 608;
								opts.height = network_height; // 608;

								// shoving everything into the backup folder
								std::string darknet_job;
								std::format_to(std::back_inserter(darknet_job), "{}/darknet.data", backup_directory);
								std::string darknet_train;
								std::format_to(std::back_inserter(darknet_train), "{}/train.txt", backup_directory);
								std::string darknet_valid;
								std::format_to(std::back_inserter(darknet_valid), "{}/valid.txt", backup_directory);
								std::string darknet_names;
								std::format_to(std::back_inserter(darknet_names), "{}/darknet.names", backup_directory);

								write_yolo_data_file(darknet_job.data(), ext_tmp, darknet_train, t_paths, darknet_valid,
												v_paths, darknet_names, training_labels, backup_directory);

								std::string tmpbuf;
								write_yolo7_cfg_file(cfgfile, tmpbuf, opts, npaths);

								train_detector_direct(cfgfile, weightfile, backup_directory, paths, npaths,
												validate_paths, nvalidatepaths, hard_paths, n_hardpaths, iou_thresh,
												thresh, labels, classes, gpus, ngpus, clear, calc_map, benchmark_layers,
												dont_show, show_imgs);
							}};
						}

						ImGui::InputInt("Network Height", (int*)&tmp_network_height, 32, 64);
						ImGui::InputInt("Network Width", (int*)&tmp_network_width, 32, 64);

						tmp_network_height = std::clamp(tmp_network_height, 32u, 640u);
						tmp_network_width  = std::clamp(tmp_network_width, 32u, 640u);

						network_height = (tmp_network_height / 32) * 32;
						network_width  = (tmp_network_width / 32) * 32;
						/*
						if ((tmp_network_height % 32) == 0) {
							network_height = tmp_network_height;
						}

						if ((tmp_network_width % 32) == 0) {
							network_width = tmp_network_width;
						}
						*/
						if (ImGui::Button("Default Mapping")) {
							from_label_to_label.clear();
							// from -> to one to one mapping
							for (size_t l = 0; l < labels.size(); l++) {
								label_mapping& back = from_label_to_label.emplace_back();
								back.from           = labels[l];
								back.to.emplace_back(back.from);
							}
						}

						if (ImGui::BeginCombo(
											"From", from_combo_label_dropdown.data(), ImGuiComboFlags_HeightRegular)) {
							for (size_t n = 0; n < labels.size(); n++) {
								imgui_id.clear();
								fmt::format_to(std::back_inserter(imgui_id), "{}##from_{}", labels[n], labels[n]);

								const bool is_selected = (from_combo_label_dropdown.compare(labels[n]) == 0);
								if (ImGui::Selectable(imgui_id.data(), is_selected)) {
									from_combo_idx = n;
									//
									from_combo_label_dropdown.clear();
									from_combo_label_dropdown.append(labels[n].data(), labels[n].size());
								}

								// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
								if (is_selected)
									ImGui::SetItemDefaultFocus();
							}
							ImGui::EndCombo();
						}

						if (ImGui::BeginCombo("To", to_combo_label_dropdown.data(), ImGuiComboFlags_HeightRegular)) {
							for (size_t n = 0; n < labels.size(); n++) {
								imgui_id.clear();
								fmt::format_to(std::back_inserter(imgui_id), "{}##to_{}", labels[n], labels[n]);

								const bool is_selected = (to_combo_label_dropdown.compare(labels[n]) == 0);
								if (ImGui::Selectable(
													imgui_id.data(), imgui_id.data() + imgui_id.size(), is_selected)) {
									to_combo_idx = n;
									//
									to_combo_label_dropdown.clear();
									to_combo_label_dropdown.append(labels[n].data(), labels[n].size());
								}

								// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
								if (is_selected)
									ImGui::SetItemDefaultFocus();
							}
							ImGui::EndCombo();
						}

						if (ImGui::Button("Add Mapping")) {
							bool found_from = false;
							for (size_t rule = 0; rule < from_label_to_label.size(); rule++) {
								if (from_combo_label_dropdown.compare(from_label_to_label[rule].from) == 0) {
									found_from = true;
									//
									bool found_to = false;
									for (size_t t = 0; t < from_label_to_label[rule].to.size(); t++) {
										if (to_combo_label_dropdown.compare(from_label_to_label[rule].to[t]) == 0) {
											found_to = true;
											break;
										}
									}
									if (!found_to) {
										std::string& to = from_label_to_label[rule].to.emplace_back();
										to.reserve(to_combo_label_dropdown.size());
										to.append(to_combo_label_dropdown.data(), to_combo_label_dropdown.size());
									}
									break;
								}
							}

							if (!found_from) {
								label_mapping& back = from_label_to_label.emplace_back();
								back.from.reserve(from_combo_label_dropdown.size());
								back.from.append(from_combo_label_dropdown.data(), from_combo_label_dropdown.size());
								std::string& to = back.to.emplace_back();
								to.reserve(to_combo_label_dropdown.size());
								to.append(to_combo_label_dropdown.data(), to_combo_label_dropdown.size());
							}
						}

						for (size_t rule = 0; rule < from_label_to_label.size(); rule++) {
							ImGui::PushID(rule);

#if 0
					if (ImGui::BeginCombo("From", from_label_to_label[rule].from.data(), ImGuiComboFlags_HeightRegular)) {
						for (size_t n = 0; n < labels.size(); n++) {
							imgui_id.clear();
							fmt::format_to(std::back_inserter(imgui_id), "{}##from_{}", labels[n], labels[n]);

							const bool is_selected = (from_label_to_label[rule].from.compare(labels[n]) == 0);
							if (ImGui::Selectable(imgui_id.data(), is_selected)) {
								from_combo_idx = n;
								//
								from_label_to_label[rule].from.clear();
								from_label_to_label[rule].from.append(labels[n].data(), labels[n].size());
							}

							// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
							if (is_selected)
								ImGui::SetItemDefaultFocus();
						}
						ImGui::EndCombo();
					}
#endif
							imgui_id.clear();
							fmt::format_to(std::back_inserter(imgui_id), "{}##from_{}", from_label_to_label[rule].from,
											from_label_to_label[rule].from);
							if (ImGui::TreeNode(imgui_id.data())) {
								// different column?
								for (size_t t = 0; t < from_label_to_label[rule].to.size(); t++) {
									ImGui::PushID(t);
									if (ImGui::BeginCombo("To", from_label_to_label[rule].to[t].data(),
														ImGuiComboFlags_HeightRegular)) {
										for (size_t n = 0; n < labels.size(); n++) {
											imgui_id.clear();
											fmt::format_to(std::back_inserter(imgui_id), "{}##to_{}", labels[n],
															labels[n]);

											const bool is_selected =
															(from_label_to_label[rule].to[t].compare(labels[n]) == 0);
											if (ImGui::Selectable(imgui_id.data(), is_selected)) {
												to_combo_idx = n;
												//
												from_label_to_label[rule].to[t].clear();
												from_label_to_label[rule].to[t].append(
																labels[n].data(), labels[n].size());
											}

											// Set the initial focus when opening the combo (scrolling + keyboard
											// navigation focus)
											if (is_selected)
												ImGui::SetItemDefaultFocus();
										}
										ImGui::EndCombo();
									}

									bool found_duplicate = false;
									for (size_t s = 0; s < t; s++) {
										if (from_label_to_label[rule].to[t].compare(from_label_to_label[rule].to[s]) ==
														0) {
											found_duplicate = true;
											break;
										}
									}

									if (!found_duplicate) {
										for (size_t s = t + 1; s < from_label_to_label[rule].to.size(); s++) {
											if (from_label_to_label[rule].to[t].compare(
																from_label_to_label[rule].to[s]) == 0) {
												found_duplicate = true;
												break;
											}
										}
									}

									ImGui::SameLine();
									bool remove = ImGui::Button("(-) mapping");
									if (found_duplicate || remove) {
										from_label_to_label[rule].to.erase(from_label_to_label[rule].to.begin() + t);
										t--;
									}
									ImGui::PopID();
								}
								ImGui::TreePop();
							}
							ImGui::PopID();
						}
						ImGui::EndTabItem();
					}
					if (ImGui::BeginTabItem("Detecting")) {
						ImGui::SliderFloat("Detector Threshold", (float*)&detector_thresh, 0.0f, 1.00f);
						ImGui::SliderFloat("Detector Visible Threshold", (float*)&detector_visible_thresh,
										std::max(0.0f, detector_thresh), 1.00f);
						ImGui::EndTabItem();
					}
					ImGui::EndTabBar();
				}
			}
			ImGui::End();
		}
// if (ImGui::Begin()) {
// }

//}
// ImGui::End();

/*
for (GLuint current_texture = 1; current_texture < last_texture; current_texture++) {
	if (ImGui::Begin("Image")) {
		ImGui::Text("pointer = %p", my_image_texture);
		ImGui::Text("size = %d x %d", my_image_width, my_image_height);
		ImGui::Image((void*)(intptr_t)my_image_texture, ImVec2(my_image_width,
my_image_height)); ImGui::End();
	}
}
*/
#if _DEBUG
		if (home_pressed)
			ImGui::SetNextWindowPos(ImVec2{50, 50});
		ImGui::ShowDemoWindow(&show_demo_window);
#endif

		if (home_pressed)
			ImGui::SetNextWindowPos(ImVec2{40, 40});
		if (ImGui::Begin("Style Editor")) {
			ImGui::ShowStyleEditor();
		}
		ImGui::End();

		ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
		// Rendering
		ImGui::Render();
		int display_w, display_h;
		glfwGetFramebufferSize(r.window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w,
						clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(r.window);
	}

	if (custom_detector) {
		free_network(*custom_detector);
		free_network_ptr(custom_detector);
	}

	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(r.window);
	glfwTerminate();

	return r.err_code;
}

// boilerplate ~for windows build~
int WinMain(int argc, char** argv)
{
	return main(argc, argv);
}