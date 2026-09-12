module kong;

import std::io;
import std::io::file;
import std::collections::list;
import std::core::mem::allocators;
import std::math::matrix;
import std::math::vector;

import vk, camera, c3w, shady;

const WIDTH                = 800;
const HEIGHT               = 600;
const VK_API_VERSION_1_3   = vk::@makeApiVersion(0, 1, 3, 0);
const MAX_FRAMES_IN_FLIGHT = 2;

const SHADER = `
// A shady shader (lib/shady.c3l), compiled to SPIR-V at runtime, in process.
//
// The vertices are not a vertex buffer. @address means this struct is reached
// through a 64-bit device address, handed in as a push constant, and the shader
// indexes it itself using @builtin(vertex_index). The pipeline below declares
// no vertex input state at all.
struct Vertex @address
{
    float3 position;
    float4 color;
    float2 uv;
}

struct Push @pushconstant
{
    Vertex* vertices;
}

struct Uniforms @uniform
{
    float4x4 projection;
    float4x4 view;
}

// A module-level variable is just a type followed by a name - what kind of
// resource it is comes from the type. Descriptors are numbered in declaration
// order, so they land on bindings 0, 1 and 2 of set 0; the push constant block
// takes no binding at all.
Push pc;
Uniforms uniforms;
texture2d tex;
sampler tex_sampler;

struct VertexIn
{
    uint index @builtin(vertex_index);
}

// @position is the clip position: BuiltIn Position on the way out of the vertex
// stage, BuiltIn FragCoord on the way into the fragment stage. Every other
// member gets a Location, numbered in declaration order.
struct FragmentIn
{
    float4 position @position;
    float4 color;
    float2 uv;
}

fn FragmentIn vert(VertexIn input) @vertex
{
    Vertex* v = pc.vertices;

    FragmentIn output;
    output.position = uniforms.projection * uniforms.view * float4(v[input.index].position, 1.0);
    output.color = v[input.index].color;
    output.uv = v[input.index].uv;
    return output;
}

fn float4 frag(FragmentIn input) @fragment
{
    return tex.sample(tex_sampler, input.uv) * input.color;
}
`;

// The GPU reads this through a device address, so the layout is not ours to
// choose: it has to be the std430 layout the shader computed for `Vertex`,
// which is offsets 0, 16, 32 and a stride of 48. float3 occupies 12
// bytes but aligns to 16, which is where the first gap comes from; the second
// is the struct's own 16-byte alignment rounding 40 up to 48.
//
// Plain arrays rather than Vec3/Vec4: those are float[<N>] vectors whose own
// alignment would pad the struct behind our back.
struct Vertex
{
	float[3] pos;
	float    _pad0;
	float[4] color;
	float[2] uv;
	float[2] _pad1;
}

// The shader says ArrayStride 48. If these ever disagree the cube renders as
// garbage, so disagree at compile time instead.
$assert Vertex::size == 48 : "Vertex must match the shader's std430 layout";

const TEXTURE_SIZE = 64;

// A checkerboard, generated rather than loaded, so the example needs no image
// decoder to show that sampling works.
char[TEXTURE_SIZE * TEXTURE_SIZE * 4] texture_pixels;

fn char[] checkerboard()
{
	char[] pixels = texture_pixels[..];
	for (int y = 0; y < TEXTURE_SIZE; y++)
	{
		for (int x = 0; x < TEXTURE_SIZE; x++)
		{
			bool light  = ((x / 8) + (y / 8)) % 2 == 0;
			usz  offset = (usz)(y * TEXTURE_SIZE + x) * 4;
			pixels[offset]     = light ? 240 : 30;
			pixels[offset + 1] = light ? 240 : (char)(x * 4);
			pixels[offset + 2] = light ? 240 : (char)(y * 4);
			pixels[offset + 3] = 255;
		}
	}
	return pixels;
}

// Per-face tint, multiplied with the sampled texel in the fragment shader.
Vertex[] vertices = {
	{ .pos = { -1.0, -1.0, 1.0 }, .color = { 1.0, 0.45, 0.45, 1.0 }, .uv = { 0.0, 0.0 } },
	{ .pos = { 1.0, -1.0, 1.0 }, .color = { 1.0, 0.45, 0.45, 1.0 }, .uv = { 1.0, 0.0 } },
	{ .pos = { 1.0, 1.0, 1.0 }, .color = { 1.0, 0.45, 0.45, 1.0 }, .uv = { 1.0, 1.0 } },
	{ .pos = { -1.0, 1.0, 1.0 }, .color = { 1.0, 0.45, 0.45, 1.0 }, .uv = { 0.0, 1.0 } },  // front
	{ .pos = { -1.0, -1.0, -1.0 }, .color = { 1.0, 1.0, 1.0, 1.0 }, .uv = { 0.0, 0.0 } },
	{ .pos = { -1.0, 1.0, -1.0 }, .color = { 1.0, 1.0, 1.0, 1.0 }, .uv = { 1.0, 0.0 } },
	{ .pos = { 1.0, 1.0, -1.0 }, .color = { 1.0, 1.0, 1.0, 1.0 }, .uv = { 1.0, 1.0 } },
	{ .pos = { 1.0, -1.0, -1.0 }, .color = { 1.0, 1.0, 1.0, 1.0 }, .uv = { 0.0, 1.0 } },  // back
	{ .pos = { -1.0, 1.0, -1.0 }, .color = { 0.55, 0.75, 1.0, 1.0 }, .uv = { 0.0, 0.0 } },
	{ .pos = { -1.0, 1.0, 1.0 }, .color = { 0.55, 0.75, 1.0, 1.0 }, .uv = { 1.0, 0.0 } },
	{ .pos = { 1.0, 1.0, 1.0 }, .color = { 0.55, 0.75, 1.0, 1.0 }, .uv = { 1.0, 1.0 } },
	{ .pos = { 1.0, 1.0, -1.0 }, .color = { 0.55, 0.75, 1.0, 1.0 }, .uv = { 0.0, 1.0 } },  // top
	{ .pos = { -1.0, -1.0, -1.0 }, .color = { 0.35, 0.6, 1.0, 1.0 }, .uv = { 0.0, 0.0 } },
	{ .pos = { 1.0, -1.0, -1.0 }, .color = { 0.35, 0.6, 1.0, 1.0 }, .uv = { 1.0, 0.0 } },
	{ .pos = { 1.0, -1.0, 1.0 }, .color = { 0.35, 0.6, 1.0, 1.0 }, .uv = { 1.0, 1.0 } },
	{ .pos = { -1.0, -1.0, 1.0 }, .color = { 0.35, 0.6, 1.0, 1.0 }, .uv = { 0.0, 1.0 } },  // bottom
	{ .pos = { 1.0, -1.0, -1.0 }, .color = { 0.5, 1.0, 0.55, 1.0 }, .uv = { 0.0, 0.0 } },
	{ .pos = { 1.0, 1.0, -1.0 }, .color = { 0.5, 1.0, 0.55, 1.0 }, .uv = { 1.0, 0.0 } },
	{ .pos = { 1.0, 1.0, 1.0 }, .color = { 0.5, 1.0, 0.55, 1.0 }, .uv = { 1.0, 1.0 } },
	{ .pos = { 1.0, -1.0, 1.0 }, .color = { 0.5, 1.0, 0.55, 1.0 }, .uv = { 0.0, 1.0 } },  // right
	{ .pos = { -1.0, -1.0, -1.0 }, .color = { 0.8, 0.8, 0.4, 1.0 }, .uv = { 0.0, 0.0 } },
	{ .pos = { -1.0, -1.0, 1.0 }, .color = { 0.8, 0.8, 0.4, 1.0 }, .uv = { 1.0, 0.0 } },
	{ .pos = { -1.0, 1.0, 1.0 }, .color = { 0.8, 0.8, 0.4, 1.0 }, .uv = { 1.0, 1.0 } },
	{ .pos = { -1.0, 1.0, -1.0 }, .color = { 0.8, 0.8, 0.4, 1.0 }, .uv = { 0.0, 1.0 } },  // left
	};

ushort[] indices = {
	0,
	1,
	2,
	0,
	2,
	3, // front
	4,
	5,
	6,
	4,
	6,
	7, // back
	8,
	9,
	10,
	8,
	10,
	11, // top
	12,
	13,
	14,
	12,
	14,
	15, // bottom
	16,
	17,
	18,
	16,
	18,
	19, // right
	20,
	21,
	22,
	20,
	22,
	23 // left
};

// The sky's push block, byte for byte the C3 struct three.c3's generator walks
// (`SkyPush`): the inverse view-projection, then x intensity, y rotation in
// radians, zw unused.
struct SkyPush
{
	Matrix4f inverse_view_projection;
	float[4] params;
}

$assert SkyPush::size == 80 : "SkyPush must stay the 80 bytes the def emits";

// The camera and the sky move together: both are the same matrix, the sky's
// un-projected. One function so a camera change cannot move the cube and leave
// the sky looking somewhere else.
fn void update_camera(Camera* camera, Memory* uniform_buffer, SkyPush* push)
{
	UniformData data = camera.matrix();
	uniform_buffer.upload(&data, uniform_buffer.size);

	Matrix4f? inverted = (data.projection * data.view).inverse();
	if (catch inverted) return;
	push.inverse_view_projection = inverted;
}

// Copy one presented image back to the host and write it as a binary PPM.
//
// PPM rather than PNG because it is a header and the bytes: nothing in this
// program has an encoder, and the point is to be able to *look* at what a
// frame contained rather than to ship an image format.
fn void capture_frame(
	Device device,
	DeviceQueue queue,
	vk::MemoryAllocator* allocator,
	vk::Swapchain* sc,
	uint image,
	String path
)
{
	ImageUsageFlagBits transfer_src = IMAGE_USAGE_TRANSFER_SRC_BIT;
	if ((sc.image_usage & transfer_src) == (ImageUsageFlagBits)0)
	{
		io::eprintfn("capture: the surface did not grant TRANSFER_SRC (usage %x)", sc.image_usage);
		return;
	}

	usz pixels = (usz)WIDTH * (usz)HEIGHT;
	vk::Memory readback = vk::new_buffer(
		allocator,
		usage: BUFFER_USAGE_TRANSFER_DST_BIT,
		properties: MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT,
		data: null,
		data_size: (sz)pixels * 4,
	)!!;
	defer readback.free();

	device.@single_time_command(queue: queue; CommandBuffer cmd)
	{
		vk::ImageMemoryBarrier2 to_src = vk::transferBarrier(
			sc.images[image],
			IMAGE_LAYOUT_PRESENT_SRC_KHR,
			IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
		);
		vk::cmdPipelineBarrier2(cmd, &&vk::dependencyInfo().setImageMemoryBarriers({ to_src }));

		vk::BufferImageCopy region = {
			.imageSubresource = {
				.aspectMask = IMAGE_ASPECT_COLOR_BIT,
				.mipLevel = 0,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
			.imageExtent = { WIDTH, HEIGHT, 1 },
		};
		vk::cmdCopyImageToBuffer(
			cmd,
			sc.images[image],
			IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			readback.buffer,
			1,
			&region
		);

		// Back where the presentation engine expects it. Leaving it in
		// TRANSFER_SRC is an error the next acquire reports instead.
		vk::ImageMemoryBarrier2 to_present = vk::transferBarrier(
			sc.images[image],
			IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			IMAGE_LAYOUT_PRESENT_SRC_KHR
		);
		vk::cmdPipelineBarrier2(cmd, &&vk::dependencyInfo().setImageMemoryBarriers({ to_present }));
	}!!;

	// `B8G8R8A8` is what the surface format is on this platform: the bytes are
	// blue, green, red, alpha and a PPM wants red, green, blue.
	String header = string::tformat("P6\n%d %d\n255\n", WIDTH, HEIGHT);
	char[] out = alloc::new_array(mem, char, (sz)(header.len + pixels * 3));
	mem::copy(out.ptr, header.ptr, header.len);

	char* source = (char*)readback.data();
	for (usz i = 0; i < pixels; i++)
	{
		out[header.len + i * 3]     = source[i * 4 + 2];
		out[header.len + i * 3 + 1] = source[i * 4 + 1];
		out[header.len + i * 3 + 2] = source[i * 4];
	}
	file::save(path, out)!!;
	free(out);
	io::printfn("capture: wrote %s", path);
}

fn void main(String[] args)
{

	// `--frames N` renders N frames and exits, so a run can be checked without
	// a hand on the Escape key. The window still opens: this is a renderer
	// test, not a headless one.
	uint frame_limit = 0;
	String capture_path = "";
	for (usz i = 0; i < args.len; i++)
	{
		if (i + 1 >= args.len) continue;
		if (args[i] == "--frames")
		{
			int? count = args[i + 1].to_int();
			if (catch count) continue;
			frame_limit = (uint)count;
		}
		else if (args[i] == "--capture")
		{
			capture_path = args[i + 1];
		}
	}

	// First, before Vulkan or the window system exist. A shady compile error is
	// a fault carrying a line and a column, so report it and leave rather than
	// standing up a device for a pipeline that cannot be built.
	shady::Diagnostic diagnostic;
	char[]? compiled = shady::compile(SHADER, &diagnostic);
	if (catch compiled)
	{
		io::eprintfn("shader error at %d:%d: %s", diagnostic.line, diagnostic.column, diagnostic.message);
		diagnostic.free();
		return;
	}
	diagnostic.free();
	char[] spirv = compiled;
	defer free(spirv);

	c3w::Window win = c3w::new(params: { .width = WIDTH, .height = HEIGHT, .x = 0, .y = 0 }, name: "Cube")!!;
	defer win.free();

	Camera camera = camera::new(eye: { 0.0, 5.0, 10 }, aspect: WIDTH / HEIGHT, min_zoom: 0.01, max_zoom: 1000);

	ZStringList device_extensions;
	device_extensions.push_all({ "VK_KHR_swapchain", "VK_KHR_push_descriptor", "VK_KHR_shader_non_semantic_info" });

	vk::init()!!;

	vk::Instance instance = vk::createDefaultInstance(
		{
			.enable_validation = false,
			.extensions = { ...vk::DEFAULT_EXTENSIONS, "VK_KHR_surface" },
			.app_name = "test"
		}
	)!!;

	// Create surface. The window library gets our vkGetInstanceProcAddr: it has
	// no way to find the loader this program opened by itself.
	SurfaceKHR surface = (SurfaceKHR)win.create_vk_surface(
		instance,
		(c3w::VkGetInstanceProcAddrFn)vk::getInstanceProcAddr
	)!!;

	// --- Physical device selection ---
	vk::DeviceSelection selection  = vk::selectPhysicalDevice(instance, surface, device_extensions)!!;
	PhysicalDevice physical_device = selection.device;
	DeviceQueue graphics_queue     = { .index = selection.graphics_queue_index };

	// --- Logical device ---
	PhysicalDeviceVulkan13Features vulkan13Features = {
		.sType = STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		.pNext = null,
	};

	PhysicalDeviceVulkan12Features vulkan12Features = {
		.sType = STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.pNext = &vulkan13Features,
	};

	PhysicalDeviceVulkan11Features vulkan11Features = {
		.sType = STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
		.pNext = &vulkan12Features,
	};

	PhysicalDeviceFeatures2 device_features = {
		.sType = STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = &vulkan11Features,
	};

	vk::getPhysicalDeviceFeatures2(physical_device, &device_features);

	DeviceQueueCreateInfo graphicsInfo = vk::deviceQueueCreateInfo().setQueueFamilyIndex(graphics_queue.index)
		.setQueuePriorities({ 1.0 });

	Device device = vk::deviceCreateInfo().setQueueCreateInfos({ graphicsInfo })
		.setEnabledExtensionNames(device_extensions.array_view())
		.setNext(&device_features)
		.build(physical_device)!!;

	defer
	{
		device.free();
		instance.free();
	}

	graphics_queue.handle = device.getQueue(graphics_queue.index);

	// --- Render pass ---
	AttachmentReference[] colorAttachments = { { .attachment = 0, .layout = IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL } };

	SubpassDescription subpass = {
		.pipelineBindPoint = PIPELINE_BIND_POINT_GRAPHICS,
		.colorAttachmentCount = colorAttachments.len,
		.pColorAttachments = colorAttachments.ptr
	};

	RenderPass render_pass = vk::renderPassCreateInfo()
		.setAttachments(
			{
				{
					.format = FORMAT_B8G8R8A8_UNORM,
					.samples = SAMPLE_COUNT_1_BIT,
					.initialLayout = IMAGE_LAYOUT_UNDEFINED,
					.finalLayout = IMAGE_LAYOUT_PRESENT_SRC_KHR,
					.loadOp = ATTACHMENT_LOAD_OP_CLEAR,
					.storeOp = ATTACHMENT_STORE_OP_STORE,
					.stencilLoadOp = ATTACHMENT_LOAD_OP_DONT_CARE,
					.stencilStoreOp = ATTACHMENT_STORE_OP_DONT_CARE
				}
			}
		)
		.setSubpasses({ subpass })
		.build(device)!!;

	// --- Swapchain (with framebuffers) ---
	vk::Swapchain sc;
	sc.init(
		device,
		physical_device,
		surface,
		WIDTH,
		HEIGHT,
		render_pass: render_pass,
		image_usage: IMAGE_USAGE_COLOR_ATTACHMENT_BIT | IMAGE_USAGE_TRANSFER_SRC_BIT
	)!!;

	// --- Command pool & buffers ---
	CommandPool command_pool          = device.createCommandPool(queue: graphics_queue.index)!!;
	CommandBufferList command_buffers = device.createCommandBuffers(command_pool, sc.image_count)!!;

	// --- Memory allocator & buffers ---
	vk::MemoryAllocator allocator = (vk::MemoryAllocator) {
		.device = device,
		.pdevice = physical_device,
		.queue = graphics_queue,
		.flags = MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT | MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT,
	}.init()!!;

	// SHADER_DEVICE_ADDRESS_BIT is what makes new_buffer ask Vulkan for the
	// address; without it Memory.get_address() is zero. It is not a vertex
	// buffer any more - nothing binds it as one.
	Memory vertex_buffer = vk::new_buffer(
		allocator: &allocator,
		usage: BUFFER_USAGE_STORAGE_BUFFER_BIT | BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		properties: MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		data: vertices.ptr,
		data_size: Vertex::size * vertices.len
	)!!;

	DeviceAddress vertices_address = vertex_buffer.get_address();

	Memory index_buffer = vk::new_buffer(
		allocator: &allocator,
		usage: BUFFER_USAGE_INDEX_BUFFER_BIT,
		properties: MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		data: indices.ptr,
		data_size: indices.len * ushort::size
	)!!;

	Memory uniform_buffer = vk::new_buffer(
		allocator: &allocator,
		usage: BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		properties: MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_CACHED_BIT,
		data: &&camera.matrix(),
		data_size: UniformData::size
	)!!;

	// --- Texture ---
	char[] pixels = checkerboard();

	Image texture_image = vk::imageCreateInfo().setImageType(IMAGE_TYPE_2D)
		.setFormat(FORMAT_R8G8B8A8_UNORM)
		.setExtent({ .width = TEXTURE_SIZE, .height = TEXTURE_SIZE, .depth = 1 })
		.setMipLevels(1)
		.setArrayLayers(1)
		.setSamples(SAMPLE_COUNT_1_BIT)
		.setTiling(IMAGE_TILING_OPTIMAL)
		.setUsage(IMAGE_USAGE_TRANSFER_DST_BIT | IMAGE_USAGE_SAMPLED_BIT)
		.setInitialLayout(IMAGE_LAYOUT_UNDEFINED)
		.build(device)!!;

	Memory texture_memory = vk::create_image_buffer(texture_image, &allocator, MEMORY_PROPERTY_DEVICE_LOCAL_BIT)!!;
	Memory texture_stage  = vk::new_buffer(
		allocator: &allocator,
		usage: BUFFER_USAGE_TRANSFER_SRC_BIT,
		properties: MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT,
		data: pixels.ptr,
		data_size: pixels.len
	)!!;

	ImageSubresourceRange whole_image = {
		.aspectMask = IMAGE_ASPECT_COLOR_BIT,
		.baseMipLevel = 0,
		.levelCount = 1,
		.baseArrayLayer = 0,
		.layerCount = 1,
	};

	device.@single_time_command(queue: graphics_queue; CommandBuffer cmd)
	{
		ImageMemoryBarrier2 to_transfer = {
			.sType = STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.image = texture_image,
			.srcStageMask = PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
			.dstStageMask = PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
			.srcAccessMask = (AccessFlags2)0,
			.dstAccessMask = ACCESS_2_TRANSFER_WRITE_BIT,
			.oldLayout = IMAGE_LAYOUT_UNDEFINED,
			.newLayout = IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.subresourceRange = whole_image,
		};
		vk::cmdPipelineBarrier2(cmd, &&vk::dependencyInfo().setImageMemoryBarriers({ to_transfer }));

		BufferImageCopy region = {
			.imageSubresource = {
				.aspectMask = IMAGE_ASPECT_COLOR_BIT,
				.mipLevel = 0,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
			.imageExtent = { TEXTURE_SIZE, TEXTURE_SIZE, 1 },
		};
		vk::cmdCopyBufferToImage(cmd, texture_stage.buffer, texture_image, IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		ImageMemoryBarrier2 to_shader = vk::transferBarrier(
			texture_image,
			IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		);
		vk::cmdPipelineBarrier2(cmd, &&vk::dependencyInfo().setImageMemoryBarriers({ to_shader }));
	}!!;

	texture_stage.free();

	ImageView texture_view = vk::imageViewCreateInfo().setViewType(IMAGE_VIEW_TYPE_2D)
		.setFormat(FORMAT_R8G8B8A8_UNORM)
		.setSubresourceRange(whole_image)
		.setImage(texture_image)
		.build(device)!!;

	// NEAREST keeps the checker edges crisp, so it is obvious the texture is
	// really being sampled rather than smeared into a flat colour.
	Sampler texture_sampler = vk::samplerCreateInfo().setMagFilter(FILTER_NEAREST)
		.setMinFilter(FILTER_NEAREST)
		.setAddressModeU(SAMPLER_ADDRESS_MODE_REPEAT)
		.setAddressModeV(SAMPLER_ADDRESS_MODE_REPEAT)
		.setAddressModeW(SAMPLER_ADDRESS_MODE_REPEAT)
		.setMipmapMode(SAMPLER_MIPMAP_MODE_NEAREST)
		.build(device)!!;

	// --- Descriptors ---
	// Bindings 0/1/2 are the ones shady auto-assigned to uniforms, tex and
	// tex_sampler, in the order they are declared in the shader.
	DescriptorSetLayout descriptor_set_layout = vk::descriptorSetLayoutCreateInfo()
		.setFlags(DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT)
		.setBindings(
			{
				{
					.binding = 0,
					.descriptorType = DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = SHADER_STAGE_VERTEX_BIT
				},
				{
					.binding = 1,
					.descriptorType = DESCRIPTOR_TYPE_SAMPLED_IMAGE,
					.descriptorCount = 1,
					.stageFlags = SHADER_STAGE_FRAGMENT_BIT
				},
				{
					.binding = 2,
					.descriptorType = DESCRIPTOR_TYPE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags = SHADER_STAGE_FRAGMENT_BIT
				}
			}
		)
		.build(device)!!;

	WriteDescriptorSet[] descriptorSets = {
		vk::writeDescriptorSet()
			.setDstBinding(0)
			.setDescriptorType(DESCRIPTOR_TYPE_UNIFORM_BUFFER)
			.setBufferInfo({ { .buffer = uniform_buffer.buffer, .offset = 0, .range = UniformData::size } }),
		vk::writeDescriptorSet()
			.setDstBinding(1)
			.setDescriptorType(DESCRIPTOR_TYPE_SAMPLED_IMAGE)
			.setImageInfo({ { .imageView = texture_view, .imageLayout = IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL } }),
		vk::writeDescriptorSet()
			.setDstBinding(2)
			.setDescriptorType(DESCRIPTOR_TYPE_SAMPLER)
			.setImageInfo({ { .sampler = texture_sampler } })
	};


	// --- Shader & pipeline ---
	// One module holding both stages, told apart by entry point name.
	ShaderModule shader_module = vk::loadShaderModule(device, spirv)!!;
	defer shader_module.free(device);

	// One 8-byte push constant: the address of the vertex buffer. This is the
	// whole of the `Push pc` block the shader declares.
	PushConstantRange[1] push_constant_ranges = {
		{ .stageFlags = SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = DeviceAddress::size }
	};

	PipelineLayout pipeline_layout = vk::pipelineLayoutCreateInfo().setSetLayouts({ descriptor_set_layout })
		.setPushConstantRanges(push_constant_ranges[..])
		.build(device)!!;

	// The entry point names are the shader's own function names.
	PipelineShaderStageCreateInfo[2] shader_stages = { vk::pipelineShaderStageCreateInfo()
		.setStage(SHADER_STAGE_VERTEX_BIT)
		.setMod(shader_module)
		.setName("vert"), vk::pipelineShaderStageCreateInfo()
			.setStage(SHADER_STAGE_FRAGMENT_BIT)
			.setMod(shader_module)
			.setName("frag") };

	// No bindings and no attributes: the vertex shader pulls what it needs
	// through the device address, so there is nothing for the input assembler
	// to fetch.
	PipelineVertexInputStateCreateInfo vertex_input_state = vk::pipelineVertexInputStateCreateInfo();

	// Use default pipeline states — caller owns array data to avoid dangling pointers
	PipelineRasterizationStateCreateInfo rasterization_state = vk::defaultRasterizationState()
		.setCullMode(CULL_MODE_FRONT_BIT);
	PipelineInputAssemblyStateCreateInfo input_assembly_state = vk::pipelineInputAssemblyStateCreateInfo()
		.setTopology(PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

	PipelineMultisampleStateCreateInfo multisample_state    = vk::defaultMultisampleState();
	PipelineDepthStencilStateCreateInfo depth_stencil_state = vk::defaultDepthStencilState();

	PipelineColorBlendAttachmentState[1] blend_attachments = { vk::defaultColorBlendAttachment() };
	PipelineColorBlendStateCreateInfo color_blend_state    = vk::defaultColorBlendState(blend_attachments[..]);

	DynamicState[2] dynamic_states = { DYNAMIC_STATE_VIEWPORT, DYNAMIC_STATE_LINE_WIDTH };

	PipelineDynamicStateCreateInfo dynamic_state = vk::pipelineDynamicStateCreateInfo()
		.setDynamicStates(dynamic_states[..]);

	Viewport[1] viewports = {
		{ .width = sc.extent.width, .height = sc.extent.height, .minDepth = 0.0, .maxDepth = 1.0 }
	};
	Rect2D[1] scissors = { { .extent = sc.extent, .offset = { 0, 0 } } };

	PipelineViewportStateCreateInfo viewport_state = vk::pipelineViewportStateCreateInfo().setViewports(viewports[..])
		.setScissors(scissors[..]);

	Pipeline pipeline = vk::graphicsPipelineCreateInfo().setStages(shader_stages[..]).setLayout(pipeline_layout)
		.setVertexInputState(&vertex_input_state)
		.setDynamicState(&dynamic_state)
		.setRasterizationState(&rasterization_state)
		.setInputAssemblyState(&input_assembly_state)
		.setMultisampleState(&multisample_state)
		.setDepthStencilState(&depth_stencil_state)
		.setColorBlendState(&color_blend_state)
		.setViewportState(&viewport_state)
		.setRenderPass(render_pass)
		.build(device)!!;

	// --- Sky: a module three.c3's ShaderDef generated ---
	//
	// three.c3 writes the whole shader - the declarations walked from its wire
	// structs, the push block, the entry points, the ported body - and this is
	// where it is compiled *and drawn*, because kong depends on shady and the
	// engine does not. Regenerate it with:
	//
	//     cd ../three.c3 && c3c test --trust=full --test-filter three_tests::shader_def
	//
	// A missing module is not fatal: the cube is the demo, the sky is the port
	// being checked.
	const String SKY_MODULE = "../three.c3/build/shaderdef/sky.shady";

	ShaderModule sky_module = (ShaderModule)null;
	Pipeline sky_pipeline = (Pipeline)null;
	PipelineLayout sky_layout = (PipelineLayout)null;
	DescriptorSetLayout sky_set_layout = (DescriptorSetLayout)null;
	WriteDescriptorSet sky_descriptor;
	SkyPush sky_push = { .params = { 1.0, 0.0, 0.0, 0.0 } };

	if (!file::is_file(SKY_MODULE))
	{
		io::printn("sky: no generated module - run three.c3's shader_def test first");
	}
	else
	{
		char[]? sky_bytes = file::load(tmem, SKY_MODULE);
		if (catch sky_bytes)
		{
			io::eprintfn("sky: cannot read %s", SKY_MODULE);
		}
		else
		{
			shady::Diagnostic sky_diagnostic;
			char[]? sky_spirv = shady::compile((String)sky_bytes, &sky_diagnostic);
			if (catch sky_spirv)
			{
				io::eprintfn("sky module error at %d:%d: %s",
					sky_diagnostic.line, sky_diagnostic.column, sky_diagnostic.message);
			}
			else
			{
				sky_module = vk::loadShaderModule(device, sky_spirv)!!;

				// One combined image sampler at set 0, binding 0 - the one the
				// def's `ShaderResource` declared, which is the only binding the
				// module has.
				sky_set_layout = vk::descriptorSetLayoutCreateInfo()
					.setFlags(DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT)
					.setBindings(
						{
							{
								.binding = 0,
								.descriptorType = DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
								.descriptorCount = 1,
								.stageFlags = SHADER_STAGE_FRAGMENT_BIT
							}
						}
					)
					.build(device)!!;

				PipelineShaderStageCreateInfo[2] sky_stages = {
					vk::pipelineShaderStageCreateInfo()
						.setStage(SHADER_STAGE_VERTEX_BIT)
						.setMod(sky_module)
						.setName("vertexMain"),
					vk::pipelineShaderStageCreateInfo()
						.setStage(SHADER_STAGE_FRAGMENT_BIT)
						.setMod(sky_module)
						.setName("fragmentMain")
				};

				sky_layout = vk::pipelineLayoutCreateInfo()
					.setSetLayouts({ sky_set_layout })
					.setPushConstantRanges(
						{ { .stageFlags = SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = SkyPush::size } }
					)
					.build(device)!!;

				// The oversized triangle is not wound for a face, and there is no
				// depth attachment to test against.
				PipelineRasterizationStateCreateInfo sky_raster = vk::defaultRasterizationState()
					.setCullMode(CULL_MODE_NONE);
				PipelineDepthStencilStateCreateInfo sky_depth = vk::defaultDepthStencilState()
					.setDepthTestEnable(0)
					.setDepthWriteEnable(0);

				sky_pipeline = vk::graphicsPipelineCreateInfo()
					.setStages(sky_stages[..])
					.setLayout(sky_layout)
					.setVertexInputState(&vertex_input_state)
					.setDynamicState(&dynamic_state)
					.setRasterizationState(&sky_raster)
					.setInputAssemblyState(&input_assembly_state)
					.setMultisampleState(&multisample_state)
					.setDepthStencilState(&sky_depth)
					.setColorBlendState(&color_blend_state)
					.setViewportState(&viewport_state)
					.setRenderPass(render_pass)
					.build(device)!!;

				// The checkerboard stands in for an equirectangular image: the
				// point here is that the ported shader samples what the def said
				// it would.
				sky_descriptor = vk::writeDescriptorSet()
					.setDstBinding(0)
					.setDescriptorType(DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
					.setImageInfo(
						{
							{
								.imageView = texture_view,
								.sampler = texture_sampler,
								.imageLayout = IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
							}
						}
					);

				io::printn("sky: generated module compiled, validated by use, and drawn");
			}
		}
	}

	// --- Frame sync ---
	vk::FrameSync frame_sync;
	frame_sync.init(device, sc.image_count)!!;

	update_camera(&camera, &uniform_buffer, &sky_push);

	bool running = true;
	uint frames_drawn = 0;
	Vec2 prev_mouse;
	while (running)
	{
		EventMap event_map = win.getEvent();

		if (event_map.is_pressed(LEFT_MOUSE))
		{
			Vec2 mouse_pos = win.getMousePos();
			camera.rotate(mouse_pos.x - prev_mouse.x, (mouse_pos.y - prev_mouse.y));
			update_camera(&camera, &uniform_buffer, &sky_push);
			prev_mouse = mouse_pos;
		}

		if (event_map.is_pressed(MOUSE_SCROLL_UP))
		{
			camera.zoom(5);
			update_camera(&camera, &uniform_buffer, &sky_push);
		}

		if (event_map.is_pressed(MOUSE_SCROLL_DOWN))
		{
			camera.zoom(-5);
			update_camera(&camera, &uniform_buffer, &sky_push);
		}

		if (event_map.is_pressed(ESCAPE))
		{
			running = false;
		}

		uint next_image = frame_sync.beginFrame(device, sc.handle)!!;

		RenderPassBeginInfo render_pass_info = vk::renderPassBeginInfo().setFramebuffer(sc.framebuffers[next_image])
			.setRenderPass(render_pass)
			.setRenderArea({ .offset = { 0, 0 }, .extent = sc.extent })
			.setClearValues({ { .color = { .float32 = { 0.0, 0.0, 0.0, 1.0 } } } });

		CommandBuffer command_buffer = command_buffers[next_image];
		vk::resetCommandBuffer(command_buffer, (vk::CommandBufferResetFlags)0)!!;
		vk::beginCommandBuffer(command_buffer, &&vk::commandBufferBeginInfo())!!;

		vk::cmdBeginRenderPass(command_buffer, &render_pass_info, SUBPASS_CONTENTS_INLINE);
		vk::cmdSetViewport(
			command_buffer,
			0,
			1,
			(Viewport[]) { { .width = WIDTH, .height = HEIGHT, .minDepth = 0.0, .maxDepth = 1.0 } }
		);
		vk::cmdSetScissor(command_buffer, 0, 1, (Rect2D[]) { { .extent = sc.extent, .offset = { .x = 0, .y = 0 } } });

		if (sky_pipeline != null)
		{
			vk::cmdBindPipeline(command_buffer, PIPELINE_BIND_POINT_GRAPHICS, sky_pipeline);
			vk::cmdPushDescriptorSetKHR(
				command_buffer,
				PIPELINE_BIND_POINT_GRAPHICS,
				sky_layout,
				0,
				1,
				&sky_descriptor
			);
			vk::cmdPushConstants(
				command_buffer,
				sky_layout,
				SHADER_STAGE_FRAGMENT_BIT,
				0,
				SkyPush::size,
				&sky_push
			);
			// No vertex buffer and no index buffer: three vertices made in the
			// shader, which is the whole geometry of a sky.
			vk::cmdDraw(command_buffer, 3, 1, 0, 0);
		}

		vk::cmdBindPipeline(command_buffer, PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		vk::cmdPushDescriptorSetKHR(
			command_buffer,
			PIPELINE_BIND_POINT_GRAPHICS,
			pipeline_layout,
			0,
			descriptorSets.len,
			&descriptorSets[0]
		);

		vk::cmdPushConstants(
			command_buffer,
			pipeline_layout,
			SHADER_STAGE_VERTEX_BIT,
			0,
			DeviceAddress::size,
			&vertices_address
		);

		vk::cmdBindIndexBuffer(command_buffer, index_buffer.buffer, 0, INDEX_TYPE_UINT16);
		vk::cmdDrawIndexed(command_buffer, indices.len, 1, 0, 0, 0);
		vk::cmdEndRenderPass(command_buffer);

		frame_sync.endFrame(graphics_queue.handle, command_buffer, sc.handle, next_image)!!;

		frames_drawn++;
		if (frame_limit > 0 && frames_drawn >= frame_limit)
		{
			if (capture_path.len) capture_frame(device, graphics_queue, &allocator, &sc, next_image, capture_path);
			running = false;
		}
	}

	// Wait until Vulkan is finished, before freeing stuff
	device.waitIdle();

	if (sky_pipeline != null)
	{
		sky_pipeline.free(device);
		sky_layout.free(device);
		sky_set_layout.free(device);
		sky_module.free(device);
	}
	sc.free(device);
	render_pass.free(device);
	command_pool.free(device);
	vk::destroySampler(device, texture_sampler, null);
	texture_view.free(device);
	// Memory.free() destroys the image it backs, so texture_image must not be
	// freed separately. Both have to go before allocator.free() tears down the
	// pages the texture lives in.
	texture_memory.free();
	vertex_buffer.free();
	index_buffer.free();
	uniform_buffer.free();
	allocator.free();
	descriptor_set_layout.free(device);
	pipeline.free(device);
	pipeline_layout.free(device);
	frame_sync.free(device);
	surface.free(instance);
}
