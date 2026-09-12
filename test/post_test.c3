<*
 three.c3's generated post module, drawn as a full-screen pass over four inputs.

 The port under test is `shaders/post.shady`, reached through
 `build/shaderdef/post.shady`: one oversized triangle built from `vertex_index`
 alone, and a fragment that samples the previous pass, the scene, the frame's
 depth buffer and the named tap before handing a `Post` to the body. What this
 test asks is what that body is handed, so each value the struct carries gets a
 body of kong's own that returns it, and the readback is compared with the bytes
 that were bound rather than with a picture of them.

 The dump cannot be drawn with those bodies as it stands: three.c3's suite
 splices an identity `post` in *ahead* of the module, and appending a second
 definition would leave the call in the fragment matching two bodies. The dump
 keeps the walked push header ahead of everything and marks each source with
 `#line`; kong keeps that header and the module from `#line 1 "post"` on, and
 writes its body after the `@required` declaration - which is the shape the
 language expects a spliced definition to have.

 Four 1x1 inputs, each chosen so a wrong one cannot pass: the three colour
 textures are distinct bytes in every channel, and the depth texture is a
 `D32_SFLOAT` texel written with a known device depth under known planes. What
 the body reads is the module's world-space reconstruction, not the device
 depth, so the depth checks are the formula's own numbers.
*>
module post_test;

import std::core::test;
import std::core::mem::allocators;
import std::io;
import std::io::file;
import vk, shady, compute_support;

faultdef MODULE_DID_NOT_COMPILE;

const String POST_MODULE = "../three.c3/build/shaderdef/post.shady";
// Deliberately not square: `resolution` carries the width and the height, and a
// square attachment would let a transposed pair read as the right answer.
const uint WIDTH = 128;
const uint HEIGHT = 96;
// The camera planes the pass reconstructs device depth with.
const float NEAR = 0.1;
const float FAR = 100.0;
// Pushed seconds. A quarter is between the byte the colours hold and the ones
// resolution holds, so a misread field cannot land on it by saturation.
const float TIME = 0.25;

// `Uniforms`, the pass's own uniform block: this build declares none, so the
// generator emits its reserved stub - one float that keeps the struct non-empty.
struct Uniforms
{
	float material_unused;
}

$assert Uniforms::size == 4 : "Uniforms no longer matches three.c3's";

// `PostPush` (`src/shader/post.c3`): the header the generator walks into the
// module's push block, plus the merged `Uniforms` member. The offsets are the
// contract - info at 0, planes at 16, uniforms at 24 - so a field added or moved
// would put the fragment's reads off target.
struct PostPush
{
	float[4] info;
	float[2] planes;
	Uniforms uniforms;
}

$assert PostPush::size == 28 : "PostPush no longer matches three.c3's";

// Bodies kong supplies, one per value the module must hand over. Each returns
// its own field, so the check below reads that field's bytes and nothing else.
macro String uv_body()
{
	return `
fn float3 post(Post p)
{
	return float3(p.uv, 0.0);
}
`;
}

macro String colour_body()
{
	return `
fn float3 post(Post p)
{
	return p.color;
}
`;
}

macro String scene_body()
{
	return `
fn float3 post(Post p)
{
	return p.scene;
}
`;
}

macro String tap_body()
{
	return `
fn float3 post(Post p)
{
	return p.tap;
}
`;
}

macro String resolution_body()
{
	return `
fn float3 post(Post p)
{
	return float3(p.resolution / 255.0, 0.0);
}
`;
}

macro String time_body()
{
	return `
fn float3 post(Post p)
{
	return float3(0.0, p.time, 0.0);
}
`;
}

// The three device depths the reconstruction is asked about, and the factor the
// body multiplies the world depth by. Each factor puts the expected byte in the
// middle of the range, where the readback can tell the formula's value from a
// raw device depth or a near/far mix-up.
struct DepthCase
{
	float device;
	float scale;
}

const DepthCase[3] DEPTH_CASES = {
	{ .device = 0.5, .scale = 1.0 },
	{ .device = 1.0, .scale = 0.0025 },
	{ .device = 0.0, .scale = 8.0 },
};

// The reconstruction the port promises, in world units along the view
// direction, under the planes the push block carried.
fn float world_depth(float d)
{
	return NEAR * FAR / (FAR - d * (FAR - NEAR));
}

fn String depth_body(float scale)
{
	return string::tformat(
		"fn float3 post(Post p)\n{\n\treturn float3(p.depth * %f, 0.0, 0.0);\n}\n",
		scale
	);
}

// The module text the generator wrote, with the identity body three.c3 spliced
// ahead of it dropped: kong's own bodies have to be the only definition of
// `post` in the compilation.
//
// The dump is the walked push header and variable, then each source behind its
// `#line` marker - the identity body first, the module `shaders/post.shady`
// second. Keeping the header and the module and cutting between the two markers
// is what leaves a compilation whose only `post` is the body kong writes.
fn String? module_text(String dump)
{
	sz? body_at = dump.index_of("#line 1 \"post_body\"");
	if (catch body_at) return MODULE_DID_NOT_COMPILE~;
	sz? module_at = dump.index_of("#line 1 \"post\"");
	if (catch module_at) return MODULE_DID_NOT_COMPILE~;
	if (module_at <= body_at) return MODULE_DID_NOT_COMPILE~;
	// Ranges include both ends, so the prefix stops one byte short of the
	// marker's `#` and the module starts at its own.
	return string::tformat("%s%s", dump[0..(usz)body_at - 1], dump[(usz)module_at..]);
}

// One body compiled against the module. The generated module is compiled with
// C3's packing, the layout mode the push struct above is written in.
fn ShaderModule? load_variant(Fixture* f, String dump, String body)
{
	String? module_source = module_text(dump);
	if (catch module_source)
	{
		io::eprintfn("post: the dump no longer marks where its own body ends");
		return MODULE_DID_NOT_COMPILE~;
	}

	String source = string::tformat("%s\n%s", module_source, body);
	shady::Diagnostic diagnostic;
	char[]? spirv = shady::compile(source, &diagnostic, scalar_layout: true);
	if (catch spirv)
	{
		io::eprintfn("post: this body does not compile against the module:\n%s\n%s",
			body, diagnostic.to_string(tmem));
		diagnostic.free();
		return MODULE_DID_NOT_COMPILE~;
	}
	diagnostic.free();
	defer free(spirv);
	return vk::loadShaderModule(f.device, spirv);
}

struct Texture
{
	Image image;
	Memory memory;
	ImageView view;
}

fn void Texture.free(&self, Device device)
{
	vk::destroyImageView(device, self.view, null);
	// `create_image_buffer` hands back a sub-allocation and its `free` destroys
	// the image too, so the image is not destroyed here as well.
	self.memory.free();
}

// A 1x1 sampled image holding one texel: the whole of what a check needs when
// the value under it is a constant, and one texel means no filter or address
// mode can change the answer.
fn Texture? solid_texture(Fixture* f, char r, char g, char b)
{
	char[4] texel = { r, g, b, 255 };
	Image image = vk::imageCreateInfo()
		.setImageType(IMAGE_TYPE_2D)
		.setFormat(FORMAT_R8G8B8A8_UNORM)
		.setExtent({ .width = 1, .height = 1, .depth = 1 })
		.setMipLevels(1)
		.setArrayLayers(1)
		.setSamples(SAMPLE_COUNT_1_BIT)
		.setTiling(IMAGE_TILING_OPTIMAL)
		.setUsage(IMAGE_USAGE_TRANSFER_DST_BIT | IMAGE_USAGE_SAMPLED_BIT)
		.setInitialLayout(IMAGE_LAYOUT_UNDEFINED)
		.build(f.device)!;
	Memory memory = vk::create_image_buffer(image, &f.allocator, MEMORY_PROPERTY_DEVICE_LOCAL_BIT)!;

	write_texel(
		f,
		image,
		IMAGE_ASPECT_COLOR_BIT,
		&texel[0],
		4,
		IMAGE_LAYOUT_UNDEFINED,
		IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
	)!;

	ImageView view = vk::imageViewCreateInfo()
		.setImage(image)
		.setViewType(IMAGE_VIEW_TYPE_2D)
		.setFormat(FORMAT_R8G8B8A8_UNORM)
		.setSubresourceRange(
			{
				.aspectMask     = IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel   = 0,
				.levelCount     = 1,
				.baseArrayLayer = 0,
				.layerCount     = 1,
			}
		)
		.build(f.device)!;

	return (Texture) { .image = image, .memory = memory, .view = view };
}

// The depth input, as the rasteriser would leave it: a D32 texel holding the
// device depth a fragment under a standard projection would have written.
fn Texture? depth_texture(Fixture* f, float value)
{
	float[1] texel = { value };
	Image image = vk::imageCreateInfo()
		.setImageType(IMAGE_TYPE_2D)
		.setFormat(FORMAT_D32_SFLOAT)
		.setExtent({ .width = 1, .height = 1, .depth = 1 })
		.setMipLevels(1)
		.setArrayLayers(1)
		.setSamples(SAMPLE_COUNT_1_BIT)
		.setTiling(IMAGE_TILING_OPTIMAL)
		.setUsage(IMAGE_USAGE_TRANSFER_DST_BIT | IMAGE_USAGE_SAMPLED_BIT)
		.setInitialLayout(IMAGE_LAYOUT_UNDEFINED)
		.build(f.device)!;
	Memory memory = vk::create_image_buffer(image, &f.allocator, MEMORY_PROPERTY_DEVICE_LOCAL_BIT)!;

	write_texel(
		f,
		image,
		IMAGE_ASPECT_DEPTH_BIT,
		&texel[0],
		float::size,
		IMAGE_LAYOUT_UNDEFINED,
		IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
	)!;

	ImageView view = vk::imageViewCreateInfo()
		.setImage(image)
		.setViewType(IMAGE_VIEW_TYPE_2D)
		.setFormat(FORMAT_D32_SFLOAT)
		.setSubresourceRange(
			{
				.aspectMask     = IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel   = 0,
				.levelCount     = 1,
				.baseArrayLayer = 0,
				.layerCount     = 1,
			}
		)
		.build(f.device)!;

	return (Texture) { .image = image, .memory = memory, .view = view };
}

// Writes a new device depth into the depth texture, from the read-only layout
// the pass leaves it in. Each depth check runs against its own texel so the
// formula is asked about a known value rather than whatever a render left.
fn void? Texture.set_depth(&self, Fixture* f, float value)
{
	float[1] texel = { value };
	return write_texel(
		f,
		self.image,
		IMAGE_ASPECT_DEPTH_BIT,
		&texel[0],
		float::size,
		IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
		IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
	);
}

// Copies one texel into an image and lands it in the layout a shader reads it
// through. One submit, waited on: every render below runs after the write.
fn void? write_texel(
	Fixture* f,
	Image image,
	ImageAspectFlags aspect,
	void* data,
	sz size,
	ImageLayout old_layout,
	ImageLayout final_layout
) {
	Memory stage = vk::new_buffer(
		allocator:  &f.allocator,
		usage:      BUFFER_USAGE_TRANSFER_SRC_BIT,
		properties: MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT,
		data:       data,
		data_size:  size,
	)!;
	defer stage.free();

	// A fresh image has nothing to synchronise with; a rewrite comes after the
	// fragment stage sampled the old value.
	PipelineStageFlags2 src_stage = old_layout == IMAGE_LAYOUT_UNDEFINED
		? PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
		: PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
	AccessFlags2 src_access = old_layout == IMAGE_LAYOUT_UNDEFINED
		? (AccessFlags2)0
		: ACCESS_2_SHADER_SAMPLED_READ_BIT;

	ImageMemoryBarrier2[1] to_transfer = {
		{
			.sType            = STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.image            = image,
			.srcStageMask     = src_stage,
			.dstStageMask     = PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
			.srcAccessMask    = src_access,
			.dstAccessMask    = ACCESS_2_TRANSFER_WRITE_BIT,
			.oldLayout        = old_layout,
			.newLayout        = IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.subresourceRange = {
				.aspectMask     = aspect,
				.baseMipLevel   = 0,
				.levelCount     = 1,
				.baseArrayLayer = 0,
				.layerCount     = 1,
			},
		}
	};
	ImageMemoryBarrier2[1] to_shader = {
		{
			.sType            = STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.image            = image,
			.srcStageMask     = PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
			.dstStageMask     = PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
			.srcAccessMask    = ACCESS_2_TRANSFER_WRITE_BIT,
			.dstAccessMask    = ACCESS_2_SHADER_SAMPLED_READ_BIT,
			.oldLayout        = IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.newLayout        = final_layout,
			.subresourceRange = {
				.aspectMask     = aspect,
				.baseMipLevel   = 0,
				.levelCount     = 1,
				.baseArrayLayer = 0,
				.layerCount     = 1,
			},
		}
	};

	BufferImageCopy region = {
		.imageSubresource = {
			.aspectMask     = aspect,
			.mipLevel       = 0,
			.baseArrayLayer = 0,
			.layerCount     = 1,
		},
		.imageExtent = { 1, 1, 1 },
	};

	vk::resetCommandBuffer(f.cmd, (vk::CommandBufferResetFlags)0)!;
	vk::beginCommandBuffer(f.cmd, &&vk::commandBufferBeginInfo())!;
	vk::cmdPipelineBarrier2(f.cmd, &&vk::dependencyInfo().setImageMemoryBarriers(to_transfer[..]));
	vk::cmdCopyBufferToImage(
		f.cmd,
		stage.buffer,
		image,
		IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1,
		&region
	);
	vk::cmdPipelineBarrier2(f.cmd, &&vk::dependencyInfo().setImageMemoryBarriers(to_shader[..]));
	vk::endCommandBuffer(f.cmd)!;

	vk::resetFences(f.device, 1, &f.fence)!;
	CommandBuffer[1] command_buffers = { f.cmd };
	SubmitInfo submit = vk::submitInfo().setCommandBuffers(command_buffers[..]);
	vk::queueSubmit(f.queue.handle, 1, &submit, f.fence)!;
	vk::waitForFences(f.device, 1, &f.fence, vk::TRUE, ~(ulong)0)!;
}

struct Colour
{
	Image image;
	Memory memory;
	ImageView view;
	RenderPass pass;
	Framebuffer framebuffer;
}

fn void Colour.free(&self, Device device)
{
	vk::destroyFramebuffer(device, self.framebuffer, null);
	vk::destroyRenderPass(device, self.pass, null);
	vk::destroyImageView(device, self.view, null);
	// `create_image_buffer` hands back a sub-allocation and its `free` destroys
	// the image too, so the image is not destroyed here as well.
	self.memory.free();
}

// The colour attachment the pass draws into, cleared to blue: the triangle
// covers every pixel, so blue in the readback is a fragment that never ran.
fn Colour? colour_up(Fixture* f)
{
	Image image = vk::imageCreateInfo()
		.setImageType(IMAGE_TYPE_2D)
		.setFormat(FORMAT_R8G8B8A8_UNORM)
		.setExtent({ .width = WIDTH, .height = HEIGHT, .depth = 1 })
		.setMipLevels(1)
		.setArrayLayers(1)
		.setSamples(SAMPLE_COUNT_1_BIT)
		.setTiling(IMAGE_TILING_OPTIMAL)
		.setUsage(IMAGE_USAGE_COLOR_ATTACHMENT_BIT | IMAGE_USAGE_TRANSFER_SRC_BIT)
		.setInitialLayout(IMAGE_LAYOUT_UNDEFINED)
		.build(f.device)!;
	Memory memory = vk::create_image_buffer(image, &f.allocator, MEMORY_PROPERTY_DEVICE_LOCAL_BIT)!;

	ImageView view = vk::imageViewCreateInfo()
		.setImage(image)
		.setViewType(IMAGE_VIEW_TYPE_2D)
		.setFormat(FORMAT_R8G8B8A8_UNORM)
		.setSubresourceRange(
			{
				.aspectMask     = IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel   = 0,
				.levelCount     = 1,
				.baseArrayLayer = 0,
				.layerCount     = 1,
			}
		)
		.build(f.device)!;

	// One colour attachment and one subpass; the depth the pass reads is an
	// input image, so the pass has no depth attachment and no depth test.
	AttachmentReference colour_attachment = {
		.attachment = 0,
		.layout     = IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	};
	AttachmentDescription[1] attachments = {
		{
			.format         = FORMAT_R8G8B8A8_UNORM,
			.samples        = SAMPLE_COUNT_1_BIT,
			.loadOp         = ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp        = ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp  = ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout  = IMAGE_LAYOUT_UNDEFINED,
			.finalLayout    = IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		}
	};
	SubpassDescription[1] subpasses = {
		{
			.pipelineBindPoint    = PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 1,
			.pColorAttachments    = &colour_attachment,
		}
	};
	RenderPass pass = vk::renderPassCreateInfo()
		.setAttachments(attachments[..])
		.setSubpasses(subpasses[..])
		.build(f.device)!;

	ImageView[1] views = { view };
	Framebuffer framebuffer = vk::framebufferCreateInfo()
		.setRenderPass(pass)
		.setAttachments(views[..])
		.setWidth(WIDTH)
		.setHeight(HEIGHT)
		.setLayers(1)
		.build(f.device)!;

	return (Colour) {
		.image       = image,
		.memory      = memory,
		.view        = view,
		.pass        = pass,
		.framebuffer = framebuffer,
	};
}

// The device, the attachment, the four inputs, and the readback the checks are
// made on.
struct Rig
{
	Fixture f;
	Colour colour;
	Texture prev;
	Texture scene;
	Texture tap;
	Texture depth;
	Sampler sampler;
	Memory readback;
}

// Everything the rig owns, released together. The fixture goes last: the
// `Memory` frees above and the images both need the device.
fn void Rig.free(&self)
{
	self.readback.free();
	vk::destroySampler(self.f.device, self.sampler, null);
	self.depth.free(self.f.device);
	self.tap.free(self.f.device);
	self.scene.free(self.f.device);
	self.prev.free(self.f.device);
	self.colour.free(self.f.device);
	self.f.free();
}

// The rig is filled in place rather than returned: every allocation below keeps
// a pointer to the fixture's allocator, and a rig returned by value would leave
// those pointers describing a frame that has ended.
fn void? rig_up(Rig* rig)
{
	rig.f = compute_support::device_up()!;
	rig.colour = colour_up(&rig.f)!;

	// Each texel is a byte no other input shares in the same channel, so a body
	// handed the wrong image cannot land on the right answer.
	rig.prev  = solid_texture(&rig.f, 64, 128, 191)!;
	rig.scene = solid_texture(&rig.f, 230, 51, 102)!;
	rig.tap   = solid_texture(&rig.f, 51, 204, 153)!;
	// The first depth check rewrites this; a known 0.5 means the image is never
	// read in the UNDEFINED layout it was created in.
	rig.depth = depth_texture(&rig.f, 0.5)!;

	rig.sampler = vk::samplerCreateInfo()
		.setMagFilter(FILTER_NEAREST)
		.setMinFilter(FILTER_NEAREST)
		.setAddressModeU(SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
		.setAddressModeV(SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
		.setAddressModeW(SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
		.build(rig.f.device)!;

	rig.readback = vk::new_buffer(
		allocator:  &rig.f.allocator,
		usage:      BUFFER_USAGE_TRANSFER_DST_BIT,
		properties: MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT,
		data:       null,
		data_size:  (sz)(WIDTH * HEIGHT) * 4,
	)!;
}

// The push block: the target's extent, the clock, and the camera's planes.
fn PostPush Rig.push(&self)
{
	return (PostPush) {
		.info   = { (float)WIDTH, (float)HEIGHT, TIME, 0.0 },
		.planes = { NEAR, FAR },
	};
}

// Draw the full-screen triangle once and hand back the attachment's bytes. The
// clear is blue and no body writes a blue this value can be confused with.
fn char[] Rig.render(&self, Pipeline pipeline, PipelineLayout layout, DescriptorSet set)
{
	PostPush push = self.push();

	ClearValue[1] clear_values = {
		{ .color = { .float32 = { 0.0, 0.0, 1.0, 1.0 } } },
	};
	RenderPassBeginInfo begin = vk::renderPassBeginInfo()
		.setFramebuffer(self.colour.framebuffer)
		.setRenderPass(self.colour.pass)
		.setRenderArea({ .offset = { 0, 0 }, .extent = { WIDTH, HEIGHT } })
		.setClearValues(clear_values[..]);

	vk::resetCommandBuffer(self.f.cmd, (vk::CommandBufferResetFlags)0)!!;
	vk::beginCommandBuffer(self.f.cmd, &&vk::commandBufferBeginInfo())!!;
	vk::cmdBeginRenderPass(self.f.cmd, &begin, SUBPASS_CONTENTS_INLINE);
	vk::cmdBindPipeline(self.f.cmd, PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	vk::cmdBindDescriptorSets(self.f.cmd, PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, null);
	vk::cmdSetViewport(
		self.f.cmd,
		0,
		1,
		(Viewport[]) { { .width = (float)WIDTH, .height = (float)HEIGHT, .minDepth = 0.0, .maxDepth = 1.0 } }
	);
	vk::cmdSetScissor(self.f.cmd, 0, 1, (Rect2D[]) { { .extent = { WIDTH, HEIGHT }, .offset = { 0, 0 } } });
	vk::cmdPushConstants(
		self.f.cmd,
		layout,
		SHADER_STAGE_FRAGMENT_BIT,
		0,
		(uint)PostPush::size,
		&push
	);
	// Three vertices and no vertex buffer: the triangle is arithmetic on
	// `vertex_index`, which is the whole of the geometry the pass owns.
	vk::cmdDraw(self.f.cmd, 3, 1, 0, 0);
	vk::cmdEndRenderPass(self.f.cmd);

	ImageMemoryBarrier2 to_src = {
		.sType            = STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.image            = self.colour.image,
		.srcStageMask     = PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		.dstStageMask     = PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
		.srcAccessMask    = ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
		.dstAccessMask    = ACCESS_2_TRANSFER_READ_BIT,
		.oldLayout        = IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		.newLayout        = IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		.subresourceRange = {
			.aspectMask     = IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel   = 0,
			.levelCount     = 1,
			.baseArrayLayer = 0,
			.layerCount     = 1,
		},
	};
	ImageMemoryBarrier2[1] barriers = { to_src };
	vk::cmdPipelineBarrier2(self.f.cmd, &&vk::dependencyInfo().setImageMemoryBarriers(barriers[..]));

	BufferImageCopy region = {
		.imageSubresource = {
			.aspectMask     = IMAGE_ASPECT_COLOR_BIT,
			.mipLevel       = 0,
			.baseArrayLayer = 0,
			.layerCount     = 1,
		},
		.imageExtent = { WIDTH, HEIGHT, 1 },
	};
	vk::cmdCopyImageToBuffer(
		self.f.cmd,
		self.colour.image,
		IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		self.readback.buffer,
		1,
		&region
	);
	vk::endCommandBuffer(self.f.cmd)!!;

	vk::resetFences(self.f.device, 1, &self.f.fence)!!;
	CommandBuffer[1] command_buffers = { self.f.cmd };
	SubmitInfo submit = vk::submitInfo().setCommandBuffers(command_buffers[..]);
	vk::queueSubmit(self.f.queue.handle, 1, &submit, self.f.fence)!!;
	vk::waitForFences(self.f.device, 1, &self.f.fence, vk::TRUE, ~(ulong)0)!!;

	// A copy on the heap rather than a view of the readback: the next render
	// overwrites that, and the caller frees this the ordinary way.
	char[] pixels = alloc::new_array(mem, char, (sz)(WIDTH * HEIGHT) * 4);
	mem::copy(&pixels[0], self.readback.data(), (sz)(WIDTH * HEIGHT) * 4);
	return pixels;
}

// The module's shape: a vertex stage that builds its own triangle and a
// fragment stage that reads the four inputs, with no vertex input and no depth
// test against the attachment - the pass samples the depth buffer instead.
fn Pipeline? post_pipeline_up(Fixture* f, RenderPass pass, PipelineLayout layout, ShaderModule shader)
{
	PipelineShaderStageCreateInfo[2] stages = {
		vk::pipelineShaderStageCreateInfo()
			.setStage(SHADER_STAGE_VERTEX_BIT)
			.setMod(shader)
			.setName("vertexMain"),
		vk::pipelineShaderStageCreateInfo()
			.setStage(SHADER_STAGE_FRAGMENT_BIT)
			.setMod(shader)
			.setName("fragmentMain"),
	};

	PipelineVertexInputStateCreateInfo vertex_input = vk::pipelineVertexInputStateCreateInfo();
	PipelineInputAssemblyStateCreateInfo input_assembly = vk::pipelineInputAssemblyStateCreateInfo()
		.setTopology(PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
	PipelineRasterizationStateCreateInfo rasterization = vk::defaultRasterizationState()
		.setCullMode(CULL_MODE_NONE);
	PipelineMultisampleStateCreateInfo multisample = vk::defaultMultisampleState();
	PipelineDepthStencilStateCreateInfo depth_stencil = vk::defaultDepthStencilState();

	DynamicState[2] dynamic_states = { DYNAMIC_STATE_VIEWPORT, DYNAMIC_STATE_SCISSOR };
	PipelineDynamicStateCreateInfo dynamic_state = vk::pipelineDynamicStateCreateInfo()
		.setDynamicStates(dynamic_states[..]);

	Viewport[1] viewports = { { .width = (float)WIDTH, .height = (float)HEIGHT, .minDepth = 0.0, .maxDepth = 1.0 } };
	Rect2D[1] scissors = { { .extent = { WIDTH, HEIGHT }, .offset = { 0, 0 } } };
	PipelineViewportStateCreateInfo viewport_state = vk::pipelineViewportStateCreateInfo()
		.setViewports(viewports[..])
		.setScissors(scissors[..]);

	PipelineColorBlendAttachmentState[1] blend_attachments = {
		{
			.blendEnable    = vk::FALSE,
			.colorWriteMask = COLOR_COMPONENT_R_BIT | COLOR_COMPONENT_G_BIT
				| COLOR_COMPONENT_B_BIT | COLOR_COMPONENT_A_BIT,
		}
	};
	PipelineColorBlendStateCreateInfo color_blend = vk::defaultColorBlendState(blend_attachments[..]);

	return vk::graphicsPipelineCreateInfo()
		.setStages(stages[..])
		.setLayout(layout)
		.setVertexInputState(&vertex_input)
		.setInputAssemblyState(&input_assembly)
		.setRasterizationState(&rasterization)
		.setMultisampleState(&multisample)
		.setDepthStencilState(&depth_stencil)
		.setColorBlendState(&color_blend)
		.setDynamicState(&dynamic_state)
		.setViewportState(&viewport_state)
		.setRenderPass(pass)
		.build(f.device);
}

// Compile one body against the module, draw it, and hand back the attachment's
// bytes. The frame is written before the caller's checks, so a failure still
// leaves something to look at.
fn char[] render_variant(Rig* rig, String dump, PipelineLayout layout, DescriptorSet set, String body, String ppm)
{
	ShaderModule shader = load_variant(&rig.f, dump, body)!!;
	defer vk::destroyShaderModule(rig.f.device, shader, null);
	Pipeline pipeline = post_pipeline_up(&rig.f, rig.colour.pass, layout, shader)!!;
	defer vk::destroyPipeline(rig.f.device, pipeline, null);

	char[] pixels = rig.render(pipeline, layout, set);
	write_ppm(ppm, pixels);
	return pixels;
}

fn char* pixel_at(char[] pixels, uint x, uint y) @inline
{
	return &pixels[(y * WIDTH + x) * 4];
}

// Unorm quantisation may round either way; a byte either side of the value the
// float maps to is the same number with a different rounding mode, not a
// different value handed to the body.
fn bool byte_near(char got, float wanted) @inline
{
	float d = (float)(uint)got - wanted;
	if (d < 0.0f) d = -d;
	return d <= 1.5f;
}

// The frame as a binary PPM, so a failure can be looked at: header and bytes,
// the same format `capture_frame` in `src/main.c3` writes, for the same reason -
// nothing here has a PNG encoder.
fn void write_ppm(String path, char[] pixels)
{
	String header = string::tformat("P6\n%d %d\n255\n", (int)WIDTH, (int)HEIGHT);
	char[] out = alloc::new_array(mem, char, (sz)(header.len + WIDTH * HEIGHT * 3));
	mem::copy(out.ptr, header.ptr, header.len);
	for (uint i = 0; i < WIDTH * HEIGHT; i++)
	{
		out[header.len + (usz)i * 3]     = pixels[i * 4];
		out[header.len + (usz)i * 3 + 1] = pixels[i * 4 + 1];
		out[header.len + (usz)i * 3 + 2] = pixels[i * 4 + 2];
	}
	file::save(path, out)!!;
	free(out);
	io::printfn("post: wrote %s", path);
}

fn void the_ported_post_hands_the_body_its_values() @test
{
	if (!file::is_file(POST_MODULE))
	{
		io::printfn("post: no %s; skipped", POST_MODULE);
		return;
	}

	String dump = (String)file::load(tmem, POST_MODULE)!!;

	Rig rig;
	rig_up(&rig)!!;
	defer rig.free();

	// Set 0 carries the module's four reserved samplers, in binding order. The
	// depth descriptor reads the depth aspect, so it names the read-only layout
	// a shadow lookup would.
	DescriptorSetLayout samplers_layout = vk::descriptorSetLayoutCreateInfo()
		.setBindings(
			{
				{
					.binding         = 0,
					.descriptorType  = DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags      = SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding         = 1,
					.descriptorType  = DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags      = SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding         = 2,
					.descriptorType  = DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags      = SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding         = 3,
					.descriptorType  = DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags      = SHADER_STAGE_FRAGMENT_BIT,
				},
			}
		)
		.build(rig.f.device)!!;
	defer vk::destroyDescriptorSetLayout(rig.f.device, samplers_layout, null);

	DescriptorPool pool = vk::descriptorPoolCreateInfo()
		.setMaxSets(1)
		.setPoolSizes(
			{ { .type = DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 4 } }
		)
		.build(rig.f.device)!!;
	defer vk::destroyDescriptorPool(rig.f.device, pool, null);

	DescriptorSet set;
	DescriptorSetLayout[1] set_layouts = { samplers_layout };
	vk::allocateDescriptorSets(
		rig.f.device,
		&&vk::descriptorSetAllocateInfo().setDescriptorPool(pool).setSetLayouts(set_layouts[..]),
		&set
	)!!;

	// The write keeps a pointer to the image info, so the four arrays below are
	// named and live until after the update.
	DescriptorImageInfo[1] prev_info = {
		{
			.imageView   = rig.prev.view,
			.imageLayout = IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.sampler     = rig.sampler,
		}
	};
	DescriptorImageInfo[1] scene_info = {
		{
			.imageView   = rig.scene.view,
			.imageLayout = IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.sampler     = rig.sampler,
		}
	};
	DescriptorImageInfo[1] depth_info = {
		{
			.imageView   = rig.depth.view,
			.imageLayout = IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
			.sampler     = rig.sampler,
		}
	};
	DescriptorImageInfo[1] tap_info = {
		{
			.imageView   = rig.tap.view,
			.imageLayout = IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.sampler     = rig.sampler,
		}
	};
	WriteDescriptorSet[4] writes = {
		vk::writeDescriptorSet()
			.setDstSet(set)
			.setDstBinding(0)
			.setDescriptorType(DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
			.setImageInfo(prev_info[..]),
		vk::writeDescriptorSet()
			.setDstSet(set)
			.setDstBinding(1)
			.setDescriptorType(DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
			.setImageInfo(scene_info[..]),
		vk::writeDescriptorSet()
			.setDstSet(set)
			.setDstBinding(2)
			.setDescriptorType(DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
			.setImageInfo(depth_info[..]),
		vk::writeDescriptorSet()
			.setDstSet(set)
			.setDstBinding(3)
			.setDescriptorType(DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
			.setImageInfo(tap_info[..]),
	};
	vk::updateDescriptorSets(rig.f.device, 4, &writes[0], 0, null);

	// The push header the pass reads is 28 bytes; the fragment stage is the one
	// that reads it.
	PipelineLayout layout = vk::pipelineLayoutCreateInfo()
		.setSetLayouts({ samplers_layout })
		.setPushConstantRanges(
			{ { .stageFlags = SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = (uint)PostPush::size } }
		)
		.build(rig.f.device)!!;
	defer vk::destroyPipelineLayout(rig.f.device, layout, null);

	// The triangle's uv (0, 0) is the clip-space corner the viewport puts at the
	// top left, so the first framebuffer pixel carries the first texel's centre:
	// a flip anywhere in the pass would leave the last row's v here instead.
	char[] uv_pixels = render_variant(&rig, dump, layout, set, uv_body(), "build/post_uv.ppm");
	defer free(uv_pixels);

	float want_u0 = 0.5f / (float)WIDTH * 255.0f;
	float want_v0 = 0.5f / (float)HEIGHT * 255.0f;
	char* uv_top_left = pixel_at(uv_pixels, 0, 0);
	test::@check(
		byte_near(uv_top_left[0], want_u0) && byte_near(uv_top_left[1], want_v0) && uv_top_left[2] == 0,
		"the body's uv at the top-left pixel is the first texel's centre: %d, %d, %d (wanted %d, %d, 0)",
		(int)(uint)uv_top_left[0], (int)(uint)uv_top_left[1], (int)(uint)uv_top_left[2],
		(int)(uint)(want_u0 + 0.5f), (int)(uint)(want_v0 + 0.5f)
	);

	// v rises with the framebuffer row and u holds down the column, which is
	// the pair of claims a vertical flip or a swapped pair of coordinates
	// breaks.
	float want_v1 = ((float)(HEIGHT - 1) + 0.5f) / (float)HEIGHT * 255.0f;
	char* uv_bottom_left = pixel_at(uv_pixels, 0, HEIGHT - 1);
	test::@check(
		byte_near(uv_bottom_left[0], want_u0) && byte_near(uv_bottom_left[1], want_v1)
			&& byte_near(uv_bottom_left[2], 0.0f),
		"and the last row carries v = 1 at the left edge: %d, %d, %d (wanted %d, %d, 0)",
		(int)(uint)uv_bottom_left[0], (int)(uint)uv_bottom_left[1], (int)(uint)uv_bottom_left[2],
		(int)(uint)(want_u0 + 0.5f), (int)(uint)(want_v1 + 0.5f)
	);

	// u rises with the column: the top-right pixel is the mirror of the
	// bottom-left one.
	float want_u1 = ((float)(WIDTH - 1) + 0.5f) / (float)WIDTH * 255.0f;
	char* uv_top_right = pixel_at(uv_pixels, WIDTH - 1, 0);
	test::@check(
		byte_near(uv_top_right[0], want_u1) && byte_near(uv_top_right[1], want_v0)
			&& byte_near(uv_top_right[2], 0.0f),
		"and the last column carries u = 1 on the first row: %d, %d, %d (wanted %d, %d, 0)",
		(int)(uint)uv_top_right[0], (int)(uint)uv_top_right[1], (int)(uint)uv_top_right[2],
		(int)(uint)(want_u1 + 0.5f), (int)(uint)(want_v0 + 0.5f)
	);

	// The previous pass's texel, which is the identity body's own value.
	char[] colour_pixels = render_variant(&rig, dump, layout, set, colour_body(), "build/post_colour.ppm");
	defer free(colour_pixels);
	char* colour = pixel_at(colour_pixels, WIDTH / 2, HEIGHT / 2);
	test::@check(
		byte_near(colour[0], 64.0f) && byte_near(colour[1], 128.0f) && byte_near(colour[2], 191.0f),
		"the body's p.color is the previous pass's texel: %d, %d, %d (wanted 64, 128, 191)",
		(int)(uint)colour[0], (int)(uint)colour[1], (int)(uint)colour[2]
	);

	// The rendered scene's texel, whose bytes share no channel with p.color's.
	char[] scene_pixels = render_variant(&rig, dump, layout, set, scene_body(), "build/post_scene.ppm");
	defer free(scene_pixels);
	char* scene = pixel_at(scene_pixels, WIDTH / 2, HEIGHT / 2);
	test::@check(
		byte_near(scene[0], 230.0f) && byte_near(scene[1], 51.0f) && byte_near(scene[2], 102.0f),
		"the body's p.scene is the scene image's texel: %d, %d, %d (wanted 230, 51, 102)",
		(int)(uint)scene[0], (int)(uint)scene[1], (int)(uint)scene[2]
	);

	// The named pass's texel, the fourth binding and not a fallback to the
	// scene the module defaults to when nothing is named.
	char[] tap_pixels = render_variant(&rig, dump, layout, set, tap_body(), "build/post_tap.ppm");
	defer free(tap_pixels);
	char* tap = pixel_at(tap_pixels, WIDTH / 2, HEIGHT / 2);
	test::@check(
		byte_near(tap[0], 51.0f) && byte_near(tap[1], 204.0f) && byte_near(tap[2], 153.0f),
		"the body's p.tap is the tap image's texel: %d, %d, %d (wanted 51, 204, 153)",
		(int)(uint)tap[0], (int)(uint)tap[1], (int)(uint)tap[2]
	);

	// The target's extent, both components: the body scales by 255 so the
	// attachment's own pixel counts are the bytes.
	char[] resolution_pixels = render_variant(&rig, dump, layout, set, resolution_body(), "build/post_resolution.ppm");
	defer free(resolution_pixels);
	char* resolution = pixel_at(resolution_pixels, WIDTH / 3, HEIGHT / 3);
	test::@check(
		byte_near(resolution[0], (float)WIDTH) && byte_near(resolution[1], (float)HEIGHT)
			&& byte_near(resolution[2], 0.0f),
		"the body's p.resolution is the attachment's extent: %d, %d, %d (wanted %d, %d, 0)",
		(int)(uint)resolution[0], (int)(uint)resolution[1], (int)(uint)resolution[2],
		(int)WIDTH, (int)HEIGHT
	);

	// The clock the engine pushed in info.z, on a channel none of the other
	// checks uses.
	char[] time_pixels = render_variant(&rig, dump, layout, set, time_body(), "build/post_time.ppm");
	defer free(time_pixels);
	char* time = pixel_at(time_pixels, WIDTH / 3, HEIGHT / 3);
	test::@check(
		byte_near(time[0], 0.0f) && byte_near(time[1], TIME * 255.0f) && byte_near(time[2], 0.0f),
		"the body's p.time is what info.z pushed: %d, %d, %d (wanted 0, %d, 0)",
		(int)(uint)time[0], (int)(uint)time[1], (int)(uint)time[2], (int)(uint)(TIME * 255.0f + 0.5f)
	);

	// Device depth is not world depth: the module reconstructs it, and each
	// case's body scales the result so the expected byte sits mid-range. A
	// missing reconstruction, a raw device depth, or the wrong near/far pair
	// all land on a different byte than the formula's own.
	foreach (c : DEPTH_CASES)
	{
		rig.depth.set_depth(&rig.f, c.device)!!;
		String ppm = string::tformat("build/post_depth_%d.ppm", (int)(c.device * 100.0f));
		char[] depth_pixels = render_variant(&rig, dump, layout, set, depth_body(c.scale), ppm);
		defer free(depth_pixels);

		char* depth = pixel_at(depth_pixels, WIDTH / 2, HEIGHT / 2);
		float world = world_depth(c.device);
		float wanted = world * c.scale * 255.0f;
		test::@check(
			byte_near(depth[0], wanted) && byte_near(depth[1], 0.0f) && byte_near(depth[2], 0.0f),
			"device depth %.1f under near %.1f and far %.1f is %.6f world units, so the body reads %d, wanted %d",
			c.device, NEAR, FAR, world, (int)(uint)depth[0], (int)(uint)(wanted + 0.5f)
		);
	}
}
