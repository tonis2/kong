<*
 three.c3's generated mesh module, drawn as a colour pass.

 The ported vertex path is the thing under test. three.c3 swaps the material
 fragment stage for a stub when it writes this dump - `float4(input.uv,
 input.color.g, 1.0)` - which turns the attachment into the varyings themselves:
 red and green are the interpolated uv and blue is the instance's own tint, so
 what the checks below ask is whether the vertex stage reached the rasteriser
 with the values the streams hold.

 A cube is the whole geometry: one face turned to an orthographic camera, the
 other five edge-on and reaching no pixel. Its uvs run 0..1 across that face, so
 the centre is (0.5, 0.5), a quarter of the frame across moves u by a quarter of
 the face's span, and v holds along any row. The tint is deliberately 0.25: far
 from both ends of the unorm range, so quantisation cannot hide a swapped
 channel.

 The geometry arrives through device addresses exactly as the engine's does:
 `DrawRecord.positions` is a pointer in a push block, the instance is a second
 pointer, and nothing is bound as a vertex buffer. An empty vertex input state is
 what makes that a real check rather than a coincidence.

 The structs below are three.c3's wire format (`scene/scene.c3`,
 `gpu/pipeline.c3`), declared by their offsets because that is what the generated
 module reads. A field this test never sets stays in the declaration: the stride
 is the contract, and one that ran short would put every read off target.
*>
module mesh_test;

import std::core::test;
import std::core::mem::allocators;
import std::io;
import std::io::file;
import std::math::matrix;
import std::math::vector;
import vk, shady, compute_support;

faultdef MODULE_DID_NOT_COMPILE;

const String MESH_MODULE = "../three.c3/build/shaderdef/mesh.shady";
const uint MAP = 128;
const float FAR = 1.0;
// The cube spans 80% of the frame each way, so the clear colour frames it.
const float SCALE = 0.8;
// The instance colour's green, which the stub fragment stage writes to blue.
const float TINT_G = 0.25;
const uint HAS_UVS = 2;
const uint HAS_COLORS = 4;

// `Instance` (`scene/scene.c3`): the model matrix and the copy's own tint, and
// the pose fields the vertex stage reads as zeroes on this path.
struct Instance
{
	Matrix4f model;
	Matrix4f normal;
	float[4] color;
	uint     variant;
	uint     pose;
	uint     pose_from;
	float    blend;
	uint     morph;
	float[4] lightmap;
}

$assert Instance::size == 180 : "Instance no longer matches three.c3's";

// `SkinnedVertex` and `MorphDelta`, through which this pass reads nothing.
struct SkinnedVertex
{
	float[3] position;
	float[3] normal;
}

struct MorphDelta
{
	float[3] position;
	float[3] normal;
}

// `DrawRecord` (`gpu/pipeline.c3`), mirrored field for field: the vertex streams
// the vertex stage reaches through pointers, the flags that decide which of them
// it reads, and the material this dump's stub fragment does not read but the lit
// path's does. The generated module's copy is walked from the C3 struct in
// declaration order, so a field skipped or moved reads some other field's bytes.
struct DrawRecord
{
	float[3]* positions;
	float[3]* normals;
	float[2]* uvs;
	float[2]* lightmap_uvs;
	float[4]* colors;
	uint[4]* joints;
	float[4]* weights;
	Matrix4f* palette;
	SkinnedVertex* posed;
	MorphDelta* morph;
	float[4] base_color;
	uint     flags;
	uint     skin_stride;
	uint     morph_targets;
	float[4] uv_transform;
	float[4] surface;
	uint     uv_variant_count;
	float[4][8] uv_variants;
	float[4] emissive;
	uint[24] texture_index;
}

$assert DrawRecord::size == 384 : "DrawRecord no longer matches three.c3's";

// `MeshPush` (`gpu/pipeline.c3`): the frame, the instance array, and this
// bucket's record. The whole of what a mesh draw pushes.
struct MeshPush
{
	FrameBlock* frame;
	Instance* instances;
	DrawRecord* draw;
}

$assert MeshPush::size == 24 : "MeshPush no longer matches three.c3's";

// `FrameBlock` (`gpu/pipeline.c3`), mirrored field for field like every other wire
// struct here: the generated module's copy is walked from the C3 struct in
// declaration order, and the stride is the contract. This rig fills only
// `view_projection`; every field stays in its place so that offset cannot move.
struct FrameBlock
{
	Matrix4f view_projection;
	float[4] lighting;
	Matrix4f* live_poses;
	Matrix4f light_view_projection;
	float[4] shadow;
	float[4] debug;
	float[4] camera;
	float[4] environment;
	void* lights;
	float* morph_weights;
	float[4] world;
	float[4] point_shadow;
	Matrix4f inverse_view_projection;
	float[4] ambient_occlusion;
	void* shadow_views;
	uint* cluster_counts;
	uint* cluster_indices;
	uint[4] cluster_grid;
	float[4] cluster_screen;
}

$assert FrameBlock::size == 400 : "FrameBlock no longer matches three.c3's";

const uint CUBE_VERTICES = 36;

// One cube as six independent quads: 36 positions, so the draw needs no index
// buffer and `vid` runs straight over the positions.
fn float[3][CUBE_VERTICES] cube_positions()
{
	float[3][8] corners;
	for (usz i = 0; i < 8; i++)
	{
		corners[i][0] = (i & 1) != 0 ? 1.0 : -1.0;
		corners[i][1] = (i & 2) != 0 ? 1.0 : -1.0;
		corners[i][2] = (i & 4) != 0 ? 1.0 : -1.0;
	}

	// The four corners of each face, in two triangles, wound consistently.
	uint[CUBE_VERTICES] faces = {
		 4, 5, 7,  4, 7, 6,   // +z
		 0, 2, 3,  0, 3, 1,   // -z
		 2, 6, 7,  2, 7, 3,   // +y
		 0, 1, 5,  0, 5, 4,   // -y
		 1, 3, 7,  1, 7, 5,   // +x
		 0, 4, 6,  0, 6, 2,   // -x
	};

	float[3][CUBE_VERTICES] positions;
	for (usz i = 0; i < CUBE_VERTICES; i++) positions[i] = corners[faces[i]];
	return positions;
}

// u runs with x and v with y, each 0..1 across the +z face - the face the
// camera below sees, and so the only mapping the readback can see; the other
// five are edge-on and reach no pixel.
fn float[2][CUBE_VERTICES] cube_uvs()
{
	float[3][CUBE_VERTICES] positions = cube_positions();
	float[2][CUBE_VERTICES] uvs;
	for (usz i = 0; i < CUBE_VERTICES; i++)
	{
		uvs[i][0] = (positions[i][0] + 1.0f) * 0.5f;
		uvs[i][1] = (positions[i][1] + 1.0f) * 0.5f;
	}
	return uvs;
}

// An orthographic camera on the +z face: x and y scale the cube's [-1, 1] to
// [-0.8, 0.8] of the frame, and z puts the near face at 0 and the far one at 1.
// The matrix is written as columns, the way the shader reads it.
fn Matrix4f camera_projection()
{
	return (Matrix4f) {
		.m = {
			SCALE, 0.0, 0.0, 0.0,
			0.0, -SCALE, 0.0, 0.0,
			0.0, 0.0, -0.5, 0.0,
			0.0, 0.0, 0.5, 1.0,
		}
	};
}

// A framebuffer pixel's centre, through the projection above, to the uv the face
// gives it. The rasteriser interpolates the same two numbers in the shader.
fn float u_at(uint x)
{
	float ndc = ((float)x + 0.5f) * 2.0f / MAP - 1.0f;
	return (ndc / SCALE + 1.0f) * 0.5f;
}

fn float v_at(uint y)
{
	float ndc = ((float)y + 0.5f) * 2.0f / MAP - 1.0f;
	return (-ndc / SCALE + 1.0f) * 0.5f;
}

// The generated module, compiled with C3's packing - the layout mode the wire
// structs above are written in.
fn ShaderModule? load_module(Fixture* f, String path)
{
	String source = (String)file::load(tmem, path)!;
	shady::Diagnostic diagnostic;
	char[]? spirv = shady::compile(source, &diagnostic, scalar_layout: true);
	if (catch spirv)
	{
		io::eprintfn("the generated module does not compile: %s", diagnostic.to_string(tmem));
		diagnostic.free();
		return MODULE_DID_NOT_COMPILE~;
	}
	diagnostic.free();
	defer free(spirv);
	return vk::loadShaderModule(f.device, spirv);
}

struct Colour
{
	Image image;
	Memory memory;
	ImageView view;
	Image depth_image;
	Memory depth_memory;
	ImageView depth_view;
	RenderPass pass;
	Framebuffer framebuffer;
}

fn void Colour.free(&self, Device device)
{
	vk::destroyFramebuffer(device, self.framebuffer, null);
	vk::destroyRenderPass(device, self.pass, null);
	vk::destroyImageView(device, self.view, null);
	vk::destroyImageView(device, self.depth_view, null);
	// `create_image_buffer` hands back a sub-allocation and its `free` destroys
	// the image too, so neither image is destroyed here as well.
	self.depth_memory.free();
	self.memory.free();
}

// A colour attachment, cleared to blue, and the depth buffer the pipeline's
// depth test needs beside it: a test turned on without an attachment is a
// pipeline the validator rejects. Only the colour is copied back out.
fn Colour? colour_up(Fixture* f)
{
	Image image = vk::imageCreateInfo()
		.setImageType(IMAGE_TYPE_2D)
		.setFormat(FORMAT_R8G8B8A8_UNORM)
		.setExtent({ .width = MAP, .height = MAP, .depth = 1 })
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
				.aspectMask = IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			}
		)
		.build(f.device)!;

	Image depth_image = vk::imageCreateInfo()
		.setImageType(IMAGE_TYPE_2D)
		.setFormat(FORMAT_D32_SFLOAT)
		.setExtent({ .width = MAP, .height = MAP, .depth = 1 })
		.setMipLevels(1)
		.setArrayLayers(1)
		.setSamples(SAMPLE_COUNT_1_BIT)
		.setTiling(IMAGE_TILING_OPTIMAL)
		.setUsage(IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
		.setInitialLayout(IMAGE_LAYOUT_UNDEFINED)
		.build(f.device)!;
	Memory depth_memory = vk::create_image_buffer(depth_image, &f.allocator, MEMORY_PROPERTY_DEVICE_LOCAL_BIT)!;

	ImageView depth_view = vk::imageViewCreateInfo()
		.setImage(depth_image)
		.setViewType(IMAGE_VIEW_TYPE_2D)
		.setFormat(FORMAT_D32_SFLOAT)
		.setSubresourceRange(
			{
				.aspectMask = IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			}
		)
		.build(f.device)!;

	// One colour attachment, one depth attachment, one subpass.
	AttachmentReference colour_attachment = {
		.attachment = 0,
		.layout     = IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	};
	AttachmentReference depth_attachment = {
		.attachment = 1,
		.layout     = IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
	};
	AttachmentDescription[2] attachments = {
		{
			.format         = FORMAT_R8G8B8A8_UNORM,
			.samples        = SAMPLE_COUNT_1_BIT,
			.loadOp         = ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp        = ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp  = ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout  = IMAGE_LAYOUT_UNDEFINED,
			.finalLayout    = IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		},
		{
			.format         = FORMAT_D32_SFLOAT,
			.samples        = SAMPLE_COUNT_1_BIT,
			.loadOp         = ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp        = ATTACHMENT_STORE_OP_DONT_CARE,
			.stencilLoadOp  = ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout  = IMAGE_LAYOUT_UNDEFINED,
			.finalLayout    = IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		}
	};
	SubpassDescription[1] subpasses = {
		{
			.pipelineBindPoint       = PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount    = 1,
			.pColorAttachments       = &colour_attachment,
			.pDepthStencilAttachment = &depth_attachment,
		}
	};
	RenderPass pass = vk::renderPassCreateInfo()
		.setAttachments(attachments[..])
		.setSubpasses(subpasses[..])
		.build(f.device)!;

	ImageView[2] views = { view, depth_view };
	Framebuffer framebuffer = vk::framebufferCreateInfo()
		.setRenderPass(pass)
		.setAttachments(views[..])
		.setWidth(MAP)
		.setHeight(MAP)
		.setLayers(1)
		.build(f.device)!;

	return (Colour) {
		.image        = image,
		.memory       = memory,
		.view         = view,
		.depth_image  = depth_image,
		.depth_memory = depth_memory,
		.depth_view   = depth_view,
		.pass         = pass,
		.framebuffer  = framebuffer,
	};
}

// The device, the colour target, and one cube reached by address: uvs and
// colours flagged on, and every skinning stream null and flagged off.
struct Rig
{
	Fixture f;
	Colour colour;
	Memory positions;
	Memory uvs;
	Memory colors;
	Memory instances;
	Memory draw_record;
	Memory frame;
	Memory readback;
}

// Everything the rig owns, released together. The fixture goes last: the
// `Memory` frees above and the images both need the device.
fn void Rig.free(&self)
{
	self.readback.free();
	self.frame.free();
	self.draw_record.free();
	self.instances.free();
	self.colors.free();
	self.uvs.free();
	self.positions.free();
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

	float[3][CUBE_VERTICES] positions = cube_positions();
	rig.positions = compute_support::upload(&rig.f, &positions[0], CUBE_VERTICES * 12)!;

	float[2][CUBE_VERTICES] uvs = cube_uvs();
	rig.uvs = compute_support::upload(&rig.f, &uvs[0], CUBE_VERTICES * 8)!;

	// The vertex stage reads the colour stream because the flag says it is
	// there; the stub never looks at it, so white is the identity and no more.
	float[4][CUBE_VERTICES] colors;
	for (usz i = 0; i < CUBE_VERTICES; i++)
	{
		colors[i][0] = 1.0;
		colors[i][1] = 1.0;
		colors[i][2] = 1.0;
		colors[i][3] = 1.0;
	}
	rig.colors = compute_support::upload(&rig.f, &colors[0], CUBE_VERTICES * 16)!;

	Instance[1] instances;
	instances[0].model = (Matrix4f) {
		.m = {
			1.0, 0.0, 0.0, 0.0,
			0.0, 1.0, 0.0, 0.0,
			0.0, 0.0, 1.0, 0.0,
			0.0, 0.0, 0.0, 1.0,
		}
	};
	instances[0].normal = instances[0].model;
	// R and b are there so a channel mix-up cannot read as this tint: only
	// green is 0.25.
	instances[0].color = { 1.0, TINT_G, 0.75, 1.0 };
	rig.instances = compute_support::upload(&rig.f, &instances[0], Instance::size)!;

	// Zero-filled first: the material fields this stub does not read - emissive,
	// the uv variant table, the texture indices - are a material declaring none
	// rather than bytes the vertex stage would still read.
	DrawRecord[1] draws = {};
	draws[0].positions     = (float[3]*)rig.positions.get_address();
	draws[0].uvs           = (float[2]*)rig.uvs.get_address();
	draws[0].colors        = (float[4]*)rig.colors.get_address();
	draws[0].base_color    = { 1.0, 1.0, 1.0, 1.0 };
	draws[0].flags         = HAS_UVS | HAS_COLORS;
	draws[0].uv_transform  = { 1.0, 1.0, 0.0, 0.0 };
	// The material half is filled with the values the lit path shades, so the
	// record this test shares with it is whole where even the stub never asks.
	draws[0].surface       = { 0.5, 0.0, 0.5, 0.0 };
	// The morph loop's whole test: zero targets and zero weights mean every
	// null pointer below it stays unread.
	draws[0].morph_targets = 0;
	rig.draw_record = compute_support::upload(&rig.f, &draws[0], DrawRecord::size)!;

	FrameBlock[1] frame = {};
	frame[0].view_projection = camera_projection();
	rig.frame = compute_support::upload(&rig.f, &frame[0], FrameBlock::size)!;

	rig.readback = vk::new_buffer(
		allocator:  &rig.f.allocator,
		usage:      BUFFER_USAGE_TRANSFER_DST_BIT,
		properties: MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT,
		data:       null,
		data_size:  (sz)(MAP * MAP) * 4,
	)!;
}

fn MeshPush Rig.push(&self)
{
	return (MeshPush) {
		.frame     = (FrameBlock*)self.frame.get_address(),
		.instances = (Instance*)self.instances.get_address(),
		.draw      = (DrawRecord*)self.draw_record.get_address(),
	};
}

// Draw the cube once and hand back the attachment's bytes. The clear is blue
// and the fragment's blue is the tint, so the two cannot be confused.
fn char[] Rig.render(&self, Pipeline pipeline, PipelineLayout layout)
{
	MeshPush push = self.push();

	ClearValue[2] clear_values = {
		{ .color = { .float32 = { 0.0, 0.0, 1.0, 1.0 } } },
		{ .depthStencil = { .depth = FAR, .stencil = 0 } },
	};
	RenderPassBeginInfo begin = vk::renderPassBeginInfo()
		.setFramebuffer(self.colour.framebuffer)
		.setRenderPass(self.colour.pass)
		.setRenderArea({ .offset = { 0, 0 }, .extent = { MAP, MAP } })
		.setClearValues(clear_values[..]);

	vk::resetCommandBuffer(self.f.cmd, (vk::CommandBufferResetFlags)0)!!;
	vk::beginCommandBuffer(self.f.cmd, &&vk::commandBufferBeginInfo())!!;
	vk::cmdBeginRenderPass(self.f.cmd, &begin, SUBPASS_CONTENTS_INLINE);
	vk::cmdBindPipeline(self.f.cmd, PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	vk::cmdSetViewport(
		self.f.cmd,
		0,
		1,
		(Viewport[]) { { .width = MAP, .height = MAP, .minDepth = 0.0, .maxDepth = 1.0 } }
	);
	vk::cmdSetScissor(self.f.cmd, 0, 1, (Rect2D[]) { { .extent = { MAP, MAP }, .offset = { 0, 0 } } });
	vk::cmdPushConstants(
		self.f.cmd,
		layout,
		SHADER_STAGE_VERTEX_BIT,
		0,
		(uint)MeshPush::size,
		&push
	);
	// No vertex buffer and no index buffer: the vertex addresses are in the
	// draw record, which is the whole point of the pass.
	vk::cmdDraw(self.f.cmd, CUBE_VERTICES, 1, 0, 0);
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
		.imageExtent = { MAP, MAP, 1 },
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

	CommandBuffer[1] command_buffers = { self.f.cmd };
	SubmitInfo submit = vk::submitInfo().setCommandBuffers(command_buffers[..]);
	vk::queueSubmit(self.f.queue.handle, 1, &submit, self.f.fence)!!;
	vk::waitForFences(self.f.device, 1, &self.f.fence, vk::TRUE, ~(ulong)0)!!;

	// A copy on the heap rather than a view of the readback: the next render
	// overwrites that, and the caller frees this the ordinary way.
	char[] pixels = alloc::new_array(mem, char, (sz)(MAP * MAP) * 4);
	mem::copy(&pixels[0], self.readback.data(), (sz)(MAP * MAP) * 4);
	return pixels;
}

fn char* pixel_at(char[] pixels, uint x, uint y) @inline
{
	return &pixels[(y * MAP + x) * 4];
}

// Unorm quantisation may round either way; a byte either side of the value the
// float maps to is the same number with a different rounding mode, not a
// different varying.
fn bool byte_near(char got, float wanted) @inline
{
	float d = (float)(uint)got - wanted;
	if (d < 0.0f) d = -d;
	return d <= 1.5f;
}

fn bool bytes_apart(char a, char b, float wanted) @inline
{
	float d = (float)(uint)a - (float)(uint)b;
	if (d < 0.0f) d = -d;
	float e = d - wanted;
	if (e < 0.0f) e = -e;
	return e <= 1.5f;
}

// The frame as a binary PPM, so a failure can be looked at: header and bytes,
// the same format `capture_frame` in `src/main.c3` writes, for the same reason -
// nothing here has a PNG encoder.
fn void write_ppm(String path, char[] pixels)
{
	String header = string::tformat("P6\n%d %d\n255\n", MAP, MAP);
	char[] out = alloc::new_array(mem, char, (sz)(header.len + MAP * MAP * 3));
	mem::copy(out.ptr, header.ptr, header.len);
	for (uint i = 0; i < MAP * MAP; i++)
	{
		out[header.len + (usz)i * 3]     = pixels[i * 4];
		out[header.len + (usz)i * 3 + 1] = pixels[i * 4 + 1];
		out[header.len + (usz)i * 3 + 2] = pixels[i * 4 + 2];
	}
	file::save(path, out)!!;
	free(out);
	io::printfn("mesh: wrote %s", path);
}

// A pipeline over the colour pass, with the module's two entry points: the shape
// a mesh draw has, and the one the shadow dump deliberately does not.
fn Pipeline? mesh_pipeline_up(Fixture* f, RenderPass pass, PipelineLayout layout, ShaderModule shader)
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
	PipelineDepthStencilStateCreateInfo depth_stencil = vk::defaultDepthStencilState()
		.setDepthTestEnable(vk::TRUE)
		.setDepthWriteEnable(vk::TRUE)
		.setDepthCompareOp(COMPARE_OP_LESS_OR_EQUAL);

	DynamicState[2] dynamic_states = { DYNAMIC_STATE_VIEWPORT, DYNAMIC_STATE_SCISSOR };
	PipelineDynamicStateCreateInfo dynamic_state = vk::pipelineDynamicStateCreateInfo()
		.setDynamicStates(dynamic_states[..]);

	Viewport[1] viewports = { { .width = MAP, .height = MAP, .minDepth = 0.0, .maxDepth = 1.0 } };
	Rect2D[1] scissors = { { .extent = { MAP, MAP }, .offset = { 0, 0 } } };
	PipelineViewportStateCreateInfo viewport_state = vk::pipelineViewportStateCreateInfo()
		.setViewports(viewports[..])
		.setScissors(scissors[..]);

	// Blending off, all four channels written: the attachment is the shader's
	// output and nothing else, which is what makes a byte a varying.
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

fn void the_ported_vertex_path_draws_its_varyings() @test
{
	if (!file::is_file(MESH_MODULE))
	{
		io::printfn("mesh: no %s; skipped", MESH_MODULE);
		return;
	}

	Rig rig;
	rig_up(&rig)!!;
	defer rig.free();

	PipelineLayout layout = vk::pipelineLayoutCreateInfo()
		.setPushConstantRanges(
			{ { .stageFlags = SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = (uint)MeshPush::size } }
		)
		.build(rig.f.device)!!;
	defer vk::destroyPipelineLayout(rig.f.device, layout, null);

	ShaderModule shader = load_module(&rig.f, MESH_MODULE)!!;
	defer vk::destroyShaderModule(rig.f.device, shader, null);
	Pipeline pipeline = mesh_pipeline_up(&rig.f, rig.colour.pass, layout, shader)!!;
	defer vk::destroyPipeline(rig.f.device, pipeline, null);

	char[] pixels = rig.render(pipeline, layout);
	defer free(pixels);
	// Written before the checks, so a failure still leaves the frame to look at.
	write_ppm("build/mesh.ppm", pixels);

	// The clear, on a corner and on the margin beside the silhouette: blue,
	// where no fragment ran.
	char* outside = pixel_at(pixels, 0, 0);
	test::@check(
		outside[0] == 0 && outside[1] == 0 && (uint)outside[2] == 255,
		"the pixels the cube does not cover are the clear colour: %d, %d, %d",
		(int)(uint)outside[0], (int)(uint)outside[1], (int)(uint)outside[2]
	);
	char* margin = pixel_at(pixels, 4, MAP / 2);
	test::@check(
		margin[0] == 0 && margin[1] == 0 && (uint)margin[2] == 255,
		"and so is the margin beside the silhouette: %d, %d, %d",
		(int)(uint)margin[0], (int)(uint)margin[1], (int)(uint)margin[2]
	);

	// The face's uv corners are (0,0), (1,0), (1,1), (0,1), so its centre is
	// (0.5, 0.5) and nothing but the vertex stage could have put it there.
	char* centre = pixel_at(pixels, MAP / 2, MAP / 2);
	uint want_u = (uint)(u_at(MAP / 2) * 255.0f + 0.5f);
	uint want_v = (uint)(v_at(MAP / 2) * 255.0f + 0.5f);
	test::@check(
		byte_near(centre[0], u_at(MAP / 2) * 255.0f) && byte_near(centre[1], v_at(MAP / 2) * 255.0f),
		"the centre of the face carries its uv: %d, %d, wanted %d, %d",
		(int)(uint)centre[0], (int)(uint)centre[1], (int)want_u, (int)want_v
	);

	// The stub writes `input.color.g` to blue, and 0.25 is neither end of the
	// byte range, so a wrong channel or a lost instance cannot read as this.
	uint want_tint = (uint)(TINT_G * 255.0f + 0.5f);
	test::@check(
		byte_near(centre[2], TINT_G * 255.0f) && centre[2] != 0 && (uint)centre[2] != 255,
		"the blue channel is the instance colour's green: %d, wanted %d",
		(int)(uint)centre[2], (int)want_tint
	);

	// A quarter of the frame right of centre: u moves by the quarter of the
	// face's span that the pixel covers, while v, constant along the row, does
	// not move at all.
	char* quarter = pixel_at(pixels, MAP / 2 + MAP / 4, MAP / 2);
	uint want_du = (uint)((u_at(MAP / 2 + MAP / 4) - u_at(MAP / 2)) * 255.0f + 0.5f);
	test::@check(
		bytes_apart(quarter[0], centre[0], (u_at(MAP / 2 + MAP / 4) - u_at(MAP / 2)) * 255.0f),
		"a quarter across the face carries u from %d to %d, a change of about %d bytes",
		(int)(uint)centre[0], (int)(uint)quarter[0], (int)want_du
	);
	test::@check(
		bytes_apart(quarter[1], centre[1], 0.0f),
		"and v holds along the row: %d, %d",
		(int)(uint)quarter[1], (int)(uint)centre[1]
	);
	test::@check(
		byte_near(quarter[2], TINT_G * 255.0f),
		"the tint holds at a second pixel: %d, wanted %d",
		(int)(uint)quarter[2], (int)want_tint
	);

	// And a quarter of the frame down, which is the axis the row check holds
	// constant: v moves there and u does not. v grows upward, so the pixel below
	// centre is the smaller one.
	char* below = pixel_at(pixels, MAP / 2, MAP / 2 + MAP / 4);
	uint want_dv = (uint)((v_at(MAP / 2) - v_at(MAP / 2 + MAP / 4)) * 255.0f + 0.5f);
	test::@check(
		bytes_apart(below[1], centre[1], (v_at(MAP / 2) - v_at(MAP / 2 + MAP / 4)) * 255.0f),
		"a quarter down the face carries v from %d to %d, a change of about %d bytes",
		(int)(uint)centre[1], (int)(uint)below[1], (int)want_dv
	);
	test::@check(
		bytes_apart(below[0], centre[0], 0.0f),
		"and u holds down the column: %d, %d",
		(int)(uint)below[0], (int)(uint)centre[0]
	);
}
