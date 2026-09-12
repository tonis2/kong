<*
 three.c3's generated shadow modules, drawn as depth passes.

 The depth-only module has one vertex stage and no fragment stage at all, so the
 pass it feeds has no colour attachment: this test renders a cube into a `D32`
 image from an orthographic "light" and reads the depth back. Two numbers then
 say whether the port is right - the pixels the cube covers hold its depth, and
 the pixels it does not hold the clear value - which is a claim about the whole
 path from a device address through the wire structs to a clip-space position.

 The cut-out module is drawn the same way, with its one extra resource: a
 combined image sampler in a runtime array, bound at the set the engine's texture
 table uses. What the test asks of it is the part a depth pass cannot do - a
 fragment stage that discards decides whether the depth lands at all - so the
 cube is drawn twice, once keeping every texel and once cutting all of them.

 The geometry arrives through device addresses exactly as the engine's does:
 `DrawRecord.positions` is a pointer in a push block, the instance is a second
 pointer, and nothing is bound as a vertex buffer. An empty vertex input state is
 what makes that a real check rather than a coincidence.

 The structs below are three.c3's wire format (`scene/scene.c3`,
 `gpu/pipeline.c3`), declared by their offsets because that is what the generated
 module reads. A field this test never sets stays in the declaration: the stride
 is the contract, and one that ran short would put every read off target.
*>
module shadow_test;

import std::core::test;
import std::core::mem::allocators;
import std::io;
import std::io::file;
import std::math::matrix;
import std::math::vector;
import vk, shady, compute_support;

faultdef MODULE_DID_NOT_COMPILE;

const String SHADOW_MODULE = "../three.c3/build/shaderdef/shadow.shady";
const String CUTOUT_MODULE = "../three.c3/build/shaderdef/shadow_cutout.shady";
const uint MAP = 128;
const float FAR = 1.0;
const uint HAS_UVS = 2;

// `Instance` (`scene/scene.c3`): the model matrix this pass reads, and the pose
// fields it reads as zeroes.
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

// `DrawRecord` (`gpu/pipeline.c3`): the vertex streams this pass reaches through
// pointers, and the flags that decide which of them it reads.
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

	// The tallest uniform table column, clamped against by the material half.
	uint material_rows;
}

$assert DrawRecord::size == 392 : "DrawRecord no longer matches three.c3's";

// `ShadowPush` (`gpu/pipeline.c3`): the frame, the instance array, this bucket's
// record, and the light's clip space.
struct ShadowPush
{
	FrameBlock* frame;
	Instance* instances;
	DrawRecord* draw;
	uint[2] padding;
	Matrix4f projection;
}

$assert ShadowPush::size == 96 : "ShadowPush no longer matches three.c3's";

// `FrameBlock` is 512 bytes and this pass reads one field of it - the live pose
// pointer, and only for an instance whose pose says it is live. The rig below
// leaves that bit clear and hands over a zeroed block, so neither the field nor
// what it points at is ever read; only its address is. A byte array is the whole
// of what a stand-in needs to be.
const uint FRAME_BLOCK_SIZE = 512;

struct FrameBlock
{
	char[FRAME_BLOCK_SIZE] bytes;
}

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

// An orthographic light looking down -z: x and y scale the cube's [-1, 1] to
// [-0.5, 0.5] of the map, and z from 0 to 3 puts the near face at 1/3.
fn Matrix4f light_projection()
{
	return (Matrix4f) {
		.m = {
			0.5, 0.0, 0.0, 0.0,
			0.0, 0.5, 0.0, 0.0,
			0.0, 0.0, 1.0 / 3.0, 0.0,
			0.0, 0.0, 0.0, 1.0,
		}
	};
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

struct Depth
{
	Image image;
	Memory memory;
	ImageView view;
	RenderPass pass;
	Framebuffer framebuffer;
}

fn void Depth.free(&self, Device device)
{
	vk::destroyFramebuffer(device, self.framebuffer, null);
	vk::destroyRenderPass(device, self.pass, null);
	vk::destroyImageView(device, self.view, null);
	// `create_image_buffer` hands back a sub-allocation and its `free` destroys
	// the image too, so the image is not destroyed here as well.
	self.memory.free();
}

// A depth-only attachment: no colour, cleared to the far value, kept so the
// host can copy it out.
fn Depth? depth_up(Fixture* f)
{
	Image image = vk::imageCreateInfo()
		.setImageType(IMAGE_TYPE_2D)
		.setFormat(FORMAT_D32_SFLOAT)
		.setExtent({ .width = MAP, .height = MAP, .depth = 1 })
		.setMipLevels(1)
		.setArrayLayers(1)
		.setSamples(SAMPLE_COUNT_1_BIT)
		.setTiling(IMAGE_TILING_OPTIMAL)
		.setUsage(IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | IMAGE_USAGE_TRANSFER_SRC_BIT)
		.setInitialLayout(IMAGE_LAYOUT_UNDEFINED)
		.build(f.device)!;
	Memory memory = vk::create_image_buffer(image, &f.allocator, MEMORY_PROPERTY_DEVICE_LOCAL_BIT)!;

	ImageView view = vk::imageViewCreateInfo()
		.setImage(image)
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

	// One attachment, one subpass, no colour anywhere.
	AttachmentReference depth_attachment = {
		.attachment = 0,
		.layout     = IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
	};
	AttachmentDescription[1] attachments = {
		{
			.format         = FORMAT_D32_SFLOAT,
			.samples        = SAMPLE_COUNT_1_BIT,
			.loadOp         = ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp        = ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp  = ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout  = IMAGE_LAYOUT_UNDEFINED,
			.finalLayout    = IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		}
	};
	SubpassDescription[1] subpasses = {
		{
			.pipelineBindPoint       = PIPELINE_BIND_POINT_GRAPHICS,
			.pDepthStencilAttachment = &depth_attachment,
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
		.setWidth(MAP)
		.setHeight(MAP)
		.setLayers(1)
		.build(f.device)!;

	return (Depth) {
		.image       = image,
		.memory      = memory,
		.view        = view,
		.pass        = pass,
		.framebuffer = framebuffer,
	};
}

// The device, the depth map, and one cube reached by address, which both tests
// draw the same way.
struct Rig
{
	Fixture f;
	Depth depth;
	Memory positions;
	Memory uvs;
	Memory instances;
	Memory draw_record;
	Memory frame;
	Memory readback;
}

// The texture the cut-out module samples: one white texel, bound as a combined
// image sampler.
struct SamplerArgument
{
	Image image;
	Memory memory;
	ImageView image_view;
	Sampler sampler;
}

fn void SamplerArgument.free(&self, Device device)
{
	vk::destroySampler(device, self.sampler, null);
	vk::destroyImageView(device, self.image_view, null);
	self.memory.free();
}

// Everything the rig owns, released together. The fixture goes last: the
// `Memory` frees above and the depth image both need the device.
fn void Rig.free(&self)
{
	self.readback.free();
	self.frame.free();
	self.draw_record.free();
	self.instances.free();
	self.uvs.free();
	self.positions.free();
	self.depth.free(self.f.device);
	self.f.free();
}

// The rig is filled in place rather than returned: every allocation below keeps
// a pointer to the fixture's allocator, and a rig returned by value would leave
// those pointers describing a frame that has ended.
fn void? rig_up(Rig* rig)
{
	rig.f = compute_support::device_up()!;
	rig.depth = depth_up(&rig.f)!;

	float[3][CUBE_VERTICES] positions = cube_positions();
	rig.positions = compute_support::upload(&rig.f, &positions[0], CUBE_VERTICES * 12)!;

	// Every vertex samples the middle of its texture: one texel, one answer, so
	// a discarded pixel is the threshold's doing and not the sampler's.
	float[2][CUBE_VERTICES] uvs;
	for (usz i = 0; i < CUBE_VERTICES; i++)
	{
		uvs[i][0] = 0.5;
		uvs[i][1] = 0.5;
	}
	rig.uvs = compute_support::upload(&rig.f, &uvs[0], CUBE_VERTICES * 8)!;

	Instance[1] instances;
	instances[0].model = (Matrix4f) {
		.m = {
			1.0, 0.0, 0.0, 0.0,
			0.0, 1.0, 0.0, 0.0,
			0.0, 0.0, 1.0, 0.0,
			0.0, 0.0, 0.0, 1.0,
		}
	};
	rig.instances = compute_support::upload(&rig.f, &instances[0], Instance::size)!;

	DrawRecord[1] draws;
	draws[0].positions = (float[3]*)rig.positions.get_address();
	draws[0].uvs       = (float[2]*)rig.uvs.get_address();
	rig.draw_record = compute_support::upload(&rig.f, &draws[0], DrawRecord::size)!;

	char[FRAME_BLOCK_SIZE] frame = {};
	rig.frame = compute_support::upload(&rig.f, &frame[0], FRAME_BLOCK_SIZE)!;

	rig.readback = vk::new_buffer(
		allocator:  &rig.f.allocator,
		usage:      BUFFER_USAGE_TRANSFER_DST_BIT,
		properties: MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT,
		data:       null,
		data_size:  (sz)(MAP * MAP) * float::size,
	)!;
}

// The record the pass reads: the identity material, optionally with uvs, the
// alpha test's threshold, and the texture the cut-out module samples.
fn void Rig.set_draw(&self, uint flags, float base_alpha, float alpha_test)
{
	DrawRecord[1] draws;
	draws[0].positions     = (float[3]*)self.positions.get_address();
	draws[0].uvs           = (float[2]*)self.uvs.get_address();
	draws[0].base_color    = { 1.0, 1.0, 1.0, base_alpha };
	draws[0].flags         = flags;
	draws[0].skin_stride   = 1;
	draws[0].surface       = { 0.0, 0.0, 0.0, alpha_test };
	draws[0].uv_transform  = { 1.0, 1.0, 0.0, 0.0 };
	draws[0].texture_index[0] = 0;
	self.draw_record.upload(&draws[0], DrawRecord::size);
}

fn ShadowPush Rig.push(&self)
{
	return (ShadowPush) {
		.frame      = (FrameBlock*)self.frame.get_address(),
		.instances  = (Instance*)self.instances.get_address(),
		.draw       = (DrawRecord*)self.draw_record.get_address(),
		.projection = light_projection(),
	};
}

// Draw the cube once and hand back the depth map. The descriptor set is bound
// only when a pipeline has one, which is only the cut-out module.
fn float[] Rig.render(&self, Pipeline pipeline, PipelineLayout layout, DescriptorSet set)
{
	ShadowPush push = self.push();

	ClearValue[1] clear_values = { { .depthStencil = { .depth = FAR, .stencil = 0 } } };
	RenderPassBeginInfo begin = vk::renderPassBeginInfo()
		.setFramebuffer(self.depth.framebuffer)
		.setRenderPass(self.depth.pass)
		.setRenderArea({ .offset = { 0, 0 }, .extent = { MAP, MAP } })
		.setClearValues(clear_values[..]);

	vk::resetCommandBuffer(self.f.cmd, (vk::CommandBufferResetFlags)0)!!;
	vk::beginCommandBuffer(self.f.cmd, &&vk::commandBufferBeginInfo())!!;
	vk::cmdBeginRenderPass(self.f.cmd, &begin, SUBPASS_CONTENTS_INLINE);
	vk::cmdBindPipeline(self.f.cmd, PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	if (set != (DescriptorSet)null) vk::cmdBindDescriptorSets(self.f.cmd, PIPELINE_BIND_POINT_GRAPHICS, layout, 1, 1, &set, 0, null);
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
		SHADER_STAGE_VERTEX_BIT | SHADER_STAGE_FRAGMENT_BIT,
		0,
		(uint)ShadowPush::size,
		&push
	);
	// No vertex buffer and no index buffer: the vertex addresses are in the
	// draw record, which is the whole point of the pass.
	vk::cmdDraw(self.f.cmd, CUBE_VERTICES, 1, 0, 0);
	vk::cmdEndRenderPass(self.f.cmd);

	ImageMemoryBarrier2 to_src = {
		.sType            = STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.image            = self.depth.image,
		.srcStageMask     = PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
		.dstStageMask     = PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
		.srcAccessMask    = ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		.dstAccessMask    = ACCESS_2_TRANSFER_READ_BIT,
		.oldLayout        = IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		.newLayout        = IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		.subresourceRange = {
			.aspectMask     = IMAGE_ASPECT_DEPTH_BIT,
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
			.aspectMask     = IMAGE_ASPECT_DEPTH_BIT,
			.mipLevel       = 0,
			.baseArrayLayer = 0,
			.layerCount     = 1,
		},
		.imageExtent = { MAP, MAP, 1 },
	};
	vk::cmdCopyImageToBuffer(
		self.f.cmd,
		self.depth.image,
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
	float[] pixels = alloc::new_array(mem, float, MAP * MAP);
	mem::copy(&pixels[0], self.readback.data(), (sz)(MAP * MAP) * float::size);
	return pixels;
}

// A pipeline over the depth-only pass, with one entry point and no fragment
// stage: the shape three.c3's `SHADOW_ENTRIES` describes.
fn Pipeline? shadow_pipeline_up(Fixture* f, RenderPass pass, PipelineLayout layout, ShaderModule shader)
{
	PipelineShaderStageCreateInfo[1] stages = {
		vk::pipelineShaderStageCreateInfo()
			.setStage(SHADER_STAGE_VERTEX_BIT)
			.setMod(shader)
			.setName("vertexMain"),
	};
	return pipeline_up(f, pass, layout, stages[..]);
}

// The same pass with the module's fragment stage and its sampler: the shape
// `SHADOW_CUTOUT_ENTRIES` describes, and the only pipeline that has a layout
// with a descriptor in it.
fn Pipeline? cutout_pipeline_up(Fixture* f, RenderPass pass, PipelineLayout layout, ShaderModule shader)
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
	return pipeline_up(f, pass, layout, stages[..]);
}

// Everything both pipelines have in common: no vertex input, no culling, depth
// test and write on, and no colour attachment to blend into.
fn Pipeline? pipeline_up(Fixture* f, RenderPass pass, PipelineLayout layout, PipelineShaderStageCreateInfo[] stages)
{
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

	return vk::graphicsPipelineCreateInfo()
		.setStages(stages)
		.setLayout(layout)
		.setVertexInputState(&vertex_input)
		.setInputAssemblyState(&input_assembly)
		.setRasterizationState(&rasterization)
		.setMultisampleState(&multisample)
		.setDepthStencilState(&depth_stencil)
		// No colour blend state: the subpass has no colour attachment, and a
		// blend state names the attachments it would blend into.
		.setDynamicState(&dynamic_state)
		.setViewportState(&viewport_state)
		.setRenderPass(pass)
		.build(f.device);
}

<*
 A cube through the depth-only module: the covered pixels hold the near face's
 depth, the rest hold the clear value.
*>
fn void the_shadow_pass_renders_depth() @test
{
	if (!file::is_file(SHADOW_MODULE))
	{
		io::printfn("shadow: no %s; skipped", SHADOW_MODULE);
		return;
	}

	Rig rig;
	rig_up(&rig)!!;
	defer rig.free();
	rig.set_draw(flags: 0, base_alpha: 1.0, alpha_test: 0.0);

	PipelineLayout layout = vk::pipelineLayoutCreateInfo()
		.setPushConstantRanges(
			{ { .stageFlags = SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = (uint)ShadowPush::size } }
		)
		.build(rig.f.device)!!;
	defer vk::destroyPipelineLayout(rig.f.device, layout, null);

	ShaderModule shader = load_module(&rig.f, SHADOW_MODULE)!!;
	defer vk::destroyShaderModule(rig.f.device, shader, null);
	Pipeline pipeline = shadow_pipeline_up(&rig.f, rig.depth.pass, layout, shader)!!;
	defer vk::destroyPipeline(rig.f.device, pipeline, null);

	float[] pixels = rig.render(pipeline, layout, (DescriptorSet)null);
	defer free(pixels);

	float* centre = &pixels[(MAP / 2) * MAP + MAP / 2];
	// The near face is at z = 1 of the [0, 3] the projection maps to [0, 1].
	float wanted = 1.0 / 3.0;
	test::@check(compute_support::close(*centre, wanted),
		"the covered pixels hold the near face's depth: %f, wanted %f", *centre, wanted);
	test::@check(compute_support::close(pixels[0], FAR) && compute_support::close(pixels[MAP * MAP - 1], FAR),
		"and the uncovered ones hold the clear value: %f, %f", pixels[0], pixels[MAP * MAP - 1]);
	// The silhouette: the cube spans half the map each way, so the pixel a
	// quarter in is inside it and the one at the edge is not.
	test::@check(pixels[(MAP / 2) * MAP + MAP / 4] < FAR && pixels[(MAP / 2) * MAP] == FAR,
		"the silhouette is the size the projection says");
}

<*
 The cut-out module, drawn twice: a texel above the threshold keeps its depth and
 one below it is cut away, so the map is the cube and then the clear value.
*>
fn void the_cut_out_variant_discards() @test
{
	if (!file::is_file(CUTOUT_MODULE))
	{
		io::printfn("shadow: no %s; skipped", CUTOUT_MODULE);
		return;
	}

	Rig rig;
	rig_up(&rig)!!;
	defer rig.free();

	// The texture the fragment stage samples, as the engine binds it: one slot
	// of a combined image sampler array, in the set the texture table uses.
	SamplerArgument white = white_texture(&rig)!!;
	defer white.free(rig.f.device);
	DescriptorSetLayout sampler_layout = vk::descriptorSetLayoutCreateInfo()
		.setBindings(
			{
				{
					.binding         = 0,
					.descriptorType  = DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags      = SHADER_STAGE_FRAGMENT_BIT,
				}
			}
		)
		.build(rig.f.device)!!;
	defer vk::destroyDescriptorSetLayout(rig.f.device, sampler_layout, null);

	// Set 0 is unused, and a layout array is indexed by set number, so it needs
	// an empty layout to sit in that slot.
	DescriptorSetLayout empty_layout = vk::descriptorSetLayoutCreateInfo().build(rig.f.device)!!;
	defer vk::destroyDescriptorSetLayout(rig.f.device, empty_layout, null);

	DescriptorPool pool = vk::descriptorPoolCreateInfo()
		.setMaxSets(1)
		.setPoolSizes({ { .type = DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1 } })
		.build(rig.f.device)!!;
	defer vk::destroyDescriptorPool(rig.f.device, pool, null);

	DescriptorSet set;
	DescriptorSetLayout[1] set_layouts = { sampler_layout };
	vk::allocateDescriptorSets(
		rig.f.device,
		&&vk::descriptorSetAllocateInfo().setDescriptorPool(pool).setSetLayouts(set_layouts[..]),
		&set
	)!!;
	// The write keeps a pointer to the image info, so both arrays are named and
	// live until after the update.
	DescriptorImageInfo[1] image_info = {
		{
			.imageView   = white.image_view,
			.imageLayout = IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.sampler     = white.sampler,
		}
	};
	WriteDescriptorSet[1] writes = {
		vk::writeDescriptorSet()
			.setDstSet(set)
			.setDstBinding(0)
			.setDescriptorType(DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
			.setImageInfo(image_info[..])
	};
	vk::updateDescriptorSets(rig.f.device, 1, &writes[0], 0, null);

	PipelineLayout layout = vk::pipelineLayoutCreateInfo()
		.setSetLayouts({ empty_layout, sampler_layout })
		.setPushConstantRanges(
			{ { .stageFlags = SHADER_STAGE_VERTEX_BIT | SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = (uint)ShadowPush::size } }
		)
		.build(rig.f.device)!!;
	defer vk::destroyPipelineLayout(rig.f.device, layout, null);

	ShaderModule shader = load_module(&rig.f, CUTOUT_MODULE)!!;
	defer vk::destroyShaderModule(rig.f.device, shader, null);
	Pipeline pipeline = cutout_pipeline_up(&rig.f, rig.depth.pass, layout, shader)!!;
	defer vk::destroyPipeline(rig.f.device, pipeline, null);

	// Kept: the sampled alpha is one and the threshold is a half.
	rig.set_draw(flags: HAS_UVS, base_alpha: 1.0, alpha_test: 0.5);
	float[] kept = rig.render(pipeline, layout, set);
	defer free(kept);
	float centre_wanted = 1.0 / 3.0;
	test::@check(compute_support::close(kept[(MAP / 2) * MAP + MAP / 2], centre_wanted),
		"a texel above the threshold keeps its depth: %f", kept[(MAP / 2) * MAP + MAP / 2]);

	// Cut: the surface's own alpha is zero, so every fragment of the cube
	// discards and the map is what the clear left.
	rig.set_draw(flags: HAS_UVS, base_alpha: 0.0, alpha_test: 0.5);
	float[] cut = rig.render(pipeline, layout, set);
	defer free(cut);
	test::@check(compute_support::close(cut[(MAP / 2) * MAP + MAP / 2], FAR),
		"and a texel below it is cut away: %f", cut[(MAP / 2) * MAP + MAP / 2]);
}

fn SamplerArgument? white_texture(Rig* rig)
{
	char[4] white = { 255, 255, 255, 255 };
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
		.build(rig.f.device)!!;
	Memory memory = vk::create_image_buffer(image, &rig.f.allocator, MEMORY_PROPERTY_DEVICE_LOCAL_BIT)!!;

	Memory stage = compute_support::upload(&rig.f, &white[0], 4)!!;
	defer stage.free();

	ImageMemoryBarrier2[1] to_transfer = {
		{
			.sType            = STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.image            = image,
			.srcStageMask     = PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
			.dstStageMask     = PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
			.srcAccessMask    = (AccessFlags2)0,
			.dstAccessMask    = ACCESS_2_TRANSFER_WRITE_BIT,
			.oldLayout        = IMAGE_LAYOUT_UNDEFINED,
			.newLayout        = IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.subresourceRange = {
				.aspectMask     = IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel   = 0,
				.levelCount     = 1,
				.baseArrayLayer = 0,
				.layerCount     = 1,
			},
		}
	};
	ImageMemoryBarrier2[1] to_shader = { {
		.sType            = STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.image            = image,
		.srcStageMask     = PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
		.dstStageMask     = PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
		.srcAccessMask    = ACCESS_2_TRANSFER_WRITE_BIT,
		.dstAccessMask    = ACCESS_2_SHADER_SAMPLED_READ_BIT,
		.oldLayout        = IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.newLayout        = IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		.subresourceRange = {
			.aspectMask     = IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel   = 0,
			.levelCount     = 1,
			.baseArrayLayer = 0,
			.layerCount     = 1,
		},
	} };

	BufferImageCopy region = {
		.imageSubresource = { .aspectMask = IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
		.imageExtent      = { 1, 1, 1 },
	};

	vk::resetCommandBuffer(rig.f.cmd, (vk::CommandBufferResetFlags)0)!!;
	vk::beginCommandBuffer(rig.f.cmd, &&vk::commandBufferBeginInfo())!!;
	vk::cmdPipelineBarrier2(rig.f.cmd, &&vk::dependencyInfo().setImageMemoryBarriers(to_transfer[..]));
	vk::cmdCopyBufferToImage(
		rig.f.cmd,
		stage.buffer,
		image,
		IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1,
		&region
	);
	vk::cmdPipelineBarrier2(rig.f.cmd, &&vk::dependencyInfo().setImageMemoryBarriers(to_shader[..]));
	vk::endCommandBuffer(rig.f.cmd)!!;

	CommandBuffer[1] command_buffers = { rig.f.cmd };
	SubmitInfo submit = vk::submitInfo().setCommandBuffers(command_buffers[..]);
	vk::queueSubmit(rig.f.queue.handle, 1, &submit, rig.f.fence)!!;
	vk::waitForFences(rig.f.device, 1, &rig.f.fence, vk::TRUE, ~(ulong)0)!!;

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
		.build(rig.f.device)!!;
	Sampler sampler = vk::samplerCreateInfo().build(rig.f.device)!!;

	return (SamplerArgument) {
		.image      = image,
		.memory     = memory,
		.image_view = view,
		.sampler    = sampler,
	};
}
