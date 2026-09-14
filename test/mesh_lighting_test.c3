<*
 three.c3's generated lit mesh module, drawn and read back.

 The port under test is `shaders/lighting.shady`, reached through
 `build/shaderdef/mesh_lit.shady`: the geometry half is the same ported vertex
 path `mesh_test` draws, and the fragment half is the one `shader_def_test`'s
 `lit_material` writes - albedo is instance colour times vertex colour,
 roughness 0.5, metalness 0, reflectance 0.5. The port deliberately has no
 shadows, no area lights and no baked light yet, so one punctual light and no
 shadow rig make every expectation exact rather than approximate.

 Three frames, each asking the lighting maths rather than the geometry:

  - a directional light lights the face it faces, and its colour is the colour
    that reaches the pixels; turned away from the same face the result is black,
    which is a refused light rather than a draw that never ran;
  - a point light reaches the face through the list the cluster dispatch wrote,
    and moving that light out of the grid's near slices - while it still reaches
    the face - leaves the face black, which is what proves the fragment shades
    its cluster's list and not the frame's whole one;
  - with `cluster_grid.w == 0` there is no cluster grid and the shader packs the
    overflow bit itself, so the whole list lights the face with no dispatch at
    all.

 The rig is `mesh_test`'s: a cube whose +z face is the only one a pixel sees, a
 colour attachment read back through a host-visible buffer, and the geometry
 reached through device addresses in the push block. The clear colour is blue,
 so black pixels are fragments that ran and contributed nothing. The projection
 is drawn through the engine's own negative-height viewport, because that is
 what the culling pass's cells are written for.

 The structs below are three.c3's wire format (`scene/scene.c3`,
 `gpu/pipeline.c3`), declared by their offsets because that is what the
 generated module reads. A field this test never sets stays in the declaration:
 the stride is the contract, and one that ran short would put every read off
 target.
*>
module mesh_lighting_test;

import std::core::test;
import std::core::mem::allocators;
import std::io;
import std::io::file;
import std::math::matrix;
import std::math::vector;
import vk, shady, compute_support;

faultdef MODULE_DID_NOT_COMPILE;

const String MESH_LIT_MODULE = "../three.c3/build/shaderdef/mesh_lit.shady";
const String CLUSTER_MODULE  = "../three.c3/build/shaderdef/cluster.shady";
const uint MAP = 128;
const float FAR = 1.0;
// The cube spans 80% of the frame each way, so the clear colour frames it.
const float SCALE = 0.8;
// Where the specular term thinks the camera is; the projection is orthographic,
// so this is the whole of the camera the lighting reads.
const float EYE_Z = 2.0;
const uint CUBE_VERTICES = 36;
const uint HAS_NORMALS = 1;
const uint HAS_COLORS  = 4;
const uint GRID_X = 16;
const uint GRID_Y = 9;
const uint GRID_Z = 24;
const uint CLUSTER_CAPACITY = 64;
const uint CLUSTER_COUNT = GRID_X * GRID_Y * GRID_Z;
const uint WORKGROUP = 64;
const uint LIGHT_POINT = 1;
// The cluster the face's centre pixel reads: the face is centred, so cell
// (8, 4), and it sits nearer than the near plane, which clamps it into slice 0.
const uint FACE_CENTRE_CLUSTER = 8 + GRID_X * 4;

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
// it reads, and the material the fragment half shades. The generated module's
// copy is walked from the C3 struct in declaration order, so a field skipped or
// moved reads some other field's bytes.
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

// `Light` (`gpu/pipeline.c3`), whose stride is its size. `meta.z` is the kind;
// a directional light has `direction.w == 0` and xyz as the surface-to-light
// direction, a point light xyz as a world position and w as its range.
struct Light
{
	float[4] direction;
	float[4] color;
	float[4] area_right;
	float[4] area_up;
	uint[4] meta;
}

$assert Light::size == 80 : "Light no longer matches three.c3's";

// `FrameBlock` (`gpu/pipeline.c3`), which the fragment half reads for the
// camera, the light list and the cluster the culling pass filled. Every field
// above the ones this test sets is kept so the ones it does set land on the
// offsets the shader reads.
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
	Light* lights;
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

// `Uniforms`, the material's declared uniform block: the generator emits it
// from the declared list and the push block carries one. This lit build declares
// `tint` and eight colour rows, in the order the module walks them.
struct Uniforms
{
	float[4] tint;
	float[4][8] colors;
}

$assert Uniforms::size == 144 : "Uniforms no longer matches three.c3's";

// `MeshPush` (`gpu/pipeline.c3`): the frame, the instance array, this bucket's
// record, and the material's uniforms. The whole of what a mesh draw pushes.
struct MeshPush
{
	FrameBlock* frame;
	Instance* instances;
	DrawRecord* draw;
	Uniforms uniforms;
}

$assert MeshPush::size == 168 : "MeshPush no longer matches three.c3's";

// `ClusterPush` (`gpu/pipeline.c3`), as the host fills it: device addresses
// rather than the pointers the shader sees.
struct ClusterPush
{
	DeviceAddress lights;
	DeviceAddress counts;
	DeviceAddress indices;
	DeviceAddress overflow;
	Matrix4f view_projection;
	uint light_count;
	float[4] camera_near; // xyz eye, w near
	float far_plane;
}

$assert ClusterPush::size == 120 : "ClusterPush no longer matches three.c3's";

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

// A normal per vertex, in `cube_positions`' own face order. Only the +z face
// reaches a pixel, but a wrong stream would move its normal first.
fn float[3][CUBE_VERTICES] cube_normals()
{
	float[3][6] faces = {
		{ 0.0, 0.0, 1.0 },  // +z
		{ 0.0, 0.0, -1.0 }, // -z
		{ 0.0, 1.0, 0.0 },  // +y
		{ 0.0, -1.0, 0.0 }, // -y
		{ 1.0, 0.0, 0.0 },  // +x
		{ -1.0, 0.0, 0.0 }, // -x
	};

	float[3][CUBE_VERTICES] normals;
	for (usz i = 0; i < CUBE_VERTICES; i++) normals[i] = faces[i / 6];
	return normals;
}

// An orthographic camera on the +z face: x and y scale the cube's [-1, 1] to
// [-0.8, 0.8] of the frame, and z puts the near face at 0 and the far one at 1.
// Unlike `mesh_test`'s copy, y is not negated - the viewport below is the
// engine's negative-height one, and the two flips would cancel.
fn Matrix4f camera_projection()
{
	return (Matrix4f) {
		.m = {
			SCALE, 0.0, 0.0, 0.0,
			0.0, SCALE, 0.0, 0.0,
			0.0, 0.0, -0.5, 0.0,
			0.0, 0.0, 0.5, 1.0,
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

// The device, the colour target, one cube with normals and colours reached by
// address, the frame's light and cluster storage, and the readback buffer.
struct Rig
{
	Fixture f;
	Colour colour;
	Memory positions;
	Memory normals;
	Memory colors;
	Memory instances;
	Memory draw_record;
	Memory lights;
	Memory frame;
	Memory counts;
	Memory indices;
	Memory overflow;
	Memory readback;
}

// Everything the rig owns, released together. The fixture goes last: the
// `Memory` frees above and the images both need the device.
fn void Rig.free(&self)
{
	self.readback.free();
	self.overflow.free();
	self.indices.free();
	self.counts.free();
	self.frame.free();
	self.lights.free();
	self.draw_record.free();
	self.instances.free();
	self.colors.free();
	self.normals.free();
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

	float[3][CUBE_VERTICES] normals = cube_normals();
	rig.normals = compute_support::upload(&rig.f, &normals[0], CUBE_VERTICES * 12)!;

	// White, so the fragment's albedo is the instance's own colour and the
	// vertex stream tints nothing.
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
	// The same tint `mesh_test` gives its cube, so this is that surface lit and
	// a channel swapped anywhere in between would show.
	// White, and so is every vertex colour: albedo is the product of the two, and
	// what these tests read out of the pixels is the *light* - an albedo with a
	// hole in a channel would be a hue nothing could tell from the light's own.
	instances[0].color = { 1.0, 1.0, 1.0, 1.0 };
	rig.instances = compute_support::upload(&rig.f, &instances[0], Instance::size)!;

	// Zero-filled first: the material fields this test does not choose -
	// emissive, the uv variant table, the texture indices - are a material
	// declaring none rather than bytes the fragment would still read.
	DrawRecord[1] draws = {};
	draws[0].positions     = (float[3]*)rig.positions.get_address();
	draws[0].normals       = (float[3]*)rig.normals.get_address();
	draws[0].colors        = (float[4]*)rig.colors.get_address();
	draws[0].base_color    = { 1.0, 1.0, 1.0, 1.0 };
	draws[0].flags         = HAS_NORMALS | HAS_COLORS;
	draws[0].uv_transform  = { 1.0, 1.0, 0.0, 0.0 };
	// The material the header describes - roughness 0.5, metalness 0,
	// reflectance 0.5 - which is what the specular term is computed from and
	// what every expectation below assumes.
	draws[0].surface       = { 0.5, 0.0, 0.5, 0.0 };
	// The morph loop's whole test: zero targets and zero weights mean the null
	// pointers below it stay unread.
	draws[0].morph_targets = 0;
	rig.draw_record = compute_support::upload(&rig.f, &draws[0], DrawRecord::size)!;

	Light[2] lights = {};
	// 0xffffffff is the "no shadow view" index: without it the fragment half
	// would follow `lights[0].meta.y` to a view that is not there.
	lights[0].meta[1] = 0xffffffff;
	lights[1].meta[1] = 0xffffffff;
	rig.lights = compute_support::upload(&rig.f, &lights[0], (sz)lights.len * Light::size)!;

	FrameBlock[1] frame = {};
	rig.frame = compute_support::upload(&rig.f, &frame[0], FrameBlock::size)!;

	// The culling pass writes through these addresses, so they are allocated
	// once and zeroed: a frame with no dispatch left every count at zero.
	uint[] counts = alloc::new_array(tmem, uint, CLUSTER_COUNT);
	uint[] indices = alloc::new_array(tmem, uint, (usz)CLUSTER_COUNT * CLUSTER_CAPACITY);
	uint[] overflow = alloc::new_array(tmem, uint, 1);
	foreach (&count : counts) *count = 0;
	foreach (&index : indices) *index = 0;
	foreach (&value : overflow) *value = 0;
	rig.counts = compute_support::upload(&rig.f, &counts[0], (sz)counts.len * uint::size)!;
	rig.indices = compute_support::upload(&rig.f, &indices[0], (sz)indices.len * uint::size)!;
	rig.overflow = compute_support::upload(&rig.f, &overflow[0], uint::size)!;

	rig.readback = vk::new_buffer(
		allocator:  &rig.f.allocator,
		usage:      BUFFER_USAGE_TRANSFER_DST_BIT,
		properties: MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT,
		data:       null,
		data_size:  (sz)(MAP * MAP) * 4,
	)!;
}

// Light 0 is the slot this test leaves zeroed; the cluster dispatch skips it,
// the way the engine's own list starts with a light no cluster references.
fn void Rig.set_lights(&self, Light[] lights)
{
	self.lights.upload(&lights[0], (sz)lights.len * Light::size);
}

// The frame block one render reads: the camera, this frame's light count, and
// the grid the fragment looks its cluster up in. `capacity` is the grid's w -
// zero is the renderer's "no cluster grid", and the shader then walks every
// light instead.
fn void Rig.set_frame(&self, float light_count, uint capacity, float near_plane, float far_plane)
{
	FrameBlock[1] frame = {};
	frame[0].view_projection = camera_projection();
	frame[0].lighting = { 0.0, light_count, 0.0, 0.0 };
	frame[0].camera = { 0.0, 0.0, EYE_Z, 0.0 };
	frame[0].lights = (Light*)self.lights.get_address();
	frame[0].cluster_counts = (uint*)self.counts.get_address();
	frame[0].cluster_indices = (uint*)self.indices.get_address();
	frame[0].cluster_grid = { GRID_X, GRID_Y, GRID_Z, capacity };
	frame[0].cluster_screen = { (float)MAP, (float)MAP, near_plane, far_plane };
	self.frame.upload(&frame[0], FrameBlock::size);
}

// The dispatch's own push block, over the same matrix and buffers the frame
// block names - the culling has to be for the camera this render uses.
fn ClusterPush Rig.cluster_push(&self, float near_plane, float far_plane)
{
	return (ClusterPush) {
		.lights          = self.lights.get_address(),
		.counts          = self.counts.get_address(),
		.indices         = self.indices.get_address(),
		.overflow        = self.overflow.get_address(),
		.view_projection = camera_projection(),
		.light_count     = 2,
		.camera_near     = { 0.0, 0.0, EYE_Z, near_plane },
		.far_plane       = far_plane,
	};
}

fn MeshPush Rig.push(&self)
{
	return (MeshPush) {
		.frame     = (FrameBlock*)self.frame.get_address(),
		.instances = (Instance*)self.instances.get_address(),
		.draw      = (DrawRecord*)self.draw_record.get_address(),
	};
}

struct ClusterPass
{
	Pipeline pipeline;
	PipelineLayout layout;
	ClusterPush push;
}

fn void ClusterPass.free(&self, Device device)
{
	vk::destroyPipeline(device, self.pipeline, null);
	vk::destroyPipelineLayout(device, self.layout, null);
}

// The cluster module as a compute pipeline, the shape `cluster_test` builds.
fn ClusterPass? cluster_pass_up(Fixture* f)
{
	ShaderModule shader = load_module(f, CLUSTER_MODULE)!;

	PipelineLayout layout = vk::pipelineLayoutCreateInfo()
		.setPushConstantRanges(
			{ { .stageFlags = SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = (uint)ClusterPush::size } }
		)
		.build(f.device)!;

	ComputePipelineCreateInfo info = vk::computePipelineCreateInfo()
		.setStage(
			vk::pipelineShaderStageCreateInfo()
				.setStage(SHADER_STAGE_COMPUTE_BIT)
				.setMod(shader)
				.setName("computeMain")
		)
		.setLayout(layout);
	Pipeline pipeline;
	vk::createComputePipelines(f.device, (PipelineCache)null, 1, &info, null, &pipeline)!;
	vk::destroyShaderModule(f.device, shader, null);

	return (ClusterPass) { .pipeline = pipeline, .layout = layout };
}

// Draw the cube once and hand back the attachment's bytes. When `bin` is given
// the cluster dispatch runs in the same command buffer immediately before the
// pass, because the fragment reads what it wrote through the frame block's
// addresses: the barrier after it is what orders the two.
fn char[] Rig.render(&self, Pipeline pipeline, PipelineLayout layout, ClusterPass* bin, compute_support::MaterialSet* materials)
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

	if (bin != null)
	{
		vk::cmdBindPipeline(self.f.cmd, PIPELINE_BIND_POINT_COMPUTE, bin.pipeline);
		vk::cmdPushConstants(self.f.cmd, bin.layout, SHADER_STAGE_COMPUTE_BIT, 0, (uint)ClusterPush::size, &bin.push);
		vk::cmdDispatch(self.f.cmd, (CLUSTER_COUNT + WORKGROUP - 1) / WORKGROUP, 1, 1);

		// Both readers of the list: the fragment stage that shades with it, and
		// the host checks below.
		vk::MemoryBarrier visible = {
			.srcAccessMask = ACCESS_SHADER_WRITE_BIT,
			.dstAccessMask = ACCESS_SHADER_READ_BIT | ACCESS_HOST_READ_BIT,
		};
		vk::cmdPipelineBarrier(
			self.f.cmd,
			PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			PIPELINE_STAGE_FRAGMENT_SHADER_BIT | PIPELINE_STAGE_HOST_BIT,
			(DependencyFlags)0,
			1,
			&visible,
			0,
			null,
			0,
			null
		);
	}

	vk::cmdBeginRenderPass(self.f.cmd, &begin, SUBPASS_CONTENTS_INLINE);
	vk::cmdBindPipeline(self.f.cmd, PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	// Set 0 is the frame samplers the generated module declares whether or not
	// this frame has shadows; set 1 is the device-wide texture array.
	DescriptorSet[2] sets = { materials.shadow_set, materials.texture_set };
	vk::cmdBindDescriptorSets(self.f.cmd, PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 2, &sets[0], 0, null);
	// Negative height, the engine's own flip: clip-space +Y lands at the top of
	// the framebuffer, which is the way the culling pass's cells are computed.
	vk::cmdSetViewport(
		self.f.cmd,
		0,
		1,
		(Viewport[]) {
			{
				.x = 0.0,
				.y = (float)MAP,
				.width = (float)MAP,
				.height = -(float)MAP,
				.minDepth = 0.0,
				.maxDepth = 1.0,
			}
		}
	);
	vk::cmdSetScissor(self.f.cmd, 0, 1, (Rect2D[]) { { .extent = { MAP, MAP }, .offset = { 0, 0 } } });
	// Both stages: the vertex stage reads the push block's pointers, and the
	// fragment stage reads `push.frame`, which the unlit stub did not.
	vk::cmdPushConstants(
		self.f.cmd,
		layout,
		SHADER_STAGE_VERTEX_BIT | SHADER_STAGE_FRAGMENT_BIT,
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
// different light.
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
	io::printfn("mesh_lit: wrote %s", path);
}

// A pipeline over the colour pass with the module's two entry points, one of
// which is now a fragment stage that shades rather than the unlit dump's stub.
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
	// output and nothing else, which is what makes a byte the lighting maths.
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

// The face the camera sees is +z at world z = 1, so the surface-to-light
// direction of a light on the camera's side is (0, 0, 1).
fn void a_directional_light_lights_only_the_face_it_faces() @test
{
	if (!file::is_file(MESH_LIT_MODULE))
	{
		io::printfn("mesh_lit: no %s; skipped", MESH_LIT_MODULE);
		return;
	}

	Rig rig;
	rig_up(&rig)!!;
	defer rig.free();

	compute_support::MaterialSet materials = compute_support::material_set_up(&rig.f)!!;
	defer materials.free();

	// Both stages: the vertex stage reads the push block's pointers and the
	// fragment stage reads `push.frame`.
	PipelineLayout layout = vk::pipelineLayoutCreateInfo()
		.setSetLayouts({ materials.shadow, materials.textures })
		.setPushConstantRanges(
			{
				{
					.stageFlags = SHADER_STAGE_VERTEX_BIT | SHADER_STAGE_FRAGMENT_BIT,
					.offset = 0,
					.size = (uint)MeshPush::size,
				}
			}
		)
		.build(rig.f.device)!!;
	defer vk::destroyPipelineLayout(rig.f.device, layout, null);

	ShaderModule shader = load_module(&rig.f, MESH_LIT_MODULE)!!;
	defer vk::destroyShaderModule(rig.f.device, shader, null);
	Pipeline pipeline = mesh_pipeline_up(&rig.f, rig.colour.pass, layout, shader)!!;
	defer vk::destroyPipeline(rig.f.device, pipeline, null);

	// A light whose colour is not white, from the camera's side of the only
	// face the camera sees. `direction.w == 0` is directional, and xyz is the
	// surface-to-light direction.
	Light[2] lights = {};
	lights[0].direction = { 0.0, 0.0, 1.0, 0.0 };
	lights[0].color = { 1.0, 0.5, 0.0, 1.0 };
	lights[0].meta[1] = 0xffffffff;
	rig.set_lights(lights[..]);
	// No punctual light is present, so the cluster list - zeroed - is never the
	// reason for anything below.
	rig.set_frame(light_count: 1.0, capacity: CLUSTER_CAPACITY, near_plane: 0.5, far_plane: 4.0);

	char[] lit = rig.render(pipeline, layout, null, &materials);
	defer free(lit);
	// Written before the checks, so a failure still leaves the frame to look at.
	write_ppm("build/mesh_lit_directional.ppm", lit);

	// The centre of the face: every term carries the light's own colour, since
	// `energy` scales the highlight as well as the diffuse. Red clamps, green
	// is the light's 0.5 times one plus the highlight - 0.05086 at the stub's
	// roughness 0.5 and reflectance 0.5 - and blue is exactly zero because the
	// light has none.
	char* centre = pixel_at(lit, MAP / 2, MAP / 2);
	test::@check(
		(uint)centre[0] == 255 && (uint)centre[0] > (uint)centre[1] && (uint)centre[1] > (uint)centre[2],
		"the lit face carries the light's colour: %d, %d, %d",
		(int)(uint)centre[0],
		(int)(uint)centre[1],
		(int)(uint)centre[2]
	);
	test::@check(
		byte_near(centre[1], 0.52543f * 255.0f) && centre[2] == 0,
		"green is the light's share plus the tinted highlight and blue is zero: %d, %d, wanted %d, 0",
		(int)(uint)centre[1],
		(int)(uint)centre[2],
		(int)(0.52543f * 255.0f + 0.5f)
	);

	// The same light turned around: `dot(normal, toward)` is negative, so the
	// shader refuses it. Black rather than the clear blue is a fragment that
	// ran and added nothing.
	lights[0].direction = { 0.0, 0.0, -1.0, 0.0 };
	rig.set_lights(lights[..]);
	char[] away = rig.render(pipeline, layout, null, &materials);
	defer free(away);
	write_ppm("build/mesh_lit_directional_away.ppm", away);

	char* dark = pixel_at(away, MAP / 2, MAP / 2);
	char* margin = pixel_at(away, 4, MAP / 2);
	test::@check(
		dark[0] == 0 && dark[1] == 0 && dark[2] == 0,
		"the face the light points away from is black: %d, %d, %d",
		(int)(uint)dark[0],
		(int)(uint)dark[1],
		(int)(uint)dark[2]
	);
	test::@check(
		margin[0] == 0 && margin[1] == 0 && (uint)margin[2] == 255,
		"and the clear colour beside it is still blue: %d, %d, %d",
		(int)(uint)margin[0],
		(int)(uint)margin[1],
		(int)(uint)margin[2]
	);
}

// A point light just in front of the visible face, inside the range that
// reaches it, found by the shader through the cluster the dispatch wrote.
fn void a_point_light_reaches_a_fragment_through_its_cluster() @test
{
	if (!file::is_file(MESH_LIT_MODULE) || !file::is_file(CLUSTER_MODULE))
	{
		io::printfn("mesh_lit: no generated modules; skipped");
		return;
	}

	Rig rig;
	rig_up(&rig)!!;
	defer rig.free();

	compute_support::MaterialSet materials = compute_support::material_set_up(&rig.f)!!;
	defer materials.free();

	PipelineLayout layout = vk::pipelineLayoutCreateInfo()
		.setSetLayouts({ materials.shadow, materials.textures })
		.setPushConstantRanges(
			{
				{
					.stageFlags = SHADER_STAGE_VERTEX_BIT | SHADER_STAGE_FRAGMENT_BIT,
					.offset = 0,
					.size = (uint)MeshPush::size,
				}
			}
		)
		.build(rig.f.device)!!;
	defer vk::destroyPipelineLayout(rig.f.device, layout, null);

	ShaderModule shader = load_module(&rig.f, MESH_LIT_MODULE)!!;
	defer vk::destroyShaderModule(rig.f.device, shader, null);
	Pipeline pipeline = mesh_pipeline_up(&rig.f, rig.colour.pass, layout, shader)!!;
	defer vk::destroyPipeline(rig.f.device, pipeline, null);

	ClusterPass cluster = cluster_pass_up(&rig.f)!!;
	defer cluster.free(rig.f.device);

	// Light 1 is the point light: xyz is its world position, w its range. The
	// culling pass starts its loop at 1, so light 0 stays the slot the engine's
	// own list leaves for the sun - with no view, which is what 0xffffffff says.
	Light[2] lights = {};
	lights[0].meta[1] = 0xffffffff;
	lights[1].direction = { 0.0, 0.0, 1.8, 1.1 };
	lights[1].color = { 0.0, 1.0, 0.5, 1.0 };
	lights[1].meta = { 0, 0xffffffff, LIGHT_POINT, 0 };
	rig.set_lights(lights[..]);

	// The face is at distance 1.0 and the near plane at 1.2, so the fragment's
	// distance clamps into slice 0; the light at 0.2 from the eye reaches into
	// that slice.
	rig.set_frame(light_count: 2.0, capacity: CLUSTER_CAPACITY, near_plane: 1.2, far_plane: 4.0);

	ClusterPass bin = {
		.pipeline = cluster.pipeline,
		.layout   = cluster.layout,
		.push     = rig.cluster_push(1.2, 4.0),
	};
	char[] lit = rig.render(pipeline, layout, &bin, &materials);
	defer free(lit);
	write_ppm("build/mesh_lit_point.ppm", lit);

	// The premise of the black frame below: the culler binned light 1 - the
	// index, not the light - in the cluster the centre pixel reads.
	uint* counts = (uint*)rig.counts.data();
	uint* indices = (uint*)rig.indices.data();
	test::@check(
		counts[FACE_CENTRE_CLUSTER] == 1 && indices[(usz)FACE_CENTRE_CLUSTER * CLUSTER_CAPACITY] == 1,
		"the culler put the light in the face's cluster: count %d, first %d",
		counts[FACE_CENTRE_CLUSTER],
		indices[(usz)FACE_CENTRE_CLUSTER * CLUSTER_CAPACITY]
	);

	// Green is 1.0 and blue 0.5 of the light, each times the light's reach at
	// the face - about 0.81, the light being 0.8 away with a range of 1.1 - and
	// times one and the highlight together, `nl` being all but one.
	char* centre = pixel_at(lit, MAP / 2, MAP / 2);
	test::@check(
		(uint)centre[1] > 200 && (uint)centre[1] > (uint)centre[2] && (uint)centre[2] > (uint)centre[0],
		"the face is lit in the light's hue: %d, %d, %d",
		(int)(uint)centre[0],
		(int)(uint)centre[1],
		(int)(uint)centre[2]
	);
	test::@check(
		byte_near(centre[1], 0.85075f * 255.0f) && byte_near(centre[2], 0.42538f * 255.0f),
		"green and blue are the light's hue times its reach at the face: %d, %d, wanted %d, %d",
		(int)(uint)centre[1],
		(int)(uint)centre[2],
		(int)(0.85075f * 255.0f + 0.5f),
		(int)(0.42538f * 255.0f + 0.5f)
	);

	// The same light moved to 0.19 from the eye: still in front of the face and
	// still reaching it - 0.81 away, range 1.0 - but its sphere ends before the
	// grid's first slice, so the culler lists it nowhere. The frame's whole list
	// would light the face green at about 0.52, so black is what proves the
	// fragment walked its cluster's list and not that one.
	lights[1].direction = { 0.0, 0.0, 1.81, 1.0 };
	rig.set_lights(lights[..]);
	ClusterPass moved = {
		.pipeline = cluster.pipeline,
		.layout   = cluster.layout,
		.push     = rig.cluster_push(1.2, 4.0),
	};
	char[] dark = rig.render(pipeline, layout, &moved, &materials);
	defer free(dark);
	write_ppm("build/mesh_lit_point_moved.ppm", dark);

	test::@check(
		counts[FACE_CENTRE_CLUSTER] == 0,
		"the moved light is in no slice of the face's cluster: count %d",
		counts[FACE_CENTRE_CLUSTER]
	);
	char* dark_centre = pixel_at(dark, MAP / 2, MAP / 2);
	test::@check(
		dark_centre[0] == 0 && dark_centre[1] == 0 && dark_centre[2] == 0,
		"and the face is black rather than lit by the frame's whole list: %d, %d, %d",
		(int)(uint)dark_centre[0],
		(int)(uint)dark_centre[1],
		(int)(uint)dark_centre[2]
	);
}

// `cluster_grid.w == 0` is the frame block's "no cluster grid": the shader packs
// the overflow bit itself and walks the frame's whole list, so the same point
// light lights the face with no dispatch at all.
fn void a_frame_with_no_cluster_grid_shades_the_whole_light_list() @test
{
	if (!file::is_file(MESH_LIT_MODULE))
	{
		io::printfn("mesh_lit: no %s; skipped", MESH_LIT_MODULE);
		return;
	}

	Rig rig;
	rig_up(&rig)!!;
	defer rig.free();

	compute_support::MaterialSet materials = compute_support::material_set_up(&rig.f)!!;
	defer materials.free();

	PipelineLayout layout = vk::pipelineLayoutCreateInfo()
		.setSetLayouts({ materials.shadow, materials.textures })
		.setPushConstantRanges(
			{
				{
					.stageFlags = SHADER_STAGE_VERTEX_BIT | SHADER_STAGE_FRAGMENT_BIT,
					.offset = 0,
					.size = (uint)MeshPush::size,
				}
			}
		)
		.build(rig.f.device)!!;
	defer vk::destroyPipelineLayout(rig.f.device, layout, null);

	ShaderModule shader = load_module(&rig.f, MESH_LIT_MODULE)!!;
	defer vk::destroyShaderModule(rig.f.device, shader, null);
	Pipeline pipeline = mesh_pipeline_up(&rig.f, rig.colour.pass, layout, shader)!!;
	defer vk::destroyPipeline(rig.f.device, pipeline, null);

	Light[2] lights = {};
	lights[0].meta[1] = 0xffffffff;
	lights[1].direction = { 0.0, 0.0, 1.8, 1.1 };
	lights[1].color = { 0.0, 1.0, 0.5, 1.0 };
	lights[1].meta = { 0, 0xffffffff, LIGHT_POINT, 0 };
	rig.set_lights(lights[..]);

	// The grid's capacity is zero, which is the flag, and the cluster buffers
	// stay as the rig left them - zeroed - because nothing dispatches.
	rig.set_frame(light_count: 2.0, capacity: 0, near_plane: 1.2, far_plane: 4.0);

	char[] lit = rig.render(pipeline, layout, null, &materials);
	defer free(lit);
	write_ppm("build/mesh_lit_overflow.ppm", lit);

	// The same light and camera as the clustered point test, so the same bytes
	// are wanted: if the overflow path had been missed the face would be black.
	char* centre = pixel_at(lit, MAP / 2, MAP / 2);
	test::@check(
		(uint)centre[1] > 200 && (uint)centre[1] > (uint)centre[2] && (uint)centre[2] > (uint)centre[0],
		"the face is lit in the light's hue with no culling: %d, %d, %d",
		(int)(uint)centre[0],
		(int)(uint)centre[1],
		(int)(uint)centre[2]
	);
	test::@check(
		byte_near(centre[1], 0.85075f * 255.0f) && byte_near(centre[2], 0.42538f * 255.0f),
		"green and blue are the light's hue times its reach at the face: %d, %d, wanted %d, %d",
		(int)(uint)centre[1],
		(int)(uint)centre[2],
		(int)(0.85075f * 255.0f + 0.5f),
		(int)(0.42538f * 255.0f + 0.5f)
	);
}
