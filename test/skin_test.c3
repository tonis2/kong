<*
 three.c3's generated skin module, dispatched on a device.

 The first compute port in the bridge, so this file is also the harness the rest
 of them get: buffers reached by device address, one dispatch, the posed
 vertices read back and compared against the arithmetic the port claims to do. A
 screenshot says something was drawn; this says where vertex 0 of instance 1
 ended up.

 The module is compiled the way the engine compiles it - `scalar_layout: true`,
 C3's packing - because every struct on both sides is a C3 wire struct, and
 std430 would stride `SkinnedVertex` by 32 rather than 24.

 What is dispatched: the baked palette, the live one, and a crossfade between
 two slots, each checked by where a vertex ends up. Not dispatched: the morph
 path, which needs a target and a weight and lands with the mesh port that reads
 the same structs.
*>
module skin_test;

import std::core::test;
import std::io;
import std::io::file;
import std::math::matrix;
import std::math::vector;
import vk, shady, compute_support;

const String SKIN_MODULE    = "../three.c3/build/shaderdef/skin.shady";
const uint VERTICES         = 4;
const uint INSTANCES        = 3;
const uint WORKGROUP        = 64;
const uint FLAG_HAS_NORMALS = 1;
const uint POSE_LIVE        = 0x80000000;

// `three::SkinPush` (`src/gpu/pipeline.c3`), as the host fills it: device
// addresses rather than the pointers the shader sees. The size is the engine's
// own number, so a drift between the two is a failed build.
struct SkinPush
{
	DeviceAddress positions;
	DeviceAddress normals;
	DeviceAddress joints;
	DeviceAddress weights;
	DeviceAddress instances;
	DeviceAddress baked;
	DeviceAddress live;
	DeviceAddress output;
	DeviceAddress morph;
	DeviceAddress morph_weights;
	uint vertex_count;
	uint instance_count;
	uint flags;
	uint morph_targets;
}

$assert SkinPush::size == 96 : "SkinPush no longer matches three.c3's";

// `three::Instance`, whose stride is its size. Only `pose`, `pose_from` and
// `blend` are read by this pass; the rest is here to keep the offsets honest.
struct Instance
{
	Matrix4f model;
	Matrix4f normal;
	float[4] color;
	uint variant;
	uint pose;
	uint pose_from;
	float blend;
	uint morph;
	float[4] lightmap;
}

$assert Instance::size == 180 : "Instance no longer matches three.c3's";

// `three::SkinnedVertex`, which is what the dispatch writes.
struct SkinnedVertex
{
	Vec3 position;
	Vec3 normal;
}

$assert SkinnedVertex::size == 24 : "SkinnedVertex no longer matches three.c3's";

fn void the_skin_dispatch_poses_the_vertices() @test
{
	if (!file::is_file(SKIN_MODULE))
	{
		io::printfn("skin: no %s; skipped", SKIN_MODULE);
		return;
	}

	Fixture f = compute_support::device_up()!!;
	defer f.free();

	// --- The module ---
	String source = (String)file::load(tmem, SKIN_MODULE)!!;
	shady::Diagnostic diagnostic;
	char[]? spirv = shady::compile(source, &diagnostic, scalar_layout: true);
	if (catch spirv)
	{
		test::@check(false, "the generated skin module compiles: %s", diagnostic.to_string(tmem));
		diagnostic.free();
		return;
	}
	diagnostic.free();
	defer free(spirv);

	ShaderModule shader = vk::loadShaderModule(f.device, spirv)!!;

	PipelineLayout layout = vk::pipelineLayoutCreateInfo()
		.setPushConstantRanges(
			{ { .stageFlags = SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = (uint)SkinPush::size } }
		)
		.build(f.device)!!;

	ComputePipelineCreateInfo pipeline_info = vk::computePipelineCreateInfo()
		.setStage(vk::pipelineShaderStageCreateInfo().setStage(SHADER_STAGE_COMPUTE_BIT).setMod(shader)
			.setName("computeMain"))
		.setLayout(layout);
	Pipeline pipeline;
	vk::createComputePipelines(f.device, (PipelineCache)null, 1, &pipeline_info, null, &pipeline)!!;

	// --- The bucket ---
	// Four vertices, one bone, full weight: the pose each instance writes is
	// then exactly the palette matrix its `pose` field selects.
	Vec3[VERTICES] positions = { { 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } };
	Vec3[VERTICES] normals = { { 0.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0 } };
	uint[4][VERTICES] joints;
	Vec4[VERTICES] weights = {
		{ 1.0, 0.0, 0.0, 0.0 },
		{ 1.0, 0.0, 0.0, 0.0 },
		{ 1.0, 0.0, 0.0, 0.0 },
		{ 1.0, 0.0, 0.0, 0.0 }
	};

	Instance[INSTANCES] instances;
	// Baked, its own slot. Crossfading out of slot 1, which is the `pose_from`
	// the blend reads. And live, which is the one bit that chooses a table.
	instances[0].pose = 0;
	instances[1].pose = 0;
	instances[1].pose_from = 1;
	instances[1].blend = 0.5;
	instances[2].pose = POSE_LIVE;

	// The matrices are written as columns, the way the shader reads them: a
	// translation lives in the fourth column, and index 0 of a column is x.
	Matrix4f[2] baked = {
		{ .m = { 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 0.0, 1.0 } },
		{ .m = { 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 20.0, 0.0, 0.0, 1.0 } },
	};
	// A quarter turn about y and 30 in x, so a normal tells a w = 0 multiply
	// from a w = 1 one: the first sends the normal to +x, the second to +31x.
	Matrix4f[1] live = {
		{ .m = { 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 1.0 } },
	};

	Memory positions_buffer = compute_support::upload(&f, &positions[0], (sz)positions.len * Vec3::size)!!;
	defer positions_buffer.free();
	Memory normals_buffer = compute_support::upload(&f, &normals[0], (sz)normals.len * Vec3::size)!!;
	defer normals_buffer.free();
	Memory joints_buffer = compute_support::upload(&f, &joints[0], (sz)joints.len * uint[4]::size)!!;
	defer joints_buffer.free();
	Memory weights_buffer = compute_support::upload(&f, &weights[0], (sz)weights.len * Vec4::size)!!;
	defer weights_buffer.free();
	Memory instances_buffer = compute_support::upload(&f, &instances[0], (sz)instances.len * Instance::size)!!;
	defer instances_buffer.free();
	Memory baked_buffer = compute_support::upload(&f, &baked[0], (sz)baked.len * Matrix4f::size)!!;
	defer baked_buffer.free();
	Memory live_buffer = compute_support::upload(&f, &live[0], (sz)live.len * Matrix4f::size)!!;
	defer live_buffer.free();

	// The output is a full bucket's worth: three instances of four vertices, and
	// the dispatch is one workgroup of sixty-four threads, so the tail guard is
	// part of what this checks.
	usz output_size = (usz)INSTANCES * VERTICES * SkinnedVertex::size;
	Memory output_buffer = vk::new_buffer(
		allocator: &f.allocator,
		usage: BUFFER_USAGE_STORAGE_BUFFER_BIT | BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		properties: MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT,
		data: null,
		data_size: (sz)output_size
	)!!;
	defer output_buffer.free();

	SkinPush push = {
		.positions = positions_buffer.get_address(),
		.normals = normals_buffer.get_address(),
		.joints = joints_buffer.get_address(),
		.weights = weights_buffer.get_address(),
		.instances = instances_buffer.get_address(),
		.baked = baked_buffer.get_address(),
		.live = live_buffer.get_address(),
		.output = output_buffer.get_address(),
		.vertex_count = VERTICES,
		.instance_count = INSTANCES,
		.flags = FLAG_HAS_NORMALS,
	};

	// --- The dispatch ---
	uint threads = VERTICES * INSTANCES;
	vk::resetCommandBuffer(f.cmd, (vk::CommandBufferResetFlags)0)!!;
	vk::beginCommandBuffer(f.cmd, &&vk::commandBufferBeginInfo())!!;
	vk::cmdBindPipeline(f.cmd, PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vk::cmdPushConstants(f.cmd, layout, SHADER_STAGE_COMPUTE_BIT, 0, (uint)SkinPush::size, &push);
	vk::cmdDispatch(f.cmd, (threads + WORKGROUP - 1) / WORKGROUP, 1, 1);

	// The shader wrote through a device address; the host reads the same bytes.
	vk::MemoryBarrier visible = { .srcAccessMask = ACCESS_SHADER_WRITE_BIT, .dstAccessMask = ACCESS_HOST_READ_BIT };
	vk::cmdPipelineBarrier(
		f.cmd,
		PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		PIPELINE_STAGE_HOST_BIT,
		(DependencyFlags)0,
		1,
		&visible,
		0,
		null,
		0,
		null
	);
	vk::endCommandBuffer(f.cmd)!!;

	SubmitInfo submit = vk::submitInfo().setCommandBuffers({ f.cmd });
	vk::queueSubmit(f.queue.handle, 1, &submit, f.fence)!!;
	vk::waitForFences(f.device, 1, &f.fence, vk::TRUE, ~(ulong)0)!!;

	// --- What the dispatch wrote ---
	SkinnedVertex* out = (SkinnedVertex*)output_buffer.data();

	float[INSTANCES] want_x        = { 10.0, 15.0, 30.0 };
	float[INSTANCES] want_normal_x = { 0.0, 0.0, 1.0 };

	for (uint i = 0; i < INSTANCES; i++)
	{
		// Vertex 0 is the origin, so its pose is the matrix's translation and
		// nothing else.
		SkinnedVertex origin = out[i * VERTICES];
		test::@check(
			compute_support::close(origin.position.x, want_x[i]),
			"instance %d puts the origin at x = %f, wanted %f",
			i,
			origin.position.x,
			want_x[i]
		);
		test::@check(
			compute_support::close(origin.position.y, 0.0) && compute_support::close(origin.position.z, 0.0),
			"and leaves it on the y = z = 0 line, got (%f, %f, %f)",
			origin.position.x,
			origin.position.y,
			origin.position.z
		);

		// Vertex 1 is one unit along x, which pins the rotation as well as the
		// translation: it turns (1,0,0) into (0,0,-1) for the live instance.
		SkinnedVertex one = out[i * VERTICES + 1];
		float want_one_x  = i == 2 ? want_x[i] : want_x[i] + 1.0;
		float want_one_z  = i == 2 ? -1.0 : 0.0;
		test::@check(
			compute_support::close(one.position.x, want_one_x) && compute_support::close(one.position.z, want_one_z),
			"instance %d poses vertex 1 to (%f, %f, %f), wanted (%f, 0, %f)",
			i,
			one.position.x,
			one.position.y,
			one.position.z,
			want_one_x,
			want_one_z
		);

		test::@check(
			compute_support::close(origin.normal.x, want_normal_x[i]) && compute_support::close(origin.normal.y, 0.0),
			"instance %d leaves the normal (%f, %f, %f), wanted (%f, 0, 0)",
			i,
			origin.normal.x,
			origin.normal.y,
			origin.normal.z,
			want_normal_x[i]
		);
		test::@check(
			compute_support::close(origin.normal.z, i == 2 ? 0.0 : 1.0),
			"because the normal is the w = 0 multiply, not the w = 1 one"
		);
	}
}
