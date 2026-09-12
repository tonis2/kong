<*
 three.c3's generated cluster module, dispatched.

 The binning pass writes a grid and an atomic, so the checks are about *which*
 clusters a light landed in rather than about an image. With the view-projection
 set to identity the clip-space cube is the world, and a near plane equal to the
 far plane collapses every z slice onto one distance: a light in the middle of
 one x/y cell is then in exactly the 24 slices of that cell, which is a
 prediction the CPU can write down without mirroring the overlap test.

 The second dispatch pushes one cell past the sixty-four lights it reserves room
 for, which is the only path that touches the atomic: the count keeps its top
 bit and the shared overflow counter takes the difference.
*>
module cluster_test;

import std::core::test;
import std::core::mem::allocators;
import std::io;
import std::io::file;
import std::math::matrix;
import std::math::vector;
import vk, shady, compute_support;

const String CLUSTER_MODULE = "../three.c3/build/shaderdef/cluster.shady";
const uint GRID_X = 16;
const uint GRID_Y = 9;
const uint GRID_Z = 24;
const uint CLUSTERS = GRID_X * GRID_Y * GRID_Z;
const uint CAPACITY = 64;
const uint WORKGROUP = 64;
const uint POINT = 1;

// `three::ClusterPush` (`src/gpu/pipeline.c3`), as the host fills it. The
// offsets are the engine's own, which `@layout_matches` pins on its side.
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

// `three::Light`, whose stride is its size.
struct Light
{
	Vec4 direction; // xyz position, w radius
	Vec4 color;
	Vec4 area_right;
	Vec4 area_up;
	uint[4] meta; // z is the kind: 0 directional, 1 point, 2/3 area
}

$assert Light::size == 80 : "Light no longer matches three.c3's";

faultdef MODULE_DID_NOT_COMPILE;

struct Pass
{
	ShaderModule shader;
	PipelineLayout layout;
	Pipeline pipeline;
}

// The generated module in a compute pipeline, with the push constant range the
// pass's block needs.
fn Pass? pass_up(Fixture* f)
{
	String source = (String)file::load(tmem, CLUSTER_MODULE)!;

	shady::Diagnostic diagnostic;
	char[]? spirv = shady::compile(source, &diagnostic, scalar_layout: true);
	if (catch spirv)
	{
		io::eprintfn("the generated cluster module does not compile: %s", diagnostic.to_string(tmem));
		diagnostic.free();
		return MODULE_DID_NOT_COMPILE~;
	}
	diagnostic.free();
	defer free(spirv);

	ShaderModule shader = vk::loadShaderModule(f.device, spirv)!;
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
	return (Pass) { .shader = shader, .layout = layout, .pipeline = pipeline };
}

// One dispatch over the whole grid. Nothing is copied back: the pass writes
// through device addresses into memory the host can read.
fn void bin(Fixture* f, Pipeline pipeline, PipelineLayout layout, ClusterPush* push)
{
	vk::resetCommandBuffer(f.cmd, (vk::CommandBufferResetFlags)0)!!;
	vk::beginCommandBuffer(f.cmd, &&vk::commandBufferBeginInfo())!!;
	vk::cmdBindPipeline(f.cmd, PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vk::cmdPushConstants(f.cmd, layout, SHADER_STAGE_COMPUTE_BIT, 0, (uint)ClusterPush::size, push);
	vk::cmdDispatch(f.cmd, (CLUSTERS + WORKGROUP - 1) / WORKGROUP, 1, 1);

	vk::MemoryBarrier visible = { .srcAccessMask = ACCESS_SHADER_WRITE_BIT,
		                          .dstAccessMask = ACCESS_HOST_READ_BIT };
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
}

<*
 One point light in the middle of one cell: in every z slice of it, in nothing
 else, and no overflow.
*>
fn void the_cluster_dispatch_bins_a_light_into_its_cell() @test
{
	if (!file::is_file(CLUSTER_MODULE))
	{
		io::printfn("cluster: no %s; skipped", CLUSTER_MODULE);
		return;
	}

	Fixture f = compute_support::device_up()!!;
	defer f.free();

	Pass pass = pass_up(&f)!!;

	Light[] lights = alloc::new_array(tmem, Light, 2);
	// Index 0 is skipped: the shader's loop starts at 1, the way the pass's own
	// array starts with a light no cluster ever references.
	lights[1].direction = { 0.0625, 0.0, 0.0, 0.05 }; // middle of cell 8, radius
	lights[1].meta      = { 0, 0, POINT, 0 };

	uint[] counts = alloc::new_array(tmem, uint, CLUSTERS);
	uint[] indices = alloc::new_array(tmem, uint, (usz)CLUSTERS * CAPACITY);
	uint[] overflow = alloc::new_array(tmem, uint, 1);
	foreach (&count : counts) *count = 0;
	foreach (&value : overflow) *value = 0;

	Memory lights_buffer = compute_support::upload(&f, &lights[0], (sz)lights.len * Light::size)!!;
	defer lights_buffer.free();
	Memory counts_buffer = compute_support::upload(&f, &counts[0], (sz)counts.len * uint::size)!!;
	defer counts_buffer.free();
	Memory indices_buffer = compute_support::upload(&f, &indices[0], (sz)indices.len * uint::size)!!;
	defer indices_buffer.free();
	Memory overflow_buffer = compute_support::upload(&f, &overflow[0], uint::size)!!;
	defer overflow_buffer.free();

	ClusterPush push = {
		.lights          = lights_buffer.get_address(),
		.counts          = counts_buffer.get_address(),
		.indices         = indices_buffer.get_address(),
		.overflow        = overflow_buffer.get_address(),
		.view_projection = { .m = { 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0 } },
		.light_count     = 2,
		.camera_near     = { 0.0, 0.0, 0.0, 0.05 },
		.far_plane       = 0.05,
	};

	bin(&f, pass.pipeline, pass.layout, &push);

	uint* read_counts = (uint*)counts_buffer.data();
	uint* read_indices = (uint*)indices_buffer.data();
	uint* read_overflow = (uint*)overflow_buffer.data();

	uint in_cell = 0;
	uint wrong   = 0;
	for (uint cluster = 0; cluster < CLUSTERS; cluster++)
	{
		uint x = cluster % GRID_X;
		uint y = (cluster / GRID_X) % GRID_Y;
		if (x == 8 && y == 4)
		{
			if (read_counts[cluster] == 1 && read_indices[cluster * CAPACITY] == 1)
			{
				in_cell++;
			}
			else
			{
				wrong++;
			}
		}
		else if (read_counts[cluster] != 0)
		{
			wrong++;
		}
	}

	test::@check(wrong == 0, "%d clusters disagree with the CPU's bounds", wrong);
	test::@check(in_cell == GRID_Z, "the light is in all %d slices of its cell, got %d", GRID_Z, in_cell);
	test::@check(read_overflow[0] == 0, "nothing overflowed, got %d", read_overflow[0]);
}

<*
 Seventy lights in the same place: the cell fills up, and the atomic counts the
 six that no longer fit. The count's top bit is how the renderer tells a full
 list from a truncated one.
*>
fn void a_full_cell_sets_its_overflow_bit() @test
{
	if (!file::is_file(CLUSTER_MODULE))
	{
		io::printfn("cluster: no %s; skipped", CLUSTER_MODULE);
		return;
	}

	Fixture f = compute_support::device_up()!!;
	defer f.free();

	Pass pass = pass_up(&f)!!;

	const uint LIGHTS_COUNT = 1 + CAPACITY + 6;
	Light[] lights = alloc::new_array(tmem, Light, LIGHTS_COUNT);
	for (uint i = 1; i < LIGHTS_COUNT; i++)
	{
		lights[i].direction = { 0.0625, 0.0, 0.0, 0.05 };
		lights[i].meta      = { 0, 0, POINT, 0 };
	}

	uint[] counts = alloc::new_array(tmem, uint, CLUSTERS);
	uint[] indices = alloc::new_array(tmem, uint, (usz)CLUSTERS * CAPACITY);
	uint[] overflow = alloc::new_array(tmem, uint, 1);
	foreach (&count : counts) *count = 0;
	foreach (&value : overflow) *value = 0;

	Memory lights_buffer = compute_support::upload(&f, &lights[0], (sz)lights.len * Light::size)!!;
	defer lights_buffer.free();
	Memory counts_buffer = compute_support::upload(&f, &counts[0], (sz)counts.len * uint::size)!!;
	defer counts_buffer.free();
	Memory indices_buffer = compute_support::upload(&f, &indices[0], (sz)indices.len * uint::size)!!;
	defer indices_buffer.free();
	Memory overflow_buffer = compute_support::upload(&f, &overflow[0], uint::size)!!;
	defer overflow_buffer.free();

	ClusterPush push = {
		.lights          = lights_buffer.get_address(),
		.counts          = counts_buffer.get_address(),
		.indices         = indices_buffer.get_address(),
		.overflow        = overflow_buffer.get_address(),
		.view_projection = { .m = { 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0 } },
		.light_count     = LIGHTS_COUNT,
		.camera_near     = { 0.0, 0.0, 0.0, 0.05 },
		.far_plane       = 0.05,
	};

	bin(&f, pass.pipeline, pass.layout, &push);

	uint* read_counts = (uint*)counts_buffer.data();
	uint* read_indices = (uint*)indices_buffer.data();
	uint* read_overflow = (uint*)overflow_buffer.data();

	const uint FULL = 0x80000040; // sixty-four written, the top bit set
	uint full = 0;
	uint wrong = 0;
	for (uint cluster = 0; cluster < CLUSTERS; cluster++)
	{
		uint x = cluster % GRID_X;
		uint y = (cluster / GRID_X) % GRID_Y;
		if (x == 8 && y == 4)
		{
			if (read_counts[cluster] == FULL)
			{
				full++;
			}
			else
			{
				wrong++;
			}
		}
		else if (read_counts[cluster] != 0)
		{
			wrong++;
		}
	}

	test::@check(wrong == 0, "%d clusters disagree with the CPU's bounds", wrong);
	test::@check(full == GRID_Z, "every slice of the cell is full and flagged, got %d", full);
	// The cell's first slice: cluster index x + y * GRID_X at z = 0.
	uint first_slice = (8 + 4 * GRID_X) * CAPACITY;
	test::@check(read_indices[first_slice] == 1 && read_indices[first_slice + CAPACITY - 1] == CAPACITY,
		"the first sixty-four lights are the ones kept");
	test::@check(read_overflow[0] == GRID_Z * 6, "the atomic counted the dropped lights: %d, wanted %d",
		read_overflow[0], GRID_Z * 6);
}
