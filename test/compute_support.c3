<*
 A headless device for the compute ports, and the buffers they dispatch over.

 Kong's tests run against three.c3's generated modules, and the compute ones
 need a device with no window, no surface and no swapchain: one buffer per
 input, all of it reachable by address, one dispatch, and whatever the pass
 wrote read straight back out of host-visible memory. The first compute port
 wrote this, and the ones after it import it.

 Nothing here is kong's renderer - that is `src/main.c3` - so a failure in a test
 that uses it is about the generated module rather than about a frame.
*>
module compute_support;

import std::io;
import vk;

struct Fixture
{
	vk::Instance instance;
	PhysicalDevice pdevice;
	Device device;
	DeviceQueue queue;
	vk::MemoryAllocator allocator;
	CommandPool pool;
	CommandBuffer cmd;
	Fence fence;
}

fn void Fixture.free(&self)
{
	vk::destroyFence(self.device, self.fence, null);
	vk::destroyCommandPool(self.device, self.pool, null);
	self.allocator.free();
	self.device.free();
	self.instance.free();
}

faultdef NO_COMPUTE_QUEUE;

fn Fixture? device_up()
{
	// `DEFAULT_EXTENSIONS` rather than an empty list: the builder indexes
	// `extensions[0]`, which panics on zero of them.
	vk::Instance instance = vk::createDefaultInstance(
		{ .app_name = "skin_test", .extensions = { ...vk::DEFAULT_EXTENSIONS } }
	)!;

	PhysicalDevice pdevice = (PhysicalDevice)null;
	uint queue_index       = 0;
	foreach (candidate : instance.getDevices()!)
	{
		QueuePropertiesList families = candidate.getQueueProperties()!;
		for (sz i = 0; i < families.len(); i++)
		{
			if (((uint)families[i].queueFlags & (uint)vk::QueueFlagBits.QUEUE_GRAPHICS_BIT) == 0) continue;
			pdevice = candidate;
			queue_index = (uint)i;
			break;
		}
		if (pdevice != null) break;
	}
	if (pdevice == null) return NO_COMPUTE_QUEUE~;

	// Everything the device offers, the way the demo does it: the queried
	// structs are handed straight back, so `bufferDeviceAddress` is on.
	PhysicalDeviceVulkan13Features vulkan13 = { .sType = STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
	PhysicalDeviceVulkan12Features vulkan12 = {
		.sType = STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.pNext = &vulkan13
	};
	PhysicalDeviceVulkan11Features vulkan11 = {
		.sType = STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
		.pNext = &vulkan12
	};
	PhysicalDeviceFeatures2 features = { .sType = STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &vulkan11 };
	vk::getPhysicalDeviceFeatures2(pdevice, &features);

	Device device = vk::deviceCreateInfo()
		.setQueueCreateInfos({ vk::deviceQueueCreateInfo().setQueueFamilyIndex(queue_index)
			.setQueuePriorities({ 1.0 }) })
		.setNext(&features)
		.build(pdevice)!;

	DeviceQueue queue = { .index = queue_index, .handle = device.getQueue(queue_index) };

	vk::MemoryAllocator allocator = (vk::MemoryAllocator) {
		.device = device,
		.pdevice = pdevice,
		.queue = queue,
		.flags = MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT | MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT,
	}.init()!;

	CommandPool pool          = device.createCommandPool(queue: queue_index)!;
	CommandBufferList buffers = device.createCommandBuffers(pool, 1)!;
	Fence fence               = vk::fenceCreateInfo().build(device)!;

	return (Fixture) {
		.instance = instance,
		.pdevice = pdevice,
		.device = device,
		.queue = queue,
		.allocator = allocator,
		.pool = pool,
		.cmd = buffers[0],
		.fence = fence,
	};
}

// A buffer the shader reads through its address and the host can write and read
// directly: one page of host-visible memory per buffer, no copies either way.
fn Memory? upload(Fixture* f, void* data, sz size)
{
	return vk::new_buffer(
		allocator: &f.allocator,
		usage: BUFFER_USAGE_STORAGE_BUFFER_BIT | BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		properties: MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT,
		data: data,
		data_size: size
	);
}

fn bool close(float a, float b) @inline
{
	float d = a - b;
	if (d < 0.0) d = -d;
	return d < 0.0001;
}

<*
 The 1x1 opaque white an empty texture slot holds, and the descriptor set the
 generated draw path reads it through.

 A material's maps are slots of one device-wide array - set 1, binding 0 - and
 the generated module declares it whether or not a draw reads a map. So every
 rig that draws a mesh module needs the binding valid: a descriptor the layout
 declares and the draw leaves unwritten is undefined behaviour, and RADV turns
 it into a crash rather than a wrong texel.
*>
struct MaterialSet
{
	Image image;
	Memory memory;
	ImageView view;
	Sampler sampler;
	// The shadow atlases' own 1x1 depth image, with a compare sampler. The
	// generated module declares set 0 bindings 18 and 21 whether or not a frame
	// has a shadow pass, and a draw through a pipeline layout that disagrees
	// with the module faults in the driver rather than reading a wrong texel.
	Image depth_image;
	Memory depth_memory;
	ImageView depth_view;
	Sampler compare;
	DescriptorSetLayout shadow;   // set 0
	DescriptorSetLayout textures; // set 1, the device-wide texture array
	DescriptorPool pool;
	DescriptorSet shadow_set;
	DescriptorSet texture_set;
	Device device;
}

fn void MaterialSet.free(&self)
{
	vk::destroyDescriptorPool(self.device, self.pool, null);
	vk::destroyDescriptorSetLayout(self.device, self.textures, null);
	vk::destroyDescriptorSetLayout(self.device, self.shadow, null);
	vk::destroySampler(self.device, self.compare, null);
	vk::destroySampler(self.device, self.sampler, null);
	vk::destroyImageView(self.device, self.depth_view, null);
	vk::destroyImageView(self.device, self.view, null);
	self.depth_memory.free();
	self.memory.free();
}

fn MaterialSet? material_set_up(Fixture* f)
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
		.build(f.device)!!;
	Memory memory = vk::create_image_buffer(image, &f.allocator, MEMORY_PROPERTY_DEVICE_LOCAL_BIT)!!;

	Memory stage = upload(f, &white[0], 4)!!;
	defer stage.free();

	ImageMemoryBarrier2[1] to_transfer = { {
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
	} };
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

	// A 1x1 depth image, never written and never read: the descriptor has to
	// exist, and an unbound one is the crash this helper is here to avoid.
	Image depth_image = vk::imageCreateInfo()
		.setImageType(IMAGE_TYPE_2D)
		.setFormat(FORMAT_D32_SFLOAT)
		.setExtent({ .width = 1, .height = 1, .depth = 1 })
		.setMipLevels(1)
		.setArrayLayers(1)
		.setSamples(SAMPLE_COUNT_1_BIT)
		.setTiling(IMAGE_TILING_OPTIMAL)
		.setUsage(IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | IMAGE_USAGE_SAMPLED_BIT)
		.setInitialLayout(IMAGE_LAYOUT_UNDEFINED)
		.build(f.device)!!;
	Memory depth_memory = vk::create_image_buffer(depth_image, &f.allocator, MEMORY_PROPERTY_DEVICE_LOCAL_BIT)!!;

	ImageMemoryBarrier2[1] depth_to_shader = { {
		.sType            = STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.image            = depth_image,
		.srcStageMask     = PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
		.dstStageMask     = PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
		.srcAccessMask    = (AccessFlags2)0,
		.dstAccessMask    = ACCESS_2_SHADER_SAMPLED_READ_BIT,
		.oldLayout        = IMAGE_LAYOUT_UNDEFINED,
		.newLayout        = IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		.subresourceRange = {
			.aspectMask     = IMAGE_ASPECT_DEPTH_BIT,
			.baseMipLevel   = 0,
			.levelCount     = 1,
			.baseArrayLayer = 0,
			.layerCount     = 1,
		},
	} };

	vk::resetCommandBuffer(f.cmd, (vk::CommandBufferResetFlags)0)!!;
	vk::beginCommandBuffer(f.cmd, &&vk::commandBufferBeginInfo())!!;
	vk::cmdPipelineBarrier2(f.cmd, &&vk::dependencyInfo().setImageMemoryBarriers(to_transfer[..]));
	vk::cmdCopyBufferToImage(f.cmd, stage.buffer, image, IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
	vk::cmdPipelineBarrier2(f.cmd, &&vk::dependencyInfo().setImageMemoryBarriers(to_shader[..]));
	vk::cmdPipelineBarrier2(f.cmd, &&vk::dependencyInfo().setImageMemoryBarriers(depth_to_shader[..]));
	vk::endCommandBuffer(f.cmd)!!;

	CommandBuffer[1] command_buffers = { f.cmd };
	SubmitInfo submit = vk::submitInfo().setCommandBuffers(command_buffers[..]);
	vk::queueSubmit(f.queue.handle, 1, &submit, f.fence)!!;
	vk::waitForFences(f.device, 1, &f.fence, vk::TRUE, ~(ulong)0)!!;

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
		.build(f.device)!!;
	ImageView depth_view = vk::imageViewCreateInfo()
		.setImage(depth_image)
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
		.build(f.device)!!;
	Sampler sampler = vk::samplerCreateInfo().build(f.device)!!;
	Sampler compare = vk::samplerCreateInfo()
		.setCompareEnable(vk::TRUE)
		.setCompareOp(COMPARE_OP_LESS_OR_EQUAL)
		.build(f.device)!!;

	DescriptorSetLayout shadow = vk::descriptorSetLayoutCreateInfo()
		.setBindings(
			{
				{
					.binding         = 2,
					.descriptorType  = DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags      = SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding         = 4,
					.descriptorType  = DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags      = SHADER_STAGE_FRAGMENT_BIT,
				},
				// The debug view's un-filtered read of the same depth image the
				// compare samplers take at 18 and 21.
				{
					.binding         = 17,
					.descriptorType  = DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags      = SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding         = 18,
					.descriptorType  = DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags      = SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding         = 20,
					.descriptorType  = DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags      = SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding         = 21,
					.descriptorType  = DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags      = SHADER_STAGE_FRAGMENT_BIT,
				},
			}
		)
		.build(f.device)!!;
	DescriptorSetLayout textures = vk::descriptorSetLayoutCreateInfo()
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
		.build(f.device)!!;

	DescriptorPool pool = vk::descriptorPoolCreateInfo()
		.setMaxSets(2)
		.setPoolSizes({ { .type = DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 7 } })
		.build(f.device)!!;

	DescriptorSet[2] sets;
	DescriptorSetLayout[2] set_layouts = { shadow, textures };
	vk::allocateDescriptorSets(
		f.device,
		&&vk::descriptorSetAllocateInfo().setDescriptorPool(pool).setSetLayouts(set_layouts[..]),
		&sets[0]
	)!!;

	DescriptorImageInfo[1] color_info = {
		{
			.imageView   = view,
			.imageLayout = IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.sampler     = sampler,
		}
	};
	DescriptorImageInfo[1] depth_info = {
		{
			.imageView   = depth_view,
			.imageLayout = IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.sampler     = compare,
		}
	};
	// The same depth image without the comparison, for the debug view's
	// `SampleLevel` read.
	DescriptorImageInfo[1] depth_read_info = {
		{
			.imageView   = depth_view,
			.imageLayout = IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.sampler     = sampler,
		}
	};
	WriteDescriptorSet[7] writes = {
		vk::writeDescriptorSet()
			.setDstSet(sets[0])
			.setDstBinding(2)
			.setDescriptorType(DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
			.setImageInfo(color_info[..]),
		vk::writeDescriptorSet()
			.setDstSet(sets[0])
			.setDstBinding(4)
			.setDescriptorType(DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
			.setImageInfo(color_info[..]),
		vk::writeDescriptorSet()
			.setDstSet(sets[0])
			.setDstBinding(17)
			.setDescriptorType(DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
			.setImageInfo(depth_read_info[..]),
		vk::writeDescriptorSet()
			.setDstSet(sets[0])
			.setDstBinding(18)
			.setDescriptorType(DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
			.setImageInfo(depth_info[..]),
		vk::writeDescriptorSet()
			.setDstSet(sets[0])
			.setDstBinding(20)
			.setDescriptorType(DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
			.setImageInfo(color_info[..]),
		vk::writeDescriptorSet()
			.setDstSet(sets[0])
			.setDstBinding(21)
			.setDescriptorType(DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
			.setImageInfo(depth_info[..]),
		vk::writeDescriptorSet()
			.setDstSet(sets[1])
			.setDstBinding(0)
			.setDescriptorType(DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
			.setImageInfo(color_info[..]),
	};
	vk::updateDescriptorSets(f.device, 7, &writes[0], 0, null);

	return (MaterialSet) {
		.image        = image,
		.memory       = memory,
		.view         = view,
		.sampler      = sampler,
		.depth_image  = depth_image,
		.depth_memory = depth_memory,
		.depth_view   = depth_view,
		.compare      = compare,
		.shadow       = shadow,
		.textures     = textures,
		.pool         = pool,
		.shadow_set   = sets[0],
		.texture_set  = sets[1],
		.device       = f.device,
	};
}
