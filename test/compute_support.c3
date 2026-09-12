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
