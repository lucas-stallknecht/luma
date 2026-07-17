package luma

import vk "vendor:vulkan"

// ────────────────────────────────────────────────────────────────
// Barriers

Image_Barrier :: struct {
	image:      ^Image,
	src_stage:  vk.PipelineStageFlags2,
	src_access: vk.AccessFlags2,
	dst_stage:  vk.PipelineStageFlags2,
	dst_access: vk.AccessFlags2,
}

image_barriers :: proc(cb: vk.CommandBuffer, barriers: ..Image_Barrier) {
	// for image layouts, everything stays in GENERAL
	// swapchain is the only thing that ever leaves GENERAL (needs PRESENT_SRC_KHR)

	vk_barriers := make([]vk.ImageMemoryBarrier2, len(barriers), context.temp_allocator)
	for b, i in barriers {
		vk_barriers[i] = vk.ImageMemoryBarrier2 {
			sType = .IMAGE_MEMORY_BARRIER_2,
			srcStageMask = b.src_stage,
			srcAccessMask = b.src_access,
			dstStageMask = b.dst_stage,
			dstAccessMask = b.dst_access,
			oldLayout = b.image.layout,
			newLayout = .GENERAL,
			image = b.image.image,
			subresourceRange = {
				aspectMask = image_aspect_mask(b.image.format),
				layerCount = max(b.image.array_layers, 1),
				levelCount = b.image.mip_levels,
			},
		}
		b.image.layout = .GENERAL
	}
	dep := vk.DependencyInfo {
		sType                   = .DEPENDENCY_INFO,
		imageMemoryBarrierCount = u32(len(vk_barriers)),
		pImageMemoryBarriers    = raw_data(vk_barriers),
	}
	vk.CmdPipelineBarrier2(cb, &dep)
}

Buffer_Barrier :: struct {
	buffer:     ^Buffer,
	src_stage:  vk.PipelineStageFlags2,
	src_access: vk.AccessFlags2,
	dst_stage:  vk.PipelineStageFlags2,
	dst_access: vk.AccessFlags2,
}

buffer_barriers :: proc(cb: vk.CommandBuffer, barriers: ..Buffer_Barrier) {
	vk_barriers := make([]vk.BufferMemoryBarrier2, len(barriers), context.temp_allocator)
	for b, i in barriers {
		vk_barriers[i] = vk.BufferMemoryBarrier2 {
			sType               = .BUFFER_MEMORY_BARRIER_2,
			srcStageMask        = b.src_stage,
			srcAccessMask       = b.src_access,
			dstStageMask        = b.dst_stage,
			dstAccessMask       = b.dst_access,
			srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
			dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
			buffer              = b.buffer.buffer,
			offset              = 0,
			size                = vk.DeviceSize(vk.WHOLE_SIZE),
		}
	}
	dep := vk.DependencyInfo {
		sType                    = .DEPENDENCY_INFO,
		bufferMemoryBarrierCount = u32(len(vk_barriers)),
		pBufferMemoryBarriers    = raw_data(vk_barriers),
	}
	vk.CmdPipelineBarrier2(cb, &dep)
}

// ────────────────────────────────────────────────────────────────
// Render Graph

Render_Resource_Target :: union {
	^Buffer,
	^Image,
}

Render_Resource_Usage :: struct {
	stage:  vk.PipelineStageFlags2,
	access: vk.AccessFlags2,
}

Render_Resource :: struct {
	target: Render_Resource_Target,
	usage:  Render_Resource_Usage,
}

Render_Pass :: struct {
	name:      string,
	resources: []Render_Resource,
	user_data: rawptr,
	callback:  proc(resources: []Render_Resource, user_data: rawptr, cb: vk.CommandBuffer),
}

// read accumulates every read seen since the last write
// a reader in a new stage still gets a barrier, instead of being treated as a no-op read-after-read
Render_Resource_Sync_State :: struct {
	write: Render_Resource_Usage,
	read:  Render_Resource_Usage,
}

RenderGraph :: struct {
	passes:          [dynamic]Render_Pass,
	image_barriers:  [dynamic]Image_Barrier,
	buffer_barriers: [dynamic]Buffer_Barrier,
	sync_states:     map[rawptr]Render_Resource_Sync_State,
}

render_graph_init :: proc(rg: ^RenderGraph) {
	rg.sync_states = make(map[rawptr]Render_Resource_Sync_State)
}

render_graph_cleanup :: proc(rg: ^RenderGraph) {
	delete(rg.passes)
	delete(rg.image_barriers)
	delete(rg.buffer_barriers)
	delete(rg.sync_states)
}

render_graph_begin :: proc(rg: ^RenderGraph) {
	clear(&rg.passes)
}

render_graph_add_pass :: proc(
	rg: ^RenderGraph,
	name: string,
	callback: proc(resources: []Render_Resource, user_data: rawptr, cb: vk.CommandBuffer),
	user_data: rawptr,
	resources: ..Render_Resource,
) {
	owned := make([]Render_Resource, len(resources), context.temp_allocator)
	copy(owned, resources)
	append(
		&rg.passes,
		Render_Pass{name = name, resources = owned, user_data = user_data, callback = callback},
	)
}

render_graph_execute :: proc(rg: ^RenderGraph, cb: vk.CommandBuffer) {
	WRITE_ACCESS :: vk.AccessFlags2 {
		.SHADER_WRITE,
		.SHADER_STORAGE_WRITE,
		.COLOR_ATTACHMENT_WRITE,
		.DEPTH_STENCIL_ATTACHMENT_WRITE,
		.TRANSFER_WRITE,
		.HOST_WRITE,
		.MEMORY_WRITE,
		.ACCELERATION_STRUCTURE_WRITE_KHR,
	}
	is_write :: proc(access: vk.AccessFlags2) -> bool {
		return access & WRITE_ACCESS != {}
	}
	resource_ptr :: proc(target: Render_Resource_Target) -> rawptr {
		switch t in target {
		case ^Image:
			return t
		case ^Buffer:
			return t
		}
		return nil
	}

	for pass in rg.passes {
		clear(&rg.image_barriers)
		clear(&rg.buffer_barriers)

		for res in pass.resources {
			// for each resource, look up its last recorded state (state/has_prev) and decide
			// whether this pass' usage needs a barrier before it can run:
			//   - no recorded state at all -> first time the graph has seen this resource,
			//     assume it's already synced externally (eg create_image's initial transition)
			//   - this usage writes -> always needs a barrier, waiting on both the last write
			//     and every read since (state.write | state.read), since a write
			//     can't safely overlap with anything that came before it
			//   - this usage reads -> only needs a barrier if its stage/access isn't already
			//     covered by an earlier read since the last write
			//     same-stage/access reads never race so they're skipped, but a read in a new
			//     stage still needs one, since the barrier that unlocked the earlier read never
			//     granted that stage visibility
			// state is then mutated into the resource's new recorded state and written back to
			// sync_states at the end of the loop, ready for the next pass (or next frame) to read
			ptr := resource_ptr(res.target)
			state, has_prev := rg.sync_states[ptr]

			needs_barrier: bool
			src_stage := state.write.stage
			src_access := state.write.access

			if is_write(res.usage.access) {
				needs_barrier = has_prev
				src_stage = state.write.stage | state.read.stage
				src_access = state.write.access | state.read.access
				state = {
					write = res.usage,
				}
			} else {
				covered :=
					(res.usage.stage - state.read.stage) == {} &&
					(res.usage.access - state.read.access) == {}
				needs_barrier = has_prev && !covered
				state.read.stage |= res.usage.stage
				state.read.access |= res.usage.access
			}

			if needs_barrier {
				switch t in res.target {
				case ^Image:
					append(
						&rg.image_barriers,
						Image_Barrier {
							image = t,
							src_stage = src_stage,
							src_access = src_access,
							dst_stage = res.usage.stage,
							dst_access = res.usage.access,
						},
					)
				case ^Buffer:
					append(
						&rg.buffer_barriers,
						Buffer_Barrier {
							buffer = t,
							src_stage = src_stage,
							src_access = src_access,
							dst_stage = res.usage.stage,
							dst_access = res.usage.access,
						},
					)
				}
			}

			rg.sync_states[ptr] = state
		}

		// all of this pass' barriers go out together as one CmdPipelineBarrier2 call each, not one per resource
		if len(rg.image_barriers) > 0 {
			image_barriers(cb, ..rg.image_barriers[:])
		}
		if len(rg.buffer_barriers) > 0 {
			buffer_barriers(cb, ..rg.buffer_barriers[:])
		}

		pass.callback(pass.resources, pass.user_data, cb)
	}
}
