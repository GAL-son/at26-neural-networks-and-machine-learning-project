#pragma once

#include <vulkan/vulkan.hpp>
#include <memory>
#include <vector>

#include "Renderer/VulkanContext.hpp"

namespace Engine::Renderer {
    class RenderPass {
        public:
            explicit RenderPass(VulkanContext& context)
                :m_context(context), m_device(context.getDevice()){}

            virtual ~RenderPass() = default;

            virtual void Create() = 0;
            virtual void Cleanup() = 0;

            cirt

        private:
            VulkanContext& m_context;
            VkDevice m_device{VK_NULL_HANDLE};
    }
}