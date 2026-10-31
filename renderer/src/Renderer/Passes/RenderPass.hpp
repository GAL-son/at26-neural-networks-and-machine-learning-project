#pragma once

#include <vulkan/vulkan.hpp>
#include <memory>
#include <vector>

#include "Renderer/VulkanContext.hpp"
#include "Renderer/Resources/Scene.hpp"
#include "Core/Camera.hpp"

namespace Engine::Renderer
{
    class RenderPass
    {
    public:
        explicit RenderPass(VulkanContext &context)
            : m_context(context), m_device(context.getDevice()) {}

        virtual ~RenderPass() = default;

        virtual void Create() = 0;
        virtual void Cleanup() = 0;

        virtual void Begin(VkCommandBuffer commandBuffer, VkExtent2D extent) = 0;
        
        // Zależnie od potrzeb: = 0 jeśli każdy pass renderuje scenę, 
        // lub puste ciało {} jeśli tylko wybrane passy tego używają.
        virtual void Render(VkCommandBuffer commandBuffer, Renderer::Scene& scene, Camera& camera) {} 
        
        virtual void End(VkCommandBuffer commandBuffer) = 0;

        VkRenderPass GetVkRenderPass() const { return m_renderPass; }
        VkFramebuffer GetFramebuffer(size_t index = 0) const
        {
            return m_framebuffers.empty() ? VK_NULL_HANDLE : m_framebuffers[index];
        }

    protected: // <-- Zmienione z private, żeby klasy pochodne miały dostęp!
        VulkanContext &m_context;
        VkDevice m_device{VK_NULL_HANDLE};
        
        VkRenderPass m_renderPass{VK_NULL_HANDLE};
        std::vector<VkFramebuffer> m_framebuffers;
        std::vector<VkImage> m_targetImages;
    };
}