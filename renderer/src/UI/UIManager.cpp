#include "UIManager.hpp"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <stdexcept>

UIManager::UIManager() = default;
UIManager::~UIManager() = default;

void UIManager::init(const Window& window, const VulkanContext& vulkanContext, VkDescriptorPool& outDescriptorPool) {
    // 1. Create Descriptor Pool for ImGui
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
    pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;
    
    if (vkCreateDescriptorPool(vulkanContext.getDevice(), &pool_info, nullptr, &outDescriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create ImGui descriptor pool!");
    }

   // 2. Initialize ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();

    // 3. Initialize GLFW binding
    ImGui_ImplGlfw_InitForVulkan(window.getGLFWWindow(), true);

    // 4. Initialize Vulkan binding with Dynamic Rendering
    m_swapchainFormat = vulkanContext.getSwapchainImageFormat(); // <-- Store in member variable!
    
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = vulkanContext.getInstance();
    init_info.PhysicalDevice = vulkanContext.getPhysicalDevice();
    init_info.Device = vulkanContext.getDevice();
    init_info.QueueFamily = vulkanContext.getGraphicsQueueFamily();
    init_info.Queue = vulkanContext.getGraphicsQueue();
    init_info.DescriptorPool = outDescriptorPool;
    init_info.MinImageCount = 2;
    init_info.ImageCount = 2;
    
    // Enable dynamic rendering and point to the persistent member variable
    init_info.UseDynamicRendering = true;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo = {};
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &m_swapchainFormat; // <-- Safe persistent pointer!

    ImGui_ImplVulkan_Init(&init_info);
    m_initialized = true;
}

void UIManager::shutdown(const VulkanContext& vulkanContext, VkDescriptorPool descriptorPool) {
    if (!m_initialized) return;
    vkDeviceWaitIdle(vulkanContext.getDevice());
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    if (descriptorPool) {
        vkDestroyDescriptorPool(vulkanContext.getDevice(), descriptorPool, nullptr);
    }
}

void UIManager::beginFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void UIManager::renderStatsAndGrid(uint32_t processedFrames, float frameTimeMs, float passTimeMs) {
    // Stats Window
    ImGui::Begin("Renderer Control & Stats");
    ImGui::Text("Mode: Windowed Visualization");
    ImGui::Text("Processed Frames: %u", processedFrames);
    ImGui::Text("Frame Time: %.2f ms", frameTimeMs);
    ImGui::Text("Pass Time: %.2f ms", passTimeMs);
    ImGui::End();

    // 2x2 Buffer View Template Window
    ImGui::Begin("Buffers (2x2 Grid View)");
    ImGui::Columns(2, "BufferColumns", false);
    ImGui::Text("1. G-Buffer (Aliased)");
    ImGui::Button("Color A", ImVec2(200, 120));
    ImGui::NextColumn();
    
    ImGui::Text("2. Depth & Normals");
    ImGui::Button("Color B", ImVec2(200, 120));
    ImGui::NextColumn();
    
    ImGui::Text("3. Motion Vectors");
    ImGui::Button("Color C", ImVec2(200, 120));
    ImGui::NextColumn();
    
    ImGui::Text("4. SSAAx32 Ground Truth");
    ImGui::Button("Color D", ImVec2(200, 120));
    ImGui::Columns(1);
    ImGui::End();
}

void UIManager::renderDrawData(VkCommandBuffer commandBuffer) {
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
}