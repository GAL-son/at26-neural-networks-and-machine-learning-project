#include "UIManager.hpp"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <stdexcept>

#include <tinyfiledialogs/tinyfiledialogs.h>

namespace Engine::UI
{
    UIManager::UIManager(const Window &window, const VulkanContext &vulkanContext)
        : m_context(vulkanContext)
    {
        // 1. Create Descriptor Pool for ImGui
        VkDescriptorPoolSize poolSizes[] = {
            {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
            {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};
        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.maxSets = 1000 * IM_ARRAYSIZE(poolSizes);
        poolInfo.poolSizeCount = (uint32_t)IM_ARRAYSIZE(poolSizes);
        poolInfo.pPoolSizes = poolSizes;

        if (vkCreateDescriptorPool(vulkanContext.getDevice(), &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create ImGui descriptor pool!");

        // 2. Initialize ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO &io = ImGui::GetIO();
        (void)io;
        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForVulkan(window.getGLFWWindow(), true);

        // 3. Initialize GLFW binding
        m_swapchainFormat = vulkanContext.getSwapchainImageFormat(); // <-- Store in member variable!

        ImGui_ImplVulkan_InitInfo initInfo = {};
        initInfo.Instance = vulkanContext.getInstance();
        initInfo.PhysicalDevice = vulkanContext.getPhysicalDevice();
        initInfo.Device = vulkanContext.getDevice();
        initInfo.QueueFamily = vulkanContext.getGraphicsQueueFamily();
        initInfo.Queue = vulkanContext.getGraphicsQueue();
        initInfo.DescriptorPool = m_descriptorPool;
        initInfo.MinImageCount = 2;
        initInfo.ImageCount = 2;

        // Enable dynamic rendering and point to the persistent member variable
        initInfo.UseDynamicRendering = true;
        initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = {};
        initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
        initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &m_swapchainFormat; // <-- Safe persistent pointer!

        ImGui_ImplVulkan_Init(&initInfo);
    }

    UIManager::~UIManager()
    {
        vkDeviceWaitIdle(m_context.getDevice());
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        vkDestroyDescriptorPool(m_context.getDevice(), m_descriptorPool, nullptr);
    };

    void UIManager::beginFrame()
    {
        {
            std::lock_guard<std::mutex> lock(m_context.getQueueMutex());
            ImGui_ImplVulkan_NewFrame();
        }

        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    ImDrawData *UIManager::endFrame()
    {
        ImGui::Render();
        return ImGui::GetDrawData();
    }

    void UIManager::renderStats(uint32_t processedFrames, float frameTimeMs, float passTimeMs)
    {
        ImGui::Begin("Renderer Control and Stats");
        ImGui::Text("Processed Frames %u", processedFrames);
        ImGui::Text("Processed Frame time %.2f ms", frameTimeMs);
        ImGui::Text("Pass Time %.2f ms", passTimeMs);
        ImGui::End();
    }

    void UIManager::renderBufferLabels()
    {
        struct BufferLabel
        {
            const char *text;
            ImVec2 quadrantOrigin;
        };

        const BufferLabel labels[] = {
            {"Color (albedo)", ImVec2(0.0f, 0.0f)},
            {"Normals", ImVec2(0.5f, 0.0f)},
            {"Motion vectors", ImVec2(0.0f, 0.5f)},
            {"Linear depth", ImVec2(0.5f, 0.5f)}};

        const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
        const ImVec2 margin(8.0f, 8.0f);
        const ImVec2 padding(6.0f, 3.0f);

        ImDrawList *drawList = ImGui::GetBackgroundDrawList();

        for (const BufferLabel &label : labels)
        {
            const ImVec2 textPos(
                displaySize.x * label.quadrantOrigin.x + margin.x + padding.x,
                displaySize.y * label.quadrantOrigin.y + margin.y + padding.y);
            const ImVec2 textSize = ImGui::CalcTextSize(label.text);

            drawList->AddRectFilled(
                ImVec2(textPos.x - padding.x, textPos.y - padding.y),
                ImVec2(textPos.x + textSize.x + padding.x, textPos.y + textSize.y + padding.y),
                IM_COL32(0, 0, 0, 160),
                4.0f);
            drawList->AddText(textPos, IM_COL32_WHITE, label.text);
        }
    }

    std::optional<std::filesystem::path> UIManager::PickDirectory(const std::filesystem::path &start)
    {
        const std::string startStr = std::filesystem::absolute(start).string();
        const char *picked = tinyfd_selectFolderDialog("Chose scenes directory", startStr.c_str());
        if (!picked)
        {
            return std::nullopt;
        }

        return std::filesystem::u8path(picked);
    }

    ControlPanelActions UIManager::renderControlPanel(const ControlPanelState &state)
    {
        ControlPanelActions actions;
        const bool loading = !state.loadingScene.empty();
        const bool hasScene = !state.activeScene.empty();
        const bool busy = loading || state.recording;

        ImGui::BeginDisabled(busy);
        if (ImGui::Button("Chose scenes directory..."))
            ;
        actions.choseScenesDir = true;

        ImGui::SameLine();
        if (ImGui::Button("Rescan"))
            actions.rescan = true;

        ImGui::EndDisabled();
        ImGui::TextDisabled("%s", state.sceneDir.string().c_str());

        ImGui::SeparatorText("Scenes");
        ImGui::BeginDisabled(busy);
        if (ImGui::BeginListBox("##scenes", ImVec2(-FLT_MIN, 8 * ImGui::GetTextLineHeightWithSpacing())))
        {
            for (const auto &file : state.sceneFiles)
            {
                const std::string label = file.lexically_relative(state.sceneDir).generic_string();
                if (ImGui::Selectable(label.c_str(), file == state.activeScene || file == state.loadingScene))
                    actions.loadScene = file;
            }

            if (state.sceneFiles.empty())
                ImGui::TextDisabled("No .obj files found");
            ImGui::EndListBox();
        }
        ImGui::EndDisabled();

        if (loading)
        {
            static const char kSpinner[] = "|/-\\";
            ImGui::Text("Loading %s %c", state.loadingScene.filename().string().c_str(), kSpinner[static_cast<int>(ImGui::GetTime() * 8.0) % 4]);
        }

        if (!state.error.empty())
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Load failed: %s", state.error.c_str());

        if (hasScene)
        {
            ImGui::Text("Active %s", state.activeScene.filename().string().c_str());
            ImGui::Text("Camera path: %s, %u frames", state.generatedPath ? "generated orbit" : "animation.json", state.animatioFrameCount);
        }

        ImGui::SeparatorText("Recording");
        if (!state.recording)
        {
            ImGui::BeginDisabled(!hasScene || loading);
            if (ImGui::Button("Start recording", ImVec2(-FLT_MIN, 0)))
                actions.startRecording = true;
            ImGui::EndDisabled();
        }
        else if (ImGui::Button("Stop recording", ImVec2(-FLT_MIN, 0)))
        {
            actions.stopRecording = true;
        }

        char overlay[32];
        std::snprintf(overlay, sizeof(overlay), "%u / %u", state.animationFrame, state.animatioFrameCount);
        const float progress = state.animatioFrameCount ? float(state.animationFrame) / float(state.animatioFrameCount) : 0.0f;

        ImGui::ProgressBar(progress, ImVec2(-FLT_MIN, 0), overlay);

        ImGui::End();
        return actions;
    }
}