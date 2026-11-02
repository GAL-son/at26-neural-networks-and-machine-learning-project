#include "Scene.hpp"

#include "Renderer/Resources/Mesh.hpp"
#include "Renderer/Resources/Texture.hpp"

namespace Engine::Renderer
{
    Scene::Scene(
        VulkanContext &context,
        VkCommandPool commandPool,
        const Engine::Scene::Scene &cpuScene,
        VkDescriptorSetLayout materialSetLayout)
        : m_device(context.getDevice()),
          m_allocator(context.getAllocator())
    {
        // 1. Konwersja i upload wszystkich tekstur na GPU
        m_gpuTextures.reserve(cpuScene.textures.size());
        for (const auto &cpuTex : cpuScene.textures)
        {
            m_gpuTextures.push_back(std::make_unique<Texture>(context, commandPool, cpuTex));
        }

        // 2. Skopiowanie czystych danych materiałowych z CPU do GPU Scene
        // Brak alokacji deskryptorów Vulcana! Materiały to teraz zwykłe dane parametrów.
        m_gpuMaterials = cpuScene.materials;

        // 3. Konwersja i upload wszystkich meshów na GPU
        m_gpuMeshes.reserve(cpuScene.meshes.size());
        for (const auto &cpuMesh : cpuScene.meshes)
        {
            m_gpuMeshes.push_back(std::make_unique<Mesh>(m_allocator, cpuMesh));
        }

        CreateMaterialResources(materialSetLayout);
    }

    Scene::~Scene()
    {
        m_materialBuffers.clear();
        if (m_materialPool != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(m_device, m_materialPool, nullptr);
    }

    VkDescriptorSet Scene::GetMaterialDescriptorSet(int materialIndex) const
    {
        if (materialIndex < 0 || static_cast<size_t>(materialIndex) >= m_materialBuffers.size())
            materialIndex = 0;

        return m_materialSets[materialIndex];
    }

    void Scene::CreateMaterialResources(VkDescriptorSetLayout materialSetLayout)
    {
        const uint32_t materialCount = static_cast<uint32_t>(m_gpuMaterials.empty() ? 1 : m_gpuMaterials.size());

        const Texture *fallbackTexture = !m_gpuMaterials.empty() ? m_gpuTextures[0].get() : nullptr;
        if (!fallbackTexture)
            throw std::runtime_error("No fallback texture");

        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, materialCount};
        poolSizes[1] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, materialCount};

        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = materialCount;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_materialPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to material create descriptor pool");

        std::vector<VkDescriptorSetLayout> layouts(materialCount, materialSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool = m_materialPool;
        allocInfo.descriptorSetCount = materialCount;
        allocInfo.pSetLayouts = layouts.data();

        m_materialSets.resize(materialCount);
        if (vkAllocateDescriptorSets(m_device, &allocInfo, m_materialSets.data()) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate descriptor sets");

        m_materialBuffers.resize(materialCount);
        for (uint32_t i = 0; i < materialCount; ++i)
        {
            Engine::Scene::Material mat;
            if (!m_gpuMaterials.empty())
                mat = m_gpuMaterials[i];

            m_materialBuffers[i] = std::make_unique<Buffer>(
                m_allocator,
                sizeof(MaterialUniformData),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VMA_MEMORY_USAGE_CPU_TO_GPU);

            MaterialUniformData uboData{};
            uboData.ambient = mat.ambient;
            uboData.diffuse = mat.diffuse;
            uboData.specular = mat.specular;
            uboData.shininess = mat.shininess;
            uboData.opacity = mat.opacity;
            uboData.hasDiffuseTexture = mat.hasDiffuseTexture() ? 1 : 0;
            m_materialBuffers[i]->UploadData(&uboData);

            const Texture *gpuTexture = fallbackTexture;
            if (
                mat.hasDiffuseTexture() &&
                mat.diffuseTextureIndex >= 0 &&
                static_cast<size_t>(mat.diffuseTextureIndex) < m_gpuTextures.size())
            {
                gpuTexture = m_gpuTextures[mat.diffuseTextureIndex].get();
            }

            VkDescriptorBufferInfo bufferInfo{m_materialBuffers[i]->GetHandle(), 0, sizeof(MaterialUniformData)};
            VkDescriptorImageInfo imageInfo{gpuTexture->GetSampler(), gpuTexture->GetImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

            std::array<VkWriteDescriptorSet, 2> writes{};
            writes[0] = {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                m_materialSets[i],
                0,
                0,
                1,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                nullptr,
                &bufferInfo,
                nullptr,
            };
            writes[1] = {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                m_materialSets[i],
                1,
                0,
                1,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                &imageInfo,
                nullptr,
                nullptr,
            };

            vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
    }

    void Scene::Draw(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout) const
    {
        for (const auto &mesh : m_gpuMeshes)
        {
            mesh->Bind(commandBuffer);
            for (const Engine::Scene::Submesh &submesh : mesh->GetSubmeshes())
            {
                VkDescriptorSet materialSet = GetMaterialDescriptorSet(submesh.materialIndex);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &materialSet, 0, nullptr);
                vkCmdDrawIndexed(commandBuffer, submesh.indexCount, 1, submesh.indexOffset, 0, 0);
            }
        }
    }

}