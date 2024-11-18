#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
    float textureOpacity;
} ubo;

layout(binding = 1) uniform sampler2D textureSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexturePosition;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 vertexColor = vec4(fragColor, 1.0);
    vec4 textureColor = texture(textureSampler, fragTexturePosition);
    outColor = mix(vertexColor, textureColor, ubo.textureOpacity);
}
