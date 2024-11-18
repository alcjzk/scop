#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
    float textureOpacity;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexturePosition;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexturePosition;

void main() {
    gl_Position = ubo.projection * ubo.view * ubo.model * vec4(inPosition, 1.0);
    int faceIndex = gl_VertexIndex / 3;
    float vertexColor = (faceIndex % 16) / 16.0;
    fragColor = vec3(vertexColor);
    fragTexturePosition = inTexturePosition;
}
