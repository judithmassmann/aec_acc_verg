#version 330 core

out vec2 a_Texcoords;

void main() {
    float x = float(((uint(gl_VertexID) + 2u) / 3u) % 2u); 
    float y = float(((uint(gl_VertexID) + 1u) / 3u) % 2u); 

    gl_Position = vec4(2.0f * x - 1.0f, 2.0f * y - 1.0f, 0.0f, 1.0f);
    a_Texcoords = vec2(x, y);
}