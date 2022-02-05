#version 330 core

#define KERNEL_SIZE 22

uniform float u_FocusDistance;
uniform float u_FocusRange;
uniform float u_BokehRadius;
uniform float u_ZNear;
uniform float u_ZFar;
uniform float u_Aberration;

uniform sampler2D u_TextureColor;
uniform sampler2D u_TextureDepth;

in  vec2 a_Texcoords;
out vec4 RT_Color;

const vec2 kernel[KERNEL_SIZE] = vec2[KERNEL_SIZE](
    vec2(0, 0),
    vec2(0.53333336, 0),
    vec2(0.3325279, 0.4169768),
    vec2(-0.11867785, 0.5199616),
    vec2(-0.48051673, 0.2314047),
    vec2(-0.48051673, -0.23140468),
    vec2(-0.11867763, -0.51996166),
    vec2(0.33252785, -0.4169769),
    vec2(1, 0),
    vec2(0.90096885, 0.43388376),
    vec2(0.6234898, 0.7818315),
    vec2(0.22252098, 0.9749279),
    vec2(-0.22252095, 0.9749279),
    vec2(-0.62349, 0.7818314),
    vec2(-0.90096885, 0.43388382),
    vec2(-1, 0),
    vec2(-0.90096885, -0.43388376),
    vec2(-0.6234896, -0.7818316),
    vec2(-0.22252055, -0.974928),
    vec2(0.2225215, -0.9749278),
    vec2(0.6234897, -0.7818316),
    vec2(0.90096885, -0.43388376)
 );


float LinearizeDepth(float d, float zNear, float zFar) {
   float z_n =  2.0 * d - 1.0;
   return 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear));
}

float ComputeSoC(vec2 texcoord) {
    float depth = LinearizeDepth(textureLod(u_TextureDepth, texcoord, 0).r, u_ZNear, u_ZFar);
    return clamp((depth - u_FocusDistance) / u_FocusRange, -1, 1) * u_BokehRadius;
}

float Weight(float coc, float radius) {
    return clamp((coc - radius + 2) / 2, 0.0, 1.0);
}

vec2 BarrelDistortion(vec2 p, vec2 amt) {
    p = 2.0 * p - 1.0;
    float maxBarrelPower = sqrt(5.0);
    float radius = dot(p, p);
    p *= pow(vec2(radius), maxBarrelPower * amt);
    return p * 0.5 + 0.5;
}

vec2 BrownConradyDistortion(vec2 uv, float scalar) {
    uv = (uv - 0.5 ) * 2.0;
    
    if (true) {
        float barrelDistortion1 = -0.02 * scalar; 
        float barrelDistortion2 = +0.00 * scalar;

        float r2 = dot(uv, uv);
        uv *= 1.0 + barrelDistortion1 * r2 + barrelDistortion2 * r2 * r2;
    }  
   return 0.5 * uv + 0.5;
}

vec4 ComputeAberration(vec2 uv) {

    float scalar = 4.0 * u_Aberration;
    vec4 colourScalar = vec4(700.0, 560.0, 490.0, 1.0); 
    colourScalar /= max(max(colourScalar.x, colourScalar.y), colourScalar.z);
    colourScalar *= 2.0 * scalar;
    
    vec4 sourceCol = textureLod(u_TextureColor, uv, 0);

    vec4 color = vec4(0.0);
    for(float tap = 0.0; tap < 8.0; tap += 1.0) {
        color.r += textureLod(u_TextureColor, BrownConradyDistortion(uv, colourScalar.r), 0).r;
        color.g += textureLod(u_TextureColor, BrownConradyDistortion(uv, colourScalar.g), 0).g;
        color.b += textureLod(u_TextureColor, BrownConradyDistortion(uv, colourScalar.b), 0).b;        
        colourScalar *= 0.99;
    }
    color /= 8.0;
    return color;
}


void main() {

    vec2 texSize = 1.0f / textureSize(u_TextureColor, 0);
    vec2 texcoord = a_Texcoords;

    vec4  color = vec4(0.0f);
    float weight = 0;
  
    for (int index = 0; index < KERNEL_SIZE; index++) {
        vec2  offset = u_BokehRadius * kernel[index];
        float radius = length(offset);
        float sw = Weight(abs(ComputeSoC(texcoord + offset * texSize)), radius);
        color  += sw * ComputeAberration(texcoord + offset * texSize); // textureLod(u_TextureColor, texcoord + offset * texSize, 0);  
        weight += sw;
    }

    color *= 1.0 / weight;
    RT_Color = vec4(color.xyz, 1.0);
}
