# Computer Graphics

A comprehensive guide to computer graphics fundamentals, rendering techniques, and modern graphics programming.

## Table of Contents

1. [Computer Graphics Fundamentals](#computer-graphics-fundamentals)
2. [Coordinate Systems and Transformations](#coordinate-systems-and-transformations)
3. [Graphics Pipeline](#graphics-pipeline)
4. [2D Graphics](#2d-graphics)
5. [3D Graphics](#3d-graphics)
6. [Rasterization](#rasterization)
7. [Shading and Lighting](#shading-and-lighting)
8. [Texturing](#texturing)
9. [Advanced Rendering Techniques](#advanced-rendering-techniques)
10. [Animation](#animation)
11. [Graphics APIs](#graphics-apis)
12. [Ray Tracing](#ray-tracing)
13. [GPU Architecture](#gpu-architecture)
14. [Modern Graphics Techniques](#modern-graphics-techniques)

---

## Computer Graphics Fundamentals

**Computer Graphics** is the field of visual computing that deals with generating, manipulating, and rendering visual content using computers.

### Key Concepts

1. **Rendering**: The process of generating an image from a model
2. **Rasterization**: Converting vector graphics to raster (pixel) format
3. **Pixel**: The smallest addressable element in a display device
4. **Frame Buffer**: Memory buffer containing the complete frame data
5. **Refresh Rate**: How many times per second the display is redrawn
6. **Resolution**: The number of pixels in each dimension (width × height)

### Color Models

#### RGB (Red, Green, Blue)
- **Additive color model** used in displays
- Each color component ranges from 0-255 (8-bit) or 0.0-1.0 (normalized)
- White = (255, 255, 255), Black = (0, 0, 0)
- Used in monitors, TVs, and digital displays

```python
# RGB color representation
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
white = (255, 255, 255)
```

#### RGBA (RGB + Alpha)
- Extends RGB with an alpha channel for transparency
- Alpha: 0 = fully transparent, 255 = fully opaque

#### HSV/HSL (Hue, Saturation, Value/Lightness)
- **Cylindrical color model** more intuitive for human perception
- Hue: Color type (0-360 degrees)
- Saturation: Color intensity (0-100%)
- Value/Lightness: Brightness (0-100%)

#### CMYK (Cyan, Magenta, Yellow, Black)
- **Subtractive color model** used in printing
- White paper + no ink = white
- All inks combined = black

---

## Coordinate Systems and Transformations

### Coordinate Systems

#### 1. Object/Model Space
- Local coordinate system for each object
- Origin typically at object's center or base
- Defined by the artist/modeler

#### 2. World Space
- Global coordinate system for the entire scene
- Objects are positioned relative to a common origin
- Result of applying **Model transformation**

#### 3. View/Camera Space
- Coordinate system relative to the camera
- Camera at origin, looking down -Z axis
- Result of applying **View transformation**

#### 4. Clip Space
- After projection, coordinates in [-1, 1] range (OpenGL) or [0, 1] (DirectX)
- Result of applying **Projection transformation**

#### 5. Screen Space
- Final 2D pixel coordinates on screen
- Result of **Viewport transformation**

### Transformation Matrices

#### Translation
Moves an object in space.

```
T(tx, ty, tz) = |1  0  0  tx|
                |0  1  0  ty|
                |0  0  1  tz|
                |0  0  0  1 |
```

#### Scaling
Changes object size.

```
S(sx, sy, sz) = |sx 0  0  0|
                |0  sy 0  0|
                |0  0  sz 0|
                |0  0  0  1|
```

#### Rotation
Rotates object around an axis.

**Rotation around Z-axis:**
```
Rz(θ) = |cos(θ) -sin(θ)  0  0|
        |sin(θ)  cos(θ)  0  0|
        |0       0       1  0|
        |0       0       0  1|
```

#### Model-View-Projection (MVP) Matrix
The fundamental transformation pipeline:

```
P_clip = Projection × View × Model × P_local
```

Where:
- **Model**: Transforms from object space to world space
- **View**: Transforms from world space to camera space
- **Projection**: Transforms from camera space to clip space

### Homogeneous Coordinates

Use 4D coordinates (x, y, z, w) to represent 3D points:
- Point: (x, y, z, 1)
- Vector: (x, y, z, 0)

**Benefits:**
- Enables translation using matrix multiplication
- Simplifies perspective projection
- Allows distinction between points and vectors

---

## Graphics Pipeline

The **graphics pipeline** is the sequence of steps used to render a 3D scene to a 2D image.

### Traditional Fixed-Function Pipeline

1. **Vertex Processing**
   - Transform vertices to clip space
   - Apply lighting calculations (per-vertex)
   - Generate texture coordinates

2. **Primitive Assembly**
   - Group vertices into primitives (triangles, lines, points)

3. **Rasterization**
   - Convert primitives to fragments (potential pixels)
   - Interpolate vertex attributes across fragments

4. **Fragment Processing**
   - Apply texturing
   - Calculate final color per fragment

5. **Output Merger**
   - Depth testing (Z-buffer)
   - Blending (transparency)
   - Write to framebuffer

### Modern Programmable Pipeline

```
Vertex Data
    ↓
Vertex Shader (programmable)
    ↓
Tessellation Control Shader (optional)
    ↓
Tessellation Evaluation Shader (optional)
    ↓
Geometry Shader (optional)
    ↓
Rasterization (fixed)
    ↓
Fragment/Pixel Shader (programmable)
    ↓
Output Merger (configurable)
    ↓
Frame Buffer
```

#### Vertex Shader
- Processes each vertex independently
- Transforms vertex positions (MVP transformation)
- Calculates lighting per vertex
- Outputs position and attributes for next stage

```glsl
// Simple GLSL vertex shader
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
```

#### Fragment Shader
- Processes each fragment (potential pixel)
- Calculates final color
- Applies texturing and lighting
- Can discard fragments

```glsl
// Simple GLSL fragment shader
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

void main() {
    // Ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}
```

---

## 2D Graphics

### Primitive Shapes

#### Line Drawing
**Bresenham's Line Algorithm** - efficient integer-only line drawing:

```python
def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points
```

#### Circle Drawing
**Midpoint Circle Algorithm**:

```python
def midpoint_circle(xc, yc, r):
    points = []
    x = 0
    y = r
    p = 1 - r

    while x <= y:
        # Plot 8 symmetric points
        points.extend([
            (xc + x, yc + y), (xc - x, yc + y),
            (xc + x, yc - y), (xc - x, yc - y),
            (xc + y, yc + x), (xc - y, yc + x),
            (xc + y, yc - x), (xc - y, yc - x)
        ])

        x += 1
        if p < 0:
            p += 2 * x + 1
        else:
            y -= 1
            p += 2 * (x - y) + 1

    return points
```

### Polygon Filling

#### Scanline Fill Algorithm
1. Find intersections of scanline with polygon edges
2. Sort intersections by x-coordinate
3. Fill between pairs of intersections

#### Flood Fill Algorithm
- Recursive or queue-based filling from a seed point
- Used in paint programs

```python
def flood_fill(image, x, y, new_color, old_color):
    if (x < 0 or x >= image.width or y < 0 or y >= image.height):
        return
    if image[x][y] != old_color:
        return

    image[x][y] = new_color
    flood_fill(image, x+1, y, new_color, old_color)
    flood_fill(image, x-1, y, new_color, old_color)
    flood_fill(image, x, y+1, new_color, old_color)
    flood_fill(image, x, y-1, new_color, old_color)
```

### 2D Transformations

Using 3×3 matrices with homogeneous coordinates (x, y, 1):

```
Translation: |1  0  tx|
            |0  1  ty|
            |0  0  1 |

Rotation:    |cos(θ) -sin(θ)  0|
            |sin(θ)  cos(θ)  0|
            |0       0       1|

Scaling:     |sx  0   0|
            |0   sy  0|
            |0   0   1|
```

---

## 3D Graphics

### 3D Representations

#### 1. Polygon Meshes
- Most common representation
- Surface approximated by connected polygons (usually triangles)
- **Vertices**: 3D points
- **Edges**: Lines connecting vertices
- **Faces**: Polygons formed by edges

```python
# Triangle mesh structure
class Mesh:
    def __init__(self):
        self.vertices = []  # List of (x, y, z) tuples
        self.faces = []     # List of vertex index tuples
        self.normals = []   # List of normal vectors
        self.uvs = []       # List of texture coordinates
```

#### 2. Parametric Surfaces
- Surfaces defined by mathematical functions
- Examples: Bezier surfaces, B-splines, NURBS

#### 3. Implicit Surfaces
- Defined by equations: f(x, y, z) = 0
- Examples: Spheres, metaballs

#### 4. Voxels
- 3D pixels - volumetric representation
- Used in medical imaging, scientific visualization

### Face Culling

**Back-face Culling** - don't render polygons facing away from camera:

```python
def is_front_facing(vertex0, vertex1, vertex2, camera_pos):
    # Calculate face normal
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    normal = cross(edge1, edge2)

    # Vector from face to camera
    to_camera = camera_pos - vertex0

    # If dot product is positive, face is front-facing
    return dot(normal, to_camera) > 0
```

### Projection

#### Orthographic Projection
- Parallel projection
- No perspective distortion
- Used in CAD, technical drawings

```
Ortho Matrix:
|2/(r-l)    0         0        -(r+l)/(r-l)|
|0          2/(t-b)   0        -(t+b)/(t-b)|
|0          0        -2/(f-n)  -(f+n)/(f-n)|
|0          0         0         1          |

where l,r = left,right; b,t = bottom,top; n,f = near,far
```

#### Perspective Projection
- Simulates how human eyes see
- Objects farther away appear smaller
- Parallel lines converge at vanishing points

```
Perspective Matrix (OpenGL):
|2n/(r-l)   0         (r+l)/(r-l)   0        |
|0          2n/(t-b)  (t+b)/(t-b)   0        |
|0          0        -(f+n)/(f-n)  -2fn/(f-n)|
|0          0        -1             0        |
```

**Field of View (FOV)** formulation:
```python
def perspective_matrix(fov_y, aspect, near, far):
    f = 1.0 / tan(fov_y / 2.0)
    return [
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
        [0, 0, -1, 0]
    ]
```

---

## Rasterization

**Rasterization** converts geometric primitives (triangles) into fragments (pixels).

### Triangle Rasterization

#### Scanline Rasterization
1. Sort vertices by y-coordinate
2. Interpolate edges
3. Fill horizontal spans between edges

#### Barycentric Coordinates
Used for attribute interpolation across triangles:

```python
def barycentric_coords(p, a, b, c):
    """
    Compute barycentric coordinates (u, v, w) for point p
    with respect to triangle (a, b, c)
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = dot(v0, v0)
    d01 = dot(v0, v1)
    d11 = dot(v1, v1)
    d20 = dot(v2, v0)
    d21 = dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return (u, v, w)

# Interpolate attribute at point p
def interpolate_attribute(p, a, b, c, attr_a, attr_b, attr_c):
    u, v, w = barycentric_coords(p, a, b, c)
    return u * attr_a + v * attr_b + w * attr_c
```

### Z-Buffer (Depth Buffer)

Solves the **visibility problem** - which surfaces are in front:

```python
def render_with_zbuffer(triangles, width, height):
    color_buffer = [[background_color] * width for _ in range(height)]
    z_buffer = [[float('inf')] * width for _ in range(height)]

    for triangle in triangles:
        for x, y in pixels_covered_by_triangle(triangle):
            z = interpolate_depth(triangle, x, y)

            if z < z_buffer[y][x]:
                z_buffer[y][x] = z
                color_buffer[y][x] = shade_pixel(triangle, x, y)

    return color_buffer
```

**Properties:**
- Most common visibility algorithm
- O(n) time complexity for n triangles
- Requires memory for depth buffer (typically 24 or 32 bits per pixel)
- Handles complex scenes efficiently

---

## Shading and Lighting

### Lighting Models

#### Phong Reflection Model

Models light-surface interaction with three components:

**1. Ambient**: Background illumination
```
I_ambient = k_a × I_a
```

**2. Diffuse**: Matte reflection (Lambertian)
```
I_diffuse = k_d × I_l × max(N · L, 0)
```

**3. Specular**: Shiny highlights
```
I_specular = k_s × I_l × max(R · V, 0)^α
```

**Total illumination:**
```
I = I_ambient + I_diffuse + I_specular
```

Where:
- `k_a, k_d, k_s`: Ambient, diffuse, specular coefficients
- `I_a, I_l`: Ambient and light intensities
- `N`: Surface normal
- `L`: Light direction
- `R`: Reflection direction
- `V`: View direction
- `α`: Shininess exponent

#### Blinn-Phong Model

More efficient variation using halfway vector:

```
I_specular = k_s × I_l × max(N · H, 0)^α

where H = normalize(L + V)
```

### Shading Techniques

#### Flat Shading
- One color per polygon
- Fast but faceted appearance
- Suitable for low-poly models

```python
def flat_shade(triangle, light_dir):
    normal = calculate_face_normal(triangle)
    intensity = max(dot(normal, light_dir), 0)
    return base_color * intensity
```

#### Gouraud Shading (Smooth Shading)
- Calculate lighting at vertices
- Interpolate colors across face
- Smooth appearance, faster than Phong

```python
def gouraud_shade(triangle, light_dir):
    # Calculate intensity at each vertex
    i0 = phong_lighting(triangle.v0, triangle.n0, light_dir)
    i1 = phong_lighting(triangle.v1, triangle.n1, light_dir)
    i2 = phong_lighting(triangle.v2, triangle.n2, light_dir)

    # Interpolate across triangle
    for pixel in triangle:
        u, v, w = barycentric_coords(pixel, triangle)
        intensity = u*i0 + v*i1 + w*i2
        pixel.color = base_color * intensity
```

#### Phong Shading
- Interpolate normals across face
- Calculate lighting per pixel
- High quality, more expensive

```python
def phong_shade(triangle, light_dir, view_dir):
    for pixel in triangle:
        # Interpolate normal at pixel
        u, v, w = barycentric_coords(pixel, triangle)
        normal = normalize(u*n0 + v*n1 + w*n2)

        # Calculate lighting for this pixel
        intensity = phong_lighting(pixel.pos, normal, light_dir, view_dir)
        pixel.color = base_color * intensity
```

### Light Types

#### 1. Directional Light
- Parallel rays (sun-like)
- No position, only direction
- Same intensity everywhere

```glsl
vec3 directional_light(vec3 direction, vec3 normal) {
    return max(dot(normal, -direction), 0.0);
}
```

#### 2. Point Light
- Radiates in all directions
- Intensity decreases with distance (attenuation)

```glsl
vec3 point_light(vec3 lightPos, vec3 fragPos, vec3 normal) {
    vec3 lightDir = normalize(lightPos - fragPos);
    float distance = length(lightPos - fragPos);
    float attenuation = 1.0 / (constant + linear * distance +
                               quadratic * distance * distance);
    return max(dot(normal, lightDir), 0.0) * attenuation;
}
```

#### 3. Spot Light
- Cone of light from a point
- Has position, direction, and cutoff angle

```glsl
vec3 spot_light(vec3 lightPos, vec3 lightDir, vec3 fragPos, vec3 normal) {
    vec3 toFragment = normalize(fragPos - lightPos);
    float theta = dot(toFragment, normalize(lightDir));

    if (theta > cutoff) {
        // Inside spotlight cone
        float intensity = (theta - outerCutoff) / (cutoff - outerCutoff);
        return intensity * point_light(lightPos, fragPos, normal);
    }
    return vec3(0.0);
}
```

#### 4. Area Light
- Extended light source
- Soft shadows
- More computationally expensive

---

## Texturing

**Texture mapping** applies images (textures) to 3D surfaces.

### Texture Coordinates (UV Mapping)

- Map 3D surface to 2D texture space
- U, V coordinates typically in range [0, 1]
- Assigned to vertices, interpolated across faces

```python
class Vertex:
    def __init__(self, position, normal, uv):
        self.position = position  # (x, y, z)
        self.normal = normal      # (nx, ny, nz)
        self.uv = uv             # (u, v)
```

### Texture Filtering

#### Nearest Neighbor (Point Sampling)
- Use closest texel
- Fast but blocky when magnified

```python
def nearest_neighbor(texture, u, v):
    x = int(u * texture.width)
    y = int(v * texture.height)
    return texture[y][x]
```

#### Bilinear Filtering
- Interpolate between 4 nearest texels
- Smoother results

```python
def bilinear_filter(texture, u, v):
    x = u * (texture.width - 1)
    y = v * (texture.height - 1)

    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1

    # Fractional parts
    fx = x - x0
    fy = y - y0

    # Get 4 texel colors
    c00 = texture[y0][x0]
    c10 = texture[y0][x1]
    c01 = texture[y1][x0]
    c11 = texture[y1][x1]

    # Interpolate
    c0 = lerp(c00, c10, fx)
    c1 = lerp(c01, c11, fx)
    return lerp(c0, c1, fy)
```

#### Trilinear Filtering
- Bilinear filtering + interpolation between mipmap levels
- Reduces aliasing

#### Anisotropic Filtering
- Adapts to surface angle
- Best quality, most expensive
- Common in modern games (2x, 4x, 8x, 16x)

### Mipmapping

Pre-filtered texture pyramid for different distances:

```
Level 0: 1024×1024 (original)
Level 1: 512×512
Level 2: 256×256
...
Level 10: 1×1
```

**Benefits:**
- Reduces aliasing at distance
- Improves performance (better cache coherency)
- 33% more memory (1 + 1/4 + 1/16 + ... = 4/3)

```python
def generate_mipmaps(texture):
    mipmaps = [texture]
    current = texture

    while current.width > 1 and current.height > 1:
        # Downsample by averaging 2×2 blocks
        next_level = Image(current.width // 2, current.height // 2)
        for y in range(next_level.height):
            for x in range(next_level.width):
                next_level[y][x] = average([
                    current[2*y][2*x],
                    current[2*y][2*x+1],
                    current[2*y+1][2*x],
                    current[2*y+1][2*x+1]
                ])
        mipmaps.append(next_level)
        current = next_level

    return mipmaps
```

### Advanced Texture Types

#### 1. Normal Mapping
- Store surface normals in texture
- Add detail without geometry
- RGB → normal vector (x, y, z)

```glsl
vec3 normal_mapping(sampler2D normalMap, vec2 uv, vec3 tangent, vec3 bitangent, vec3 normal) {
    // Sample normal from texture
    vec3 texNormal = texture(normalMap, uv).rgb * 2.0 - 1.0;

    // Transform from tangent space to world space
    mat3 TBN = mat3(tangent, bitangent, normal);
    return normalize(TBN * texNormal);
}
```

#### 2. Displacement Mapping
- Actually modify geometry based on texture
- More expensive than normal mapping
- True geometric detail

#### 3. Specular Mapping
- Control specular intensity per pixel
- Allows different material properties on one surface

#### 4. Environment Mapping (Reflection Mapping)
- Simulate reflections using pre-rendered environment
- Cube maps: 6 textures forming a cube
- Sphere maps: single texture mapped to sphere

```glsl
vec3 environment_mapping(samplerCube envMap, vec3 viewDir, vec3 normal) {
    vec3 reflected = reflect(viewDir, normal);
    return texture(envMap, reflected).rgb;
}
```

#### 5. Shadow Mapping
- Store depth from light's perspective
- Compare with fragment depth to determine shadow

---

## Advanced Rendering Techniques

### Physically Based Rendering (PBR)

Modern rendering approach based on physical light behavior.

#### Key Principles
1. **Energy Conservation**: Reflected light never exceeds incoming light
2. **Fresnel Effect**: Reflectivity varies with viewing angle
3. **Microsurface Theory**: Surfaces have microscopic geometry

#### PBR Material Properties

**Metallic Workflow:**
- Base Color (albedo)
- Metallic (0 = dielectric, 1 = metal)
- Roughness (0 = smooth, 1 = rough)
- Ambient Occlusion

**Specular Workflow:**
- Diffuse Color
- Specular Color
- Glossiness

#### Cook-Torrance BRDF

```glsl
vec3 cook_torrance(vec3 N, vec3 V, vec3 L, vec3 albedo, float roughness, float metallic) {
    vec3 H = normalize(V + L);

    // Normal Distribution Function (GGX/Trowbridge-Reitz)
    float NDF = DistributionGGX(N, H, roughness);

    // Geometry Function (Smith's method)
    float G = GeometrySmith(N, V, L, roughness);

    // Fresnel (Schlick's approximation)
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    // Specular term
    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;

    // Energy conservation
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

    // Lambertian diffuse
    vec3 diffuse = kD * albedo / PI;

    float NdotL = max(dot(N, L), 0.0);
    return (diffuse + specular) * NdotL;
}
```

### Deferred Shading

Separate geometry rendering from lighting calculations.

#### G-Buffer (Geometry Buffer)
Multiple render targets storing:
1. Position
2. Normal
3. Albedo/Color
4. Specular
5. Depth

```glsl
// G-Buffer pass (fragment shader)
layout (location = 0) out vec3 gPosition;
layout (location = 1) out vec3 gNormal;
layout (location = 2) out vec4 gAlbedoSpec;

void main() {
    gPosition = FragPos;
    gNormal = normalize(Normal);
    gAlbedoSpec.rgb = texture(albedoMap, TexCoords).rgb;
    gAlbedoSpec.a = texture(specularMap, TexCoords).r;
}

// Lighting pass
vec3 lighting = vec3(0.0);
for (Light light : lights) {
    vec3 position = texture(gPosition, TexCoords).rgb;
    vec3 normal = texture(gNormal, TexCoords).rgb;
    vec3 albedo = texture(gAlbedoSpec, TexCoords).rgb;

    lighting += calculate_light(light, position, normal, albedo);
}
```

**Advantages:**
- Handle many lights efficiently
- Lighting calculated once per visible pixel
- Separate geometry and lighting complexity

**Disadvantages:**
- High memory bandwidth
- No hardware MSAA support
- Transparency requires separate pass

### Screen Space Techniques

#### Screen Space Ambient Occlusion (SSAO)
- Approximate ambient occlusion in screen space
- Sample depth buffer around each pixel
- Darken occluded areas

```glsl
float ssao(vec2 texCoord, vec3 position, vec3 normal) {
    float occlusion = 0.0;

    for (int i = 0; i < kernelSize; i++) {
        // Sample position
        vec3 samplePos = position + kernel[i] * radius;

        // Project to screen space
        vec4 offset = projection * vec4(samplePos, 1.0);
        offset.xy /= offset.w;
        offset.xy = offset.xy * 0.5 + 0.5;

        // Get depth
        float sampleDepth = texture(depthTexture, offset.xy).r;

        // Range check and accumulate
        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(position.z - sampleDepth));
        occlusion += (sampleDepth >= samplePos.z ? 1.0 : 0.0) * rangeCheck;
    }

    return 1.0 - (occlusion / kernelSize);
}
```

#### Screen Space Reflections (SSR)
- Ray march through depth buffer
- Approximate reflections without environment maps
- Works well for planar surfaces

### Shadow Techniques

#### 1. Shadow Mapping
```glsl
// Render depth from light's perspective
float shadow = 0.0;
vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
projCoords = projCoords * 0.5 + 0.5;

float closestDepth = texture(shadowMap, projCoords.xy).r;
float currentDepth = projCoords.z;

shadow = currentDepth > closestDepth ? 1.0 : 0.0;
```

#### 2. Percentage Closer Filtering (PCF)
- Sample multiple shadow map locations
- Soft shadow edges

```glsl
float shadow = 0.0;
vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
for(int x = -1; x <= 1; x++) {
    for(int y = -1; y <= 1; y++) {
        float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
        shadow += currentDepth > pcfDepth ? 1.0 : 0.0;
    }
}
shadow /= 9.0;
```

#### 3. Cascaded Shadow Maps (CSM)
- Multiple shadow maps for different distances
- Higher resolution near camera
- Common in outdoor scenes

#### 4. Variance Shadow Maps (VSM)
- Store depth and depth² in shadow map
- Use Chebyshev's inequality for smooth shadows

---

## Animation

### Keyframe Animation

Store key poses at specific times, interpolate between them.

```python
class KeyFrame:
    def __init__(self, time, value):
        self.time = time
        self.value = value

class Animation:
    def __init__(self):
        self.keyframes = []

    def add_keyframe(self, time, value):
        self.keyframes.append(KeyFrame(time, value))
        self.keyframes.sort(key=lambda k: k.time)

    def evaluate(self, time):
        # Find surrounding keyframes
        for i in range(len(self.keyframes) - 1):
            k0 = self.keyframes[i]
            k1 = self.keyframes[i + 1]

            if k0.time <= time <= k1.time:
                # Interpolate
                t = (time - k0.time) / (k1.time - k0.time)
                return self.interpolate(k0.value, k1.value, t)

        return self.keyframes[-1].value
```

### Interpolation Methods

#### Linear Interpolation (LERP)
```python
def lerp(a, b, t):
    return a + (b - a) * t
```

#### Spherical Linear Interpolation (SLERP)
For quaternions (rotations):

```python
def slerp(q1, q2, t):
    dot = q1.dot(q2)

    # Clamp dot product
    dot = max(-1.0, min(1.0, dot))

    theta = acos(dot) * t
    q3 = (q2 - q1 * dot).normalize()

    return q1 * cos(theta) + q3 * sin(theta)
```

#### Cubic Hermite Spline (Smooth)
```python
def hermite(p0, p1, m0, m1, t):
    t2 = t * t
    t3 = t2 * t

    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2

    return h00*p0 + h10*m0 + h01*p1 + h11*m1
```

### Skeletal Animation (Skinning)

#### Skeleton Structure
```python
class Bone:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.local_transform = Matrix4x4.identity()
        self.inverse_bind_pose = Matrix4x4.identity()

    def get_world_transform(self):
        if self.parent:
            return self.parent.get_world_transform() * self.local_transform
        return self.local_transform
```

#### Vertex Skinning
```glsl
// Vertex shader with skinning
const int MAX_BONES = 100;
const int MAX_BONE_INFLUENCE = 4;

uniform mat4 bones[MAX_BONES];

in vec3 position;
in ivec4 boneIDs;
in vec4 weights;

void main() {
    mat4 boneTransform = bones[boneIDs[0]] * weights[0];
    boneTransform += bones[boneIDs[1]] * weights[1];
    boneTransform += bones[boneIDs[2]] * weights[2];
    boneTransform += bones[boneIDs[3]] * weights[3];

    vec4 localPosition = boneTransform * vec4(position, 1.0);
    gl_Position = projection * view * model * localPosition;
}
```

### Inverse Kinematics (IK)

Calculate joint angles to reach a target position.

```python
def solve_two_bone_ik(root, mid, end, target):
    """
    Solve 2-bone IK (e.g., arm: shoulder-elbow-wrist)
    """
    # Distances
    a = distance(root, mid)  # Upper bone
    b = distance(mid, end)   # Lower bone
    c = distance(root, target)  # To target

    # Law of cosines
    # Angle at middle joint
    cos_B = (a*a + b*b - c*c) / (2*a*b)
    cos_B = clamp(cos_B, -1, 1)
    angle_B = acos(cos_B)

    # Angle at root
    cos_A = (a*a + c*c - b*b) / (2*a*c)
    cos_A = clamp(cos_A, -1, 1)
    angle_A = acos(cos_A)

    # Calculate rotations
    to_target = normalize(target - root)
    to_mid = normalize(mid - root)

    # Apply rotations to skeleton
    root.rotation = quaternion_from_to(to_mid, to_target) * angle_A
    mid.rotation = quaternion_axis_angle(perpendicular(to_mid), angle_B)
```

### Blend Shapes (Morph Targets)

Linear interpolation between different mesh shapes.

```python
def blend_shapes(base_mesh, targets, weights):
    """
    targets: list of displacement vectors
    weights: blend weight for each target
    """
    result = base_mesh.copy()

    for i, (target, weight) in enumerate(zip(targets, weights)):
        for v in range(len(result.vertices)):
            result.vertices[v] += target.displacements[v] * weight

    return result
```

---

## Graphics APIs

### OpenGL

Cross-platform graphics API, widely supported.

#### Basic OpenGL Rendering Loop

```c
// Initialization
GLuint VAO, VBO;
glGenVertexArrays(1, &VAO);
glGenBuffers(1, &VBO);

glBindVertexArray(VAO);
glBindBuffer(GL_ARRAY_BUFFER, VBO);
glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
glEnableVertexAttribArray(0);

// Render loop
while (!glfwWindowShouldClose(window)) {
    // Clear
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Use shader
    glUseProgram(shaderProgram);

    // Set uniforms
    glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));

    // Draw
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, vertexCount);

    // Swap buffers
    glfwSwapBuffers(window);
    glfwPollEvents();
}
```

#### OpenGL Versions
- **OpenGL 2.1**: Fixed-function pipeline
- **OpenGL 3.3**: Core profile, deprecated fixed-function
- **OpenGL 4.x**: Compute shaders, advanced features
- **OpenGL ES**: Mobile/embedded variant
- **WebGL**: JavaScript binding for browsers (based on OpenGL ES)

### DirectX

Microsoft's graphics API for Windows and Xbox.

#### DirectX 11 Example

```cpp
// Create device and swap chain
D3D_FEATURE_LEVEL featureLevel;
ID3D11Device* device;
ID3D11DeviceContext* context;
IDXGISwapChain* swapChain;

D3D11CreateDeviceAndSwapChain(
    nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
    0, nullptr, 0, D3D11_SDK_VERSION,
    &swapChainDesc, &swapChain,
    &device, &featureLevel, &context
);

// Render loop
while (running) {
    // Clear
    context->ClearRenderTargetView(renderTargetView, clearColor);
    context->ClearDepthStencilView(depthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0);

    // Set shaders
    context->VSSetShader(vertexShader, nullptr, 0);
    context->PSSetShader(pixelShader, nullptr, 0);

    // Draw
    context->DrawIndexed(indexCount, 0, 0);

    // Present
    swapChain->Present(1, 0);
}
```

#### DirectX Versions
- **DirectX 9**: Legacy, still used in some older games
- **DirectX 11**: Widely supported, good balance
- **DirectX 12**: Low-level, explicit control, more complex

### Vulkan

Modern low-level cross-platform API.

**Key Concepts:**
- **Instance**: Connection to Vulkan library
- **Physical Device**: GPU representation
- **Logical Device**: Interface to physical device
- **Queue**: Submit command buffers
- **Command Buffer**: Record rendering commands
- **Pipeline**: Complete rendering state

```cpp
// Create instance
VkInstanceCreateInfo createInfo{};
createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
createInfo.pApplicationInfo = &appInfo;

VkInstance instance;
vkCreateInstance(&createInfo, nullptr, &instance);

// Create logical device
VkDevice device;
vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);

// Create command pool
VkCommandPool commandPool;
vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);

// Record command buffer
vkBeginCommandBuffer(commandBuffer, &beginInfo);
vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
vkCmdDraw(commandBuffer, vertexCount, 1, 0, 0);
vkCmdEndRenderPass(commandBuffer);
vkEndCommandBuffer(commandBuffer);

// Submit and present
vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence);
vkQueuePresentKHR(presentQueue, &presentInfo);
```

**Advantages:**
- Explicit control over GPU
- Multi-threaded command buffer recording
- Less driver overhead
- Better performance potential

**Disadvantages:**
- Verbose (1000+ lines for triangle)
- Complex memory management
- Steep learning curve

### Metal

Apple's graphics API for iOS and macOS.

```swift
// Create device
let device = MTLCreateSystemDefaultDevice()
let commandQueue = device.makeCommandQueue()

// Create render pipeline
let pipelineDescriptor = MTLRenderPipelineDescriptor()
pipelineDescriptor.vertexFunction = vertexFunction
pipelineDescriptor.fragmentFunction = fragmentFunction
let pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)

// Render
let commandBuffer = commandQueue.makeCommandBuffer()
let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)

renderEncoder.setRenderPipelineState(pipelineState)
renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

renderEncoder.endEncoding()
commandBuffer.present(drawable)
commandBuffer.commit()
```

### WebGL

OpenGL ES for the web.

```javascript
// Get context
const canvas = document.getElementById('canvas');
const gl = canvas.getContext('webgl2');

// Create shader program
const program = gl.createProgram();
gl.attachShader(program, vertexShader);
gl.attachShader(program, fragmentShader);
gl.linkProgram(program);
gl.useProgram(program);

// Create buffer
const buffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

// Render loop
function render() {
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.uniformMatrix4fv(mvpLocation, false, mvpMatrix);
    gl.drawArrays(gl.TRIANGLES, 0, vertexCount);

    requestAnimationFrame(render);
}
render();
```

---

## Ray Tracing

**Ray tracing** simulates light physics by tracing rays from camera through pixels.

### Basic Ray Tracing Algorithm

```python
def ray_trace(scene, camera, width, height):
    image = create_image(width, height)

    for y in range(height):
        for x in range(width):
            # Generate ray from camera through pixel
            ray = camera.generate_ray(x, y, width, height)

            # Trace ray and get color
            color = trace_ray(scene, ray, max_depth=5)

            image[y][x] = color

    return image

def trace_ray(scene, ray, depth):
    if depth <= 0:
        return BLACK

    # Find closest intersection
    hit = scene.intersect(ray)

    if not hit:
        return scene.background_color

    # Calculate shading at hit point
    color = shade(scene, hit, ray)

    # Handle reflection
    if hit.material.reflective:
        reflect_dir = reflect(ray.direction, hit.normal)
        reflect_ray = Ray(hit.point + hit.normal * EPSILON, reflect_dir)
        reflect_color = trace_ray(scene, reflect_ray, depth - 1)
        color = color * (1 - hit.material.reflectivity) + reflect_color * hit.material.reflectivity

    # Handle refraction (transparency)
    if hit.material.transparent:
        refract_dir = refract(ray.direction, hit.normal, hit.material.ior)
        refract_ray = Ray(hit.point - hit.normal * EPSILON, refract_dir)
        refract_color = trace_ray(scene, refract_ray, depth - 1)
        color = mix(color, refract_color, hit.material.transparency)

    return color
```

### Ray-Object Intersection

#### Ray-Sphere Intersection

```python
class Sphere:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def intersect(self, ray):
        # Ray: P(t) = origin + t * direction
        # Sphere: |P - center|² = radius²

        oc = ray.origin - self.center
        a = dot(ray.direction, ray.direction)
        b = 2.0 * dot(oc, ray.direction)
        c = dot(oc, oc) - self.radius * self.radius

        discriminant = b*b - 4*a*c

        if discriminant < 0:
            return None  # No intersection

        t = (-b - sqrt(discriminant)) / (2.0 * a)

        if t < 0:
            return None  # Behind ray origin

        hit_point = ray.at(t)
        normal = normalize(hit_point - self.center)

        return Hit(t, hit_point, normal, self)
```

#### Ray-Triangle Intersection (Möller-Trumbore)

```python
def ray_triangle_intersect(ray, v0, v1, v2):
    edge1 = v1 - v0
    edge2 = v2 - v0

    h = cross(ray.direction, edge2)
    a = dot(edge1, h)

    if abs(a) < EPSILON:
        return None  # Ray parallel to triangle

    f = 1.0 / a
    s = ray.origin - v0
    u = f * dot(s, h)

    if u < 0.0 or u > 1.0:
        return None

    q = cross(s, edge1)
    v = f * dot(ray.direction, q)

    if v < 0.0 or u + v > 1.0:
        return None

    t = f * dot(edge2, q)

    if t > EPSILON:
        hit_point = ray.at(t)
        normal = normalize(cross(edge1, edge2))
        return Hit(t, hit_point, normal, (u, v))

    return None
```

### Acceleration Structures

#### Bounding Volume Hierarchy (BVH)

```python
class BVHNode:
    def __init__(self, objects):
        if len(objects) == 1:
            self.left = self.right = None
            self.object = objects[0]
            self.bbox = objects[0].bounding_box()
        else:
            # Split objects
            axis = random.choice([0, 1, 2])
            objects.sort(key=lambda obj: obj.center()[axis])
            mid = len(objects) // 2

            self.left = BVHNode(objects[:mid])
            self.right = BVHNode(objects[mid:])
            self.object = None
            self.bbox = union(self.left.bbox, self.right.bbox)

    def intersect(self, ray):
        if not self.bbox.intersect(ray):
            return None

        if self.object:
            return self.object.intersect(ray)

        hit_left = self.left.intersect(ray) if self.left else None
        hit_right = self.right.intersect(ray) if self.right else None

        if hit_left and hit_right:
            return hit_left if hit_left.t < hit_right.t else hit_right
        return hit_left or hit_right
```

### Path Tracing (Global Illumination)

More physically accurate than basic ray tracing.

```python
def path_trace(scene, ray, depth):
    if depth <= 0:
        return BLACK

    hit = scene.intersect(ray)
    if not hit:
        return scene.background_color

    # Direct lighting
    direct = sample_lights(scene, hit)

    # Indirect lighting (Monte Carlo integration)
    if random.random() < 0.5:  # Russian roulette
        # Sample random direction in hemisphere
        random_dir = sample_hemisphere(hit.normal)
        indirect_ray = Ray(hit.point + hit.normal * EPSILON, random_dir)
        indirect = path_trace(scene, indirect_ray, depth - 1)

        # BRDF evaluation
        brdf = hit.material.evaluate_brdf(ray.direction, random_dir, hit.normal)
        cos_theta = max(0, dot(hit.normal, random_dir))

        return direct + 2.0 * brdf * indirect * cos_theta

    return direct
```

### Real-Time Ray Tracing

Modern GPUs (NVIDIA RTX, AMD RDNA 2) support hardware-accelerated ray tracing.

#### DirectX Raytracing (DXR)

```hlsl
[shader("raygeneration")]
void RayGen() {
    uint2 launchIndex = DispatchRaysIndex().xy;

    RayDesc ray;
    ray.Origin = cameraPos;
    ray.Direction = calculateRayDirection(launchIndex);
    ray.TMin = 0.001;
    ray.TMax = 10000.0;

    RayPayload payload;
    TraceRay(scene, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

    output[launchIndex] = payload.color;
}

[shader("closesthit")]
void ClosestHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    payload.color = shade(attr);
}
```

---

## GPU Architecture

### GPU vs CPU

| Feature | CPU | GPU |
|---------|-----|-----|
| Cores | Few (4-64) | Thousands |
| Clock Speed | High (3-5 GHz) | Lower (1-2 GHz) |
| Design | Latency optimized | Throughput optimized |
| Cache | Large | Small per core |
| Best For | Serial tasks | Parallel tasks |

### GPU Pipeline

```
Application (CPU)
    ↓
Command Processor (GPU)
    ↓
Vertex Fetch
    ↓
Vertex Shader (Programmable)
    ↓
Tessellation (Optional)
    ↓
Geometry Shader (Optional)
    ↓
Rasterizer (Fixed)
    ↓
Pixel/Fragment Shader (Programmable)
    ↓
ROP (Render Output Unit)
    ↓
Frame Buffer
```

### SIMD and Warps

**SIMD** (Single Instruction, Multiple Data):
- Same instruction executed on multiple data simultaneously
- GPU cores execute in groups (warps/wavefronts)
- Warp size: 32 (NVIDIA), 64 (AMD)

**Divergence:**
```glsl
// Bad: causes thread divergence
if (threadID % 2 == 0) {
    // Half the warp executes this
    result = expensiveOperation1();
} else {
    // Other half executes this
    result = expensiveOperation2();
}
// Both paths must be executed, other threads idle

// Better: avoid divergence
result = mix(expensiveOperation1(), expensiveOperation2(), threadID % 2);
```

### Memory Hierarchy

1. **Registers**: Fastest, per-thread, very limited
2. **Shared/Local Memory**: Fast, shared within workgroup
3. **Constant Memory**: Read-only, cached
4. **Texture Memory**: Optimized for 2D spatial access
5. **Global Memory**: Slowest, largest, accessible by all

### Compute Shaders

General-purpose GPU computing within graphics pipeline.

```glsl
#version 430

layout (local_size_x = 16, local_size_y = 16) in;
layout (rgba32f, binding = 0) uniform image2D imgOutput;

void main() {
    ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);

    // Perform computation
    vec4 color = computePixelColor(pixelCoords);

    imageStore(imgOutput, pixelCoords, color);
}
```

**Use Cases:**
- Particle systems
- Post-processing effects
- Physics simulation
- Image processing
- Procedural generation

---

## Modern Graphics Techniques

### Temporal Anti-Aliasing (TAA)

Combines current and previous frames to reduce aliasing.

```glsl
vec4 TAA(vec2 uv, vec4 currentColor, sampler2D historyTexture) {
    // Reproject to previous frame
    vec2 velocity = texture(velocityBuffer, uv).xy;
    vec2 prevUV = uv - velocity;

    // Sample history
    vec4 historyColor = texture(historyTexture, prevUV);

    // Neighborhood clamping to reduce ghosting
    vec4 nearColor0 = textureOffset(currentTexture, uv, ivec2(1, 0));
    vec4 nearColor1 = textureOffset(currentTexture, uv, ivec2(-1, 0));
    vec4 nearColor2 = textureOffset(currentTexture, uv, ivec2(0, 1));
    vec4 nearColor3 = textureOffset(currentTexture, uv, ivec2(0, -1));

    vec4 boxMin = min(currentColor, min(min(nearColor0, nearColor1), min(nearColor2, nearColor3)));
    vec4 boxMax = max(currentColor, max(max(nearColor0, nearColor1), max(nearColor2, nearColor3)));

    historyColor = clamp(historyColor, boxMin, boxMax);

    // Blend
    float blendFactor = 0.1;
    return mix(historyColor, currentColor, blendFactor);
}
```

### High Dynamic Range (HDR)

Represent wider range of luminance values.

```glsl
// Tone mapping (Reinhard)
vec3 reinhard(vec3 hdrColor) {
    return hdrColor / (hdrColor + vec3(1.0));
}

// Tone mapping (ACES Filmic)
vec3 acesFilmic(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Exposure adjustment
vec3 exposureToneMapping(vec3 hdrColor, float exposure) {
    vec3 exposed = hdrColor * exposure;
    return acesFilmic(exposed);
}
```

### Bloom

Glow effect for bright areas.

```glsl
// 1. Extract bright areas
vec3 extractBright(vec3 color, float threshold) {
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    return brightness > threshold ? color : vec3(0.0);
}

// 2. Gaussian blur (separable)
vec3 gaussianBlur(sampler2D tex, vec2 uv, vec2 direction) {
    vec3 result = vec3(0.0);
    float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    result += texture(tex, uv).rgb * weights[0];
    for(int i = 1; i < 5; i++) {
        result += texture(tex, uv + direction * i).rgb * weights[i];
        result += texture(tex, uv - direction * i).rgb * weights[i];
    }
    return result;
}

// 3. Combine with original
vec3 finalColor = originalColor + bloomColor * bloomIntensity;
```

### Level of Detail (LOD)

Render different detail levels based on distance/importance.

```python
class LODMesh:
    def __init__(self):
        self.lods = [
            (1000.0, high_poly_mesh),    # < 1000 units
            (5000.0, medium_poly_mesh),  # < 5000 units
            (float('inf'), low_poly_mesh) # > 5000 units
        ]

    def get_mesh(self, distance):
        for threshold, mesh in self.lods:
            if distance < threshold:
                return mesh
        return self.lods[-1][1]
```

### Frustum Culling

Don't render objects outside camera view.

```python
def frustum_cull(camera, objects):
    planes = extract_frustum_planes(camera.view_projection)
    visible = []

    for obj in objects:
        bbox = obj.bounding_box
        if is_bbox_in_frustum(bbox, planes):
            visible.append(obj)

    return visible

def is_bbox_in_frustum(bbox, planes):
    for plane in planes:
        # Test if all corners are on negative side of plane
        if all(plane.distance(corner) < 0 for corner in bbox.corners):
            return False  # Completely outside
    return True  # At least partially inside
```

### Occlusion Culling

Don't render objects hidden behind others.

```glsl
// Hierarchical Z-buffer approach
// 1. Render depth of occluders to mip-mapped depth buffer
// 2. Test object bounds against appropriate mip level

bool isOccluded(vec3 bboxMin, vec3 bboxMax, sampler2D hierZ) {
    // Project bounding box to screen space
    vec4 screenMin = projection * view * vec4(bboxMin, 1.0);
    vec4 screenMax = projection * view * vec4(bboxMax, 1.0);

    screenMin.xyz /= screenMin.w;
    screenMax.xyz /= screenMax.w;

    // Sample appropriate mip level
    float width = screenMax.x - screenMin.x;
    float level = log2(width * screenWidth);

    float occluderDepth = textureLod(hierZ, screenMin.xy, level).r;

    return screenMin.z > occluderDepth;
}
```

### Tessellation

Dynamically subdivide geometry on GPU.

```glsl
// Tessellation Control Shader
layout (vertices = 3) out;

void main() {
    // Pass through vertex
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

    // Set tessellation levels based on distance
    float distance = length(cameraPos - gl_in[gl_InvocationID].gl_Position.xyz);
    float tessLevel = mix(64.0, 1.0, clamp(distance / 100.0, 0.0, 1.0));

    gl_TessLevelOuter[gl_InvocationID] = tessLevel;
    gl_TessLevelInner[0] = tessLevel;
}

// Tessellation Evaluation Shader
layout (triangles, equal_spacing, ccw) in;

void main() {
    // Barycentric interpolation
    vec3 p0 = gl_TessCoord.x * gl_in[0].gl_Position.xyz;
    vec3 p1 = gl_TessCoord.y * gl_in[1].gl_Position.xyz;
    vec3 p2 = gl_TessCoord.z * gl_in[2].gl_Position.xyz;
    vec3 pos = p0 + p1 + p2;

    // Displacement mapping
    float height = texture(heightMap, uv).r;
    pos += normal * height * displacementScale;

    gl_Position = projection * view * vec4(pos, 1.0);
}
```

### Virtual Texturing (Mega Textures)

Stream texture data on demand.

- Divide large texture into tiles
- Load only visible tiles
- Indirection texture maps UV to tile location
- Used in large open-world games

### Clustered Shading

Handle many lights efficiently.

```glsl
// Divide screen into tiles (clusters)
// Assign lights to clusters
// Each pixel only processes lights in its cluster

ivec3 getCluster(vec3 fragPos) {
    ivec2 tile = ivec2(gl_FragCoord.xy / TILE_SIZE);
    int zSlice = int(log(fragPos.z) * zSlices / log(farPlane / nearPlane));
    return ivec3(tile, zSlice);
}

void main() {
    ivec3 cluster = getCluster(FragPos);

    // Get light list for this cluster
    int lightCount = clusterLightCounts[cluster];
    int lightOffset = clusterLightOffsets[cluster];

    vec3 lighting = vec3(0.0);
    for (int i = 0; i < lightCount; i++) {
        int lightIndex = clusterLightIndices[lightOffset + i];
        lighting += calculateLight(lights[lightIndex]);
    }

    FragColor = vec4(lighting * albedo, 1.0);
}
```

---

## Conclusion

Computer graphics is a vast and evolving field combining mathematics, physics, computer science, and art. Modern real-time graphics leverage:

- **Programmable pipelines** for flexibility
- **Physically-based rendering** for realism
- **Advanced algorithms** for performance
- **Parallel computing** via GPUs
- **Machine learning** for upscaling (DLSS, FSR)

### Further Topics

- **Volumetric Rendering**: Clouds, fog, subsurface scattering
- **Procedural Generation**: Noise functions, fractals
- **Non-Photorealistic Rendering**: Toon shading, sketching
- **Virtual Reality**: Stereoscopic rendering, foveated rendering
- **Ray Marching**: Distance fields, fractals
- **Neural Rendering**: NeRF, neural textures

### Resources

- **Books**: "Real-Time Rendering", "Physically Based Rendering", "Graphics Gems"
- **APIs**: OpenGL, Vulkan, DirectX, Metal, WebGL
- **Tools**: Blender, Unity, Unreal Engine, Godot
- **Websites**: Learn OpenGL, Scratchapixel, ShaderToy
