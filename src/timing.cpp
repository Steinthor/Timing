#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <chrono>
#include <math.h>
#include <string>
#include <vector>
#include "version.h"

#include "gl3w.h"
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
// To run in windows subsystem for linux (WSL): export LIBGL_ALWAYS_INDIRECT=0

// Simple helper function to load an image into a OpenGL texture with common settings
bool LoadTextureFromFile(const char* filename, GLuint& out_texture, int& out_width, int& out_height)
{
    // Load from file
    int image_width = 0;
    int image_height = 0;
    unsigned char* image_data = stbi_load(filename, &image_width, &image_height, NULL, 4);
    if (image_data == NULL)
        return false;

    // Create a OpenGL texture identifier
    GLuint image_texture;
    glGenTextures(1, &image_texture);
    glBindTexture(GL_TEXTURE_2D, image_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

    // Upload pixels into texture
#if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
#endif
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data);
    stbi_image_free(image_data);

    out_texture = image_texture;
    out_width = image_width;
    out_height = image_height;

    return true;
}

// Simple helper function to save a OpenGL texture into an image
void saveImage(char* filepath, GLFWwindow* w) {
    int width, height;
    glfwGetFramebufferSize(w, &width, &height);
    GLsizei nrChannels = 3;
    GLsizei stride = nrChannels * width;
    stride += (stride % 4) ? (4 - stride % 4) : 0;
    GLsizei bufferSize = stride * height;
    std::vector<char> buffer(bufferSize);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
    stbi_flip_vertically_on_write(true);
    stbi_write_png(filepath, width, height, nrChannels, buffer.data(), stride);
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// returns true if 's' key was pressed
// ---------------------------------------------------------------------------------------------------------
bool processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        return true;
    }
    return false;
}

// settings
const unsigned int SCR_WIDTH = 1000;
const unsigned int SCR_HEIGHT = 1000;

const char *vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 position;\n" // 'position' is a global variable that can be referenced with glVertexAttribPointer
    "layout (location = 1) in vec2 texture_coordinate;\n"
    "out vec2 texCoord;\n"
    "void main()\n"
    "{\n"
    "   gl_Position = vec4(position.x, position.y, position.z, 1.0);\n" // gl_Position is a special variable identifier
    "   texCoord = texture_coordinate;\n"
    "}\0";
const char *fragmentShaderSource_input_color = "#version 330 core\n"
    "uniform vec3 triangleColor;\n" // 'uniform' is a type of global variable that can be written to
    "out vec4 FragColor;\n"
    "void main()\n"
    "{\n"
    "   FragColor = vec4(triangleColor, 1.0f);\n"
    "}\n\0";
const char *fragmentShaderSource_texture = "#version 330 core\n"
    "uniform sampler2D ourTexture;\n"
    "in vec2 texCoord;\n" // 'uniform' is a type of global variable that can be written to
    "out vec4 FragColor;\n"
    "void main()\n"
    "{\n"
    "   FragColor = texture(ourTexture, texCoord);\n"
    "}\n\0";

ImFont* AddDefaultFont( float pixel_size )
{
    ImGuiIO &io = ImGui::GetIO();
    ImFontConfig config;
    config.SizePixels = pixel_size;
    config.OversampleH = config.OversampleV = 1;
    config.PixelSnapH = true;
    ImFont *font = io.Fonts->AddFontDefault(&config);
    return font;
}

void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

// Function to parse arguments, returns true if app should continue, else false
bool parseArguments(int argc, char* argv[], bool& file_arg, bool& file_save) {
    file_arg = (argc >= 2);
    file_save = file_arg;

    for (int i = 1; i < argc; ++i) {
        std::string argument = argv[i];
        if (argument == "-h" || argument == "--help") {
            printHelp();
            file_arg = false;
            file_save = false;
            return false; // Exit after printing help
        }
        else if (argument == "-v" || argument == "--version") {
            printVersion();
            file_arg = false;
            file_save = false;
            return false; // Exit after printing version
        }
        else {
            std::cout << "File path specified: " << argument << "\n";
            // Handle file path
        }
    }
    return true;
}

int main(int argc, char *argv[])
{
    bool file_arg = true;
    bool file_save = true;
    if (!parseArguments(argc, argv, file_arg, file_save)) return 0;

    GLFWwindow* window;

    { // initialize the window
        /* Initialize the gl3w library */
        if (!gl3wInit()) {
            std::cout << "Failed to initialize the gl3w library" << std::endl;
            return -1;
        }

        /* Initialize the glfw library */
        if (!glfwInit()) {
            std::cout << "Failed to initialize the glfw library" << std::endl;
            return -1;
        }

        glfwSetErrorCallback(error_callback);

        glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
        glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
        glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

        /* Create a windowed mode window and its OpenGL context */
        window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Timing App", NULL, NULL);
        if (!window)
        {
            std::cout << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return -1;
        }

        /* Make the window's context current */
        glfwMakeContextCurrent(window);
        glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.IniFilename = NULL; // do not save imgui.ini file from where the program is executed
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    const char* glsl_version = "#version 150";
    ImGui_ImplOpenGL3_Init(glsl_version);

    // get corner marker images
    std::string filenames[4] = {"/usr/local/include/timing_marker0.png",
                                "/usr/local/include/timing_marker1.png",
                                "/usr/local/include/timing_marker2.png",
                                "/usr/local/include/timing_marker3.png"};
    int img_width[4] = {0,0,0,0};
    int img_height[4] = {0,0,0,0};
    GLuint images[4] = {0,0,0,0};
    for (uint i = 0; i < 4; ++i) {
        bool ret = LoadTextureFromFile(filenames[i].c_str(), images[i], img_width[i], img_height[i]);
        (void)ret; // Suppress unused variable warning
        IM_ASSERT(ret);
    }

    // build and compile our shader program
    // ------------------------------------
    // setting up vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // setting up fragment shader - one input color
    unsigned int fragmentShader1 = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader1, 1, &fragmentShaderSource_input_color, NULL);
    glCompileShader(fragmentShader1);
    // check for shader compile errors
    glGetShaderiv(fragmentShader1, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader1, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // setting up fragment shader - texture
    unsigned int fragmentShader2 = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader2, 1, &fragmentShaderSource_texture, NULL);
    glCompileShader(fragmentShader2);
    // check for shader compile errors
    glGetShaderiv(fragmentShader2, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader2, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // link shaders for 1 color
    unsigned int shaderProgram_color = glCreateProgram();
    glAttachShader(shaderProgram_color, vertexShader);
    glAttachShader(shaderProgram_color, fragmentShader1);
    glLinkProgram(shaderProgram_color);
    // check for linking errors
    glGetProgramiv(shaderProgram_color, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram_color, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    // link shaders for texture
    unsigned int shaderProgram_tex = glCreateProgram();
    glAttachShader(shaderProgram_tex, vertexShader);
    glAttachShader(shaderProgram_tex, fragmentShader2);
    glLinkProgram(shaderProgram_tex);
    // check for linking errors
    glGetProgramiv(shaderProgram_tex, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram_tex, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    // safe to delete shaders after linking
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader1);
    glDeleteShader(fragmentShader2);

    // setting up marker cells
    // origin is the center of the screen with extrema being -1.0 (left, bottom) or 1.0 (right, top)
    const float left_bot = -1.0;
    const float right_top = 1.0;
    const float marker_cell_size = 0.50;
    const float left_bot_marker_cell = marker_cell_size * left_bot;
    const float right_top_marker_cell = marker_cell_size * right_top;
    const float border_outer = 0.0667;
    const float border_inner = 0.00;
    const float left_bot_border_outer = left_bot * border_outer;
    const float left_bot_border_inner = left_bot * border_inner;
    const float left_bot_outer = left_bot - left_bot_border_outer;
    const float right_top_border_outer = right_top * border_outer;
    const float right_top_border_inner = right_top * border_inner;
    const float left_bot_inner = left_bot - left_bot_marker_cell - right_top_border_inner;
    const float right_top_outer = right_top - right_top_border_outer;
    const float right_top_inner = right_top - right_top_marker_cell - left_bot_border_inner;
    const float left_bot_timer_box = left_bot - left_bot_marker_cell;
    const float right_top_timer_box = right_top - right_top_marker_cell;


    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float marker_verts[] = {
        // planar coordinate x,y,z             // texture coordinate u,v
        // top left square
        left_bot_outer, right_top_outer, 0.0f, /*top l*/ 0.0f, 0.0f, /*bot l*/
        left_bot_outer, right_top_inner, 0.0f, /*bot l*/ 0.0f, 1.0f, /*top l*/
        left_bot_inner, right_top_outer, 0.0f, /*top r*/ 1.0f, 0.0f, /*bot r*/
        left_bot_inner, right_top_inner, 0.0f, /*bot r*/ 1.0f, 1.0f, /*top r*/
        // top right square
        right_top_inner, right_top_outer, 0.0f, /*top l*/ 0.0f, 0.0f,
        right_top_inner, right_top_inner, 0.0f, /*bot l*/ 0.0f, 1.0f,
        right_top_outer, right_top_outer, 0.0f, /*top r*/ 1.0f, 0.0f,
        right_top_outer, right_top_inner, 0.0f, /*bot r*/ 1.0f, 1.0f,
        // bot left
        left_bot_outer, left_bot_inner, 0.0f, /*top l*/ 0.0f, 0.0f,
        left_bot_outer, left_bot_outer, 0.0f, /*bot l*/ 0.0f, 1.0f,
        left_bot_inner, left_bot_inner, 0.0f, /*top r*/ 1.0f, 0.0f,
        left_bot_inner, left_bot_outer, 0.0f, /*bot r*/ 1.0f, 1.0f,
        // bot right
        right_top_inner, left_bot_inner, 0.0f, /*top l*/ 0.0f, 0.0f,
        right_top_inner, left_bot_outer, 0.0f, /*bot l*/ 0.0f, 1.0f,
        right_top_outer, left_bot_inner, 0.0f, /*top r*/ 1.0f, 0.0f,
        right_top_outer, left_bot_outer, 0.0f, /*bot r*/ 1.0f, 1.0f,
         // black square to cover the counting area
        left_bot_timer_box, right_top_timer_box, 0.0f, /*top l*/ 0.0f, 0.0f,
        left_bot_timer_box, left_bot_timer_box, 0.0f, /*bot l*/ 0.0f, 1.0f,
        right_top_timer_box, right_top_timer_box, 0.0f, /*top r*/ 1.0f, 0.0f,
        right_top_timer_box, left_bot_timer_box, 0.0f, /*bot r*/ 1.0f, 1.0f,
    };

    unsigned int marker_faces[] = {
        0, 1, 2,
        1, 2, 3,
        4, 5, 6,
        5, 6, 7,
        8, 9, 10,
        9, 10, 11,
        12, 13, 14,
        13, 14, 15,
        16, 17, 18,
        17, 18, 19,
    };

    const float box_size = right_top_timer_box - left_bot_timer_box;
    // timing_rows, timing_columns are used later to draw the timing cells
    const int timing_rows = 6;
    const int timing_columns = timing_rows;
    const float timer_cell_height = box_size / timing_rows;
    const float timer_cell_width = box_size / timing_columns;
    float vertices[timing_rows * timing_columns * 2 * 3];
    float lower_x, upper_x, temp_y;
    const float z = 0.0;
    // populating vertices
    for (int c = 0; c < timing_columns; ++c) {
        const int column = (timing_rows * 2 * 3) * c;
        lower_x = right_top_timer_box - (c+1) * timer_cell_width ;
        upper_x = right_top_timer_box - c * timer_cell_width;
        temp_y = left_bot_timer_box;
        for (int i = 0; i < timing_rows; ++i) {
            const int row = i * 6;
            vertices[column + row] = lower_x;
            vertices[column + row + 1] = temp_y;
            vertices[column + row + 2] = z;
            vertices[column + row + 3] = upper_x;
            vertices[column + row + 4] = temp_y;
            vertices[column + row + 5] = z;
            temp_y += timer_cell_height;
        }
    }
    // populating faces
    unsigned int faces[(timing_rows-1)* timing_columns * 2 * 3];
    int i1, i2, i3, v1, v2, v3;
    const int row_shift = (timing_rows -1) * 2;
    for (int c = 0; c < timing_columns; ++c) {
        for (int i = 0; i < row_shift; ++i) {
            i1 = (timing_rows - 1) * 2 * 3 * c + i * 3;
            i2 = i1 + 1;
            i3 = i1 + 2;
            v1 = timing_rows * 2 * c + i;
            v2 = v1 + 1;
            v3 = v1 + 2;
            faces[i1] = v1;
            faces[i2] = v2;
            faces[i3] = v3;
        }
    }

    // setting up buffers on the GPU
    GLuint vbo_vertices, vbo_faces, vao_faces; // GLuint/GLint is an unsigned int/int substitute that's cross-platform
    GLuint vbo_marker_vertices, vbo_marker_faces, vao_marker_faces;
    // getting references to input variables of shader programs
    GLint shader_program_position = glGetAttribLocation(shaderProgram_tex, "position");
    GLint shader_program_texcoord = glGetAttribLocation(shaderProgram_tex, "texture_coordinate");
    GLint shader_program_tricolor = glGetUniformLocation(shaderProgram_color, "triangleColor");

    // bind marker verts and faces Vertex Buffer Objects (VBO) to a Vertex Array Object (VAO)
    // VAOs can be thought of as a class with information about VBOs and attribute references
    glGenVertexArrays(1, &vao_marker_faces); // generate a VAO
    glBindVertexArray(vao_marker_faces); // sets the VAO as the current or active VAO
    glGenBuffers(1, &vbo_marker_vertices);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_marker_vertices); // makes the VBO the 'active' buffer
    glBufferData(GL_ARRAY_BUFFER, sizeof(marker_verts), marker_verts, GL_STATIC_DRAW); // copies data to the buffer
    glGenBuffers(1, &vbo_marker_faces);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_marker_faces);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(marker_faces), marker_faces, GL_STATIC_DRAW);
    // glVertexAttribPointer:
    // adds a 'attribute' pointer value to the current VAO (set with glBindVertexArray)
    // index: the index of 'position' in 'shader_program'
    // size: the number of sequential values,
    // type: the type of values,
    // normalized: should values be normalized between -1.0 and 1.0,
    // stride: how many bytes to the next position attribute in the linked VBO array
    // pointer: the offset from the start of the array to the position attribute in the linked VBO array
    glVertexAttribPointer(shader_program_position, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(shader_program_position);  // necessary to enable the use of the variable in the VAO
    // enable texture coordinate pointer in the current VAO
    glVertexAttribPointer(shader_program_texcoord, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(shader_program_texcoord);

    // bind the Vertex Array Object (VAO) first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glGenVertexArrays(1, &vao_faces);
    glBindVertexArray(vao_faces);
    glGenBuffers(1, &vbo_vertices);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices); // makes the buffer the 'active' buffer
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW); // copies data to the buffer
    glGenBuffers(1, &vbo_faces);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_faces);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(faces), faces, GL_STATIC_DRAW);
    glVertexAttribPointer(shader_program_position, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(shader_program_position);
    // the call to glVertexAttribPointer registered vbo_vertices as the vertex attribute's bound
    // vertex buffer object so afterwards we enable the variable, and then we can safely unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens.
    // Modifying other VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs)
    // when it's not directly necessary.
    glBindVertexArray(0);

    // vsync option, 0: disabled, -1: enabled, 1: wait for 1 full frame, 2: wait 2 frames, etc.
    glfwSwapInterval(-1);

    // imgui variables
    ImFont *font_large = AddDefaultFont(64);
    ImVec4 clear_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    bool win_open = true;
    // timing variables
    int count = 0;
    int max_count = 0;
    int frame_columns = 3;
    const int64_t max_col_time = 10;
    const int64_t half_col_time = max_col_time / 2;
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto time_interval = std::chrono::nanoseconds(0).count();
    auto time_sec = std::chrono::seconds(0).count();
    auto second = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)).count();

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Poll for and process events */
        glfwPollEvents();
        file_save = file_save || processInput(window);

        start = std::chrono::high_resolution_clock::now();
        count++;
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        /*
        {
            //static float f = 0.0f;
            static int counter = 0;

            ImFont *font_medium = AddDefaultFont(13);
            ImGui::PushFont(font_medium);
            ImGui::Begin("Analysis"); // Create a window and append into it.

            if (ImGui::Button("Button")) // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::PopFont();
            ImGui::End();
        }
        */
        if (!file_save) { // testing a clear "window" with just text.
            ImGui::SetNextWindowBgAlpha(0.0f);
            ImVec2 win_position = ImVec2(io.DisplaySize.x/3.5, 0);
            ImGui::SetNextWindowPos(win_position);
            ImGui::PushFont(font_large);
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0,0,0,255));
            ImGui::Begin("test", &win_open, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings |
                                            ImGuiWindowFlags_NoInputs);
            ImGui::Text("time: %03ld:%03d", time_sec, count);
            ImGui::PopStyleColor();
            //ImGui::Text("count: %03d", count);
            //ImGui::Text("max count: %03d", max_count);
            ImGui::PopFont();
            ImGui::End();
        }

        /* Render here */
        // imgui rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);

        if (!file_save) { // draw the boundary square
            glUseProgram(shaderProgram_color);
            glBindVertexArray(vao_marker_faces);
            // the whole screen
            glUniform3f(shader_program_tricolor, 0.0f, 0.0f, 0.0f);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void*) ((4 * 6) * sizeof(GL_UNSIGNED_INT)));
        }

        { // draw our marker squares
            glUseProgram(shaderProgram_tex);
            for (int i = 0; i < 4; ++i) {
                glBindTexture(GL_TEXTURE_2D, images[i]);
                glBindVertexArray(vao_marker_faces); // when using glDrawElements we're going to draw using indices 
                // provided in the element buffer object that is currently bound with glBindVertexArray
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void*) ((i * 6) * sizeof(GL_UNSIGNED_INT)));
            }
        }

        if (!file_save) { // draw our counting squares
            glUseProgram(shaderProgram_color);

            glBindVertexArray(vao_faces); // when using glDrawElements we're going to draw using indices 
            // provided in the element buffer object that is currently bound with glBindVertexArray
            glUniform3f(shader_program_tricolor, 1.0f, 1.0f, 1.0f); // set a triangle color

            // loop for displaying columns of monitor frame counts
            for (int column = 0; column < frame_columns; column++) {
                const double position = pow(max_col_time, column);
                const double column_value = count / position;
                const int64_t counter = static_cast<int>(column_value) % max_col_time;
                const int64_t lower = static_cast<float>(counter) / half_col_time  < 1.0 ? 0 : counter % half_col_time;
                const int64_t upper = static_cast<float>(counter) / half_col_time  < 1.0 ? counter : half_col_time;
                for (int i = lower; i < upper; ++i) {
                    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void*) (((column * (timing_rows - 1) + i) * 6) * sizeof(GL_UNSIGNED_INT)));
                }
            }
            // loop for displaying seconds
            for (int column = frame_columns; column < timing_columns; column++) {
                const double position = pow(max_col_time, column - frame_columns);
                const double column_value = time_sec / position;
                const int64_t counter = static_cast<int>(column_value) % max_col_time;
                const int64_t lower = static_cast<float>(counter) / half_col_time  < 1.0 ? 0 : counter % half_col_time;
                const int64_t upper = static_cast<float>(counter) / half_col_time  < 1.0 ? counter : half_col_time;
                for (int i = lower; i < upper; ++i) {
                    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void*) (((column * (timing_rows - 1) + i) * 6) * sizeof(GL_UNSIGNED_INT)));
                }
            }
        }

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        if (file_save && file_arg) {
            file_save = false;
            saveImage(argv[1], window);
        }
        stop = std::chrono::high_resolution_clock::now();
        time_interval += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        if (time_interval >= second) {
            if (max_count < count)
                max_count = count;
            time_interval = std::chrono::nanoseconds(0).count();
            count = 0;
            time_sec += std::chrono::nanoseconds(1).count();
        }
    }

    glfwTerminate();
    return 0;
}
