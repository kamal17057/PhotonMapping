//
// Created on 14-09-2017.
//

#include "utility.hpp"

#include <string>
#include <vector>
#include <cassert>
#include <fstream>
#include <iostream>

std::vector<char> fileToCharArr(const std::string & filename,bool printFile) {

    std::ifstream inputFileHandle;
    inputFileHandle.open(filename);
    assert(!inputFileHandle.fail());
    inputFileHandle.seekg(0, inputFileHandle.end);
    size_t len = inputFileHandle.tellg();
    inputFileHandle.seekg(0,inputFileHandle.beg);
    std::vector<char> buffer(len);
    inputFileHandle.read(&buffer[0],len);


    buffer.resize(inputFileHandle.gcount()+1);
    buffer[inputFileHandle.gcount()] = 0;

    inputFileHandle.close();


    if(printFile) {
        std::cout <<"print len is" << len << std::endl;
        for (int i = 0; i < len; ++i) {
            std::cout << buffer[i];
        }
    }
    return buffer;

}





shDf::shDf(GLenum type, const char *src, const GLint len) : type(type), src(src), len(len) {}

GLuint compileShader(const shDf & sd){

    //TODO see everything ok here
    GLuint Shader = glCreateShader(sd.type);
    glShaderSource(Shader, 1, &sd.src,&sd.len);
    glCompileShader(Shader);
// Check for compile time errors
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(Shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(Shader, 512, NULL, infoLog);
        //TODO pass info to this function to know what type of error here
        std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    return Shader;


}

GLuint makeProgram(GLuint vS, GLuint fS,bool deleteDetachShaders){
    GLint success = 0;
    GLchar infoLog[512];
// Link shaders
    GLuint sp = glCreateProgram();
    glAttachShader(sp, vS);
    glAttachShader(sp, fS);
    glLinkProgram(sp);
// Check for linking errors
    glGetProgramiv(sp, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(sp, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    if(deleteDetachShaders) {
        glDetachShader(sp,vS);glDeleteShader(vS);
        glDetachShader(sp,fS);glDeleteShader(fS);
    }
    return sp;
}



// GLFWwindow * setupGlfw(int width, int height, const std::string &title) {
//     glfwInit();

//     glfwSetTime(0);

//     glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);

//     glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//     glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
//     glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
//     glfwWindowHint(GLFW_SAMPLES, 4);
//     glfwWindowHint(GLFW_RED_BITS, 8);
//     glfwWindowHint(GLFW_GREEN_BITS, 8);
//     glfwWindowHint(GLFW_BLUE_BITS, 8);
//     glfwWindowHint(GLFW_ALPHA_BITS, 8);
//     glfwWindowHint(GLFW_STENCIL_BITS, 8);
//     glfwWindowHint(GLFW_DEPTH_BITS, 24);

//     glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

// //
// // Create a GLFWwindow object
//     GLFWwindow *window = glfwCreateWindow(width,height,title.c_str(), nullptr, nullptr);
//     if (window == nullptr) {
//         std::cout << "Failed to create GLFW window" << std::endl;
//         glfwTerminate();
//         return nullptr;
//     }
//     glfwMakeContextCurrent(window);
//     return window;
// }


