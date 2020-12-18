//
// Created on 14-09-2017.
//

#ifndef LAB5_UTILITY_HPP
#define LAB5_UTILITY_HPP

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/glew.h>
#endif

#include <vector>
#include <string>

#include <GLFW/glfw3.h>

std::vector<char> fileToCharArr(const std::string & filename,bool printFile = false);






struct shDf{
    shDf(GLenum type, const char *src,GLint len);
    GLenum type;
    const char * src;
    const GLint len;
};

GLuint compileShader(const shDf & sd);
GLuint makeProgram(GLuint vS,GLuint fS,bool deleteDetachShaders);




// GLFWwindow * setupGlfw(int width, int height, const std::string &title);





















#endif //LAB5_UTILITY_HPP
