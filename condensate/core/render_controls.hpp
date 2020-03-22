#ifndef RENDER_CONTROLS_H
#define RENDER_CONTROLS_H

#include "gpcore.hpp"

int DIM = gpcore::chamber.DIM;
static int clicked  = 0;

void click(int button, int updown, int x, int y) {
    gpcore::chamber.spoon1.pos.x = x/2;
    gpcore::chamber.spoon1.pos.y = y/2;
    clicked = !clicked;
    if (clicked) {
        gpcore::chamber.spoon1.strength = gpcore::chamber.spoon1.strengthSetting;
    } else {
        gpcore::chamber.spoon1.strength *= 1e-20; // reset
    }
    glutPostRedisplay();
}

void motion(int x, int y) {
    if (clicked)
    {
        gpcore::chamber.spoon1.pos.x = x/2;
        gpcore::chamber.spoon1.pos.y = y/2;
    } 
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
    if (key == 27) gpcore::chamber.stopSim = true;
    glutPostRedisplay();
}

void special(int key, int x, int y) {
    if (key == GLUT_KEY_DOWN) gpcore::chamber.cmapscale-=1e6;
    if (key == GLUT_KEY_UP)   gpcore::chamber.cmapscale+=1e6;
    glutPostRedisplay();
}

#endif