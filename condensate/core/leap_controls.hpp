#ifndef LEAP_CONTROLS_HPP
#define LEAP_CONTROLS_HPP


#include <cstdlib>
#include <cstring>
#include <cmath>


#include "Leap.h"
#include "gpcore.hpp"

using namespace Leap;

const std::string fingerNames[] = {"Thumb", "Index", "Middle", "Ring", "Pinky"};
const std::string boneNames[] = {"Metacarpal", "Proximal", "Middle", "Distal"};

void LeapControl(const Controller controller) {

  Bone::Type boneType = static_cast<Bone::Type>(3);

  const Frame frame = controller.frame();

  if (frame.fingers().extended().count()){
    HandList hands = frame.hands();
    const Hand hand = hands[0];
    const FingerList fingers = hand.fingers();
    const Finger finger = fingers[1];
    Bone bone = finger.bone(boneType);
    Leap::Vector boneEnd = bone.nextJoint();

    gpcore::chamber.spoon1.pos.x = (int) floor(gpcore::chamber.LeapProps.z * (boneEnd.x - gpcore::chamber.LeapProps.x));
    gpcore::chamber.spoon1.pos.y = (int) floor(gpcore::chamber.LeapProps.w * (gpcore::chamber.LeapProps.y - boneEnd.y));

    // // To debug the leap controller:
    // std::cout <<  std::string(4, ' ') << fingerNames[finger.type()]
    //           << std::string(6, ' ') <<  boneNames[boneType] << ", end: "<< boneEnd << std::endl;
    // std::cout <<  "cx= "<< gpcore::chamber.spoon1.pos.x  <<"   cy= "<< gpcore::chamber.spoon1.pos.y <<
    //   "   cz= "<< gpcore::chamber.spoon1.strength <<std::endl;

    if (gpcore::chamber.useLeapZ) {
      gpcore::chamber.spoon1.strength = .01*boneEnd.z *gpcore::chamber.spoon1.strengthSetting;
    } else {
      gpcore::chamber.spoon1.strength = gpcore::chamber.spoon1.strengthSetting;
    }

  } else {

      gpcore::chamber.spoon1.strength *= 1e-20; // reset
  }
}

#endif