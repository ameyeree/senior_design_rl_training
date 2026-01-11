from math import atan2, degrees, sqrt, acos, radians, degrees

def calc_inv_kin(x, y, z, r):
    try:
        angles = [0.0, 0.0, 0.0, 0.0]
        rear_arm = 135
        forearm = 147 

        # Currently just sets the end-effector rotation to r in degrees
        angles[3] = radians(r)

        # Get Joint1 Angle (Base Joint)
        angles[0] = atan2(y,x)

        # NOTE: This is to handle the way Joint 2 and Joint 3 work to handle
        # the parallelogram joint. Starts from a reference of 90 degrees (vertical)

        # We 
        hypotenuse = pow(x,2) + pow(y,2) + pow(z,2)

        # Elvation angle from horizontal to the target
        # (Where the target is in the vertical plane)
        beta = atan2(z, sqrt(pow(x,2) + pow(y,2)))

        # The shoulder triangle angle from the law of cosines
        psi = abs(acos(( hypotenuse + pow(rear_arm,2) - pow(forearm,2) ) / ( 2 * rear_arm * sqrt(hypotenuse) )))

        
        if beta >= 0:   # If target above the horizontal plane
            angles[1] = radians(90) - (beta + psi)
        elif beta < 0:  # If target below the horizontal plane
            angles[1] = radians(90) - (psi - abs(beta))

        # This is the angle between the rear_arm and the forearm
        theta_3 = abs(acos(( hypotenuse - pow(rear_arm,2) - pow(forearm,2) ) / ( 2 * rear_arm*  forearm)))

        # J3 value to match offset
        angles[2] = theta_3 - radians(90) 

        dobot_angles = [ degrees(angles[0]), degrees(angles[1]), degrees(angles[2]) + degrees(angles[1]), degrees(angles[3])] # in Dobot convention
    except:
        return False

    return dobot_angles


values = calc_inv_kin(100,-50,80,0)

print(values)