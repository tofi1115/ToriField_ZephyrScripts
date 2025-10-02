def Eliminate_Planes(StateVector_lst):
    for StateVector in StateVector_lst:
        if StateVector.baro_altitude>25000:
            print("Plane Detected")

    print("Eliminate Planes!")