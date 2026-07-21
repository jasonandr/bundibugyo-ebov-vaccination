import ebola_stochastic_ring_cpp

try:
    ebola_stochastic_ring_cpp.simulate_ring_vaccination_cpp(
        1000, [[1, 2], [0], [0]], [], 0.08,
        8.5, 6.0,
        1, 0.3, 10.0, 0.8,
        [], 1.0,
        4.0, 2.0, -1,
        100, -1,
        0.454, 0.227, 5, 0,
        300, 0.0, False,
        1.0, False,
        [], 
        -1.0,
        -1.0, -1.0,
        0.75, 2.0, False, True, 0.5, 1, 0.0, -1
    )
    print("SUCCESS")
except Exception as e:
    print(e)
