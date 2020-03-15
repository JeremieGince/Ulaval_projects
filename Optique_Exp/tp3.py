import sympy as sy
import numpy as np


if __name__ == '__main__':
    ratio = 0.5
    theta_max = np.arccos(np.sqrt(1-ratio))

    f_3 = sy.symbols("f_3")

    y_0 = 0  # [mm]
    theta_0 = theta_max  # [rad]

    f_2 = 200  # [mm]
    f_obj = 5  # [mm]

    y_obj = -0.5  # [mm]
    theta_obj = 0  # [rad]

    ray_0 = sy.Matrix([
        [y_0],
        [theta_0]
    ])

    sys_2f = sy.Matrix([
        [0, f_3],
        [-1/f_3, 0]
    ])

    sys_4f = sy.Matrix([
        [-f_obj/f_2, 0],
        [0, -f_2/f_obj]
    ])

    ray_obj = sy.Matrix([
        [y_obj],
        [theta_obj]
    ])

    equation_to_solve = ray_obj - sys_4f * sys_2f * ray_0  # ray_obj = sys_4f * sys_2f * ray_0
    solved = sy.solve(equation_to_solve, f_3)

    NA_lampe = np.sin(theta_max)
    print(f"ratio: {ratio}, theta_max = {theta_max:.3f} rad, NA_lampe = {NA_lampe:.3f} mm")
    print(f"f_3 = {solved[f_3]:.3f} mm, D_3 = {2*solved[f_3]*NA_lampe:.3f} mm")

    f_tube = 200  # [mm]
    D_tube = 50  # [mm]

    D_camera = 13.312  # [mm]
    theta_0 = (D_tube-D_camera)/2

    sys_4f = sy.Matrix([
        [-f_obj / f_tube, 0],
        [0, -f_tube / f_obj]
    ])

    ray_0 = sy.Matrix([
        [D_camera/2],
        [0]
    ])

    r_0_p = sy.Matrix([
        [-D_camera/2],
        [0]
    ])

    print(f"{sy.latex(sys_4f*ray_0, 'inline')} &= {sy.latex(sys_4f, 'inline')} \cdot {sy.latex(ray_0, 'inline')}")
    print(f"{sy.latex(sys_4f * r_0_p, 'inline')} &= {sy.latex(sys_4f, 'inline')} \cdot {sy.latex(r_0_p, 'inline')}")
    print(abs(-0.1664-0.1664)**2)
    print(6e-6*f_obj / f_tube)


