# Authored by Luhuai Jiao
import taichi as ti

ti.init(arch=ti.gpu)
PI = 3.141592653589793
L = 8 * PI
dt = 0.1
substepping = 4
ng = 32
np = 16384
vb = 1.0
vt = 0.3
wp = 1  #Plasma frequence
qm = -1
q = wp * wp / (qm * np / L)
rho_back = -q * np / L  #background charge
dx = L / ng
inv_dx = 1.0 / dx
x = ti.Vector.field(1, ti.f32, np)
v = ti.Vector.field(1, ti.f32, np)
rho = ti.Vector.field(1, ti.f32, ng)
e = ti.Vector.field(1, ti.f32, ng)
v_x_pos1 = ti.Vector.field(2, ti.f32, int(np / 2))  #to plot Vx-X
v_x_pos2 = ti.Vector.field(2, ti.f32, int(np / 2))


@ti.kernel
def initialize():
    for p in x:
        x[p].x = (p + 1) * L / np
        v[p].x = vt * ti.randn() + (-1)**p * vb


@ti.kernel
def substep():
    for p in x:
        x[p] += v[p] * dt
        if x[p].x >= L:
            x[p] += -L
        if x[p].x < 0:
            x[p] += L
    rho.fill(rho_back)
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - 0.5 - base.cast(float)
        rho[base] += (1.0 - fx) * q * inv_dx
        rho[base + 1] += fx * q * inv_dx
    e.fill(0.0)
    ti.loop_config(serialize=True)
    for i in range(ng):
        e[i] = e[i - 1] + (rho[i - 1] + rho[i]) * dx * 0.5
    s = 0.0
    for i in e:
        s += e[i].x
    for i in e:
        e[i] += -s / ng
    for p in v:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - 0.5 - base.cast(float)
        a = (e[base] * (1.0 - fx) + e[base + 1] * fx) * qm
        v[p] += a * dt


@ti.kernel
def vx_pos():
    for p in x:
        if p % 2:
            v_x_pos1[int((p - 1) / 2)].x = x[p].x / L
            v_x_pos1[int((p - 1) / 2)].y = (v[p].x) / 10 + 0.5
        else:
            v_x_pos2[int(p / 2)].x = x[p].x / L
            v_x_pos2[int(p / 2)].y = (v[p].x) / 10 + 0.5


def main():
    initialize()
    gui = ti.GUI("pic88", (800, 800))
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for s in range(substepping):
            substep()
        vx_pos()
        gui.circles(v_x_pos1.to_numpy(), color=0x0000ff, radius=2)
        gui.circles(v_x_pos2.to_numpy(), color=0xff0000, radius=2)
        gui.show()


if __name__ == '__main__':
    main()
