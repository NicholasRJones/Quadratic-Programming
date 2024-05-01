import numpy as np
from Optimization.Algorithm import classy as cl, QP


input = [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, -5, -4, -4, -3, 4]

alpha = 1
beta = 2

G = np.block([[np.diag((alpha + beta) / 2 * np.ones(8)), np.diag(-beta * np.ones(8))], [np.zeros((8, 8)), np.diag(beta / 2 * np.ones(8))]])
G = (G + np.transpose(G)) / 2
parameters = [G, np.zeros(16)]

E = []
gE = []
be = []

B1 = np.array([[-2, 4], [-2, -1], [0, -3], [1, -1], [3, 1]])
B2 = np.array([[2, -2], [2, 3], [-2, 0], [-2, -1]])
B3 = np.array([[-2, 0], [0, -3], [2, 0], [0, 3]])
B4 = np.array([[1, 4], [-3, -3], [2, -1]])
B5 = np.array([[1, -2], [2, 0], [-3, 4], [0, -2]])

I = np.block([[B1, np.zeros((5, 14))], [np.zeros((5, 2)), B1, np.zeros((5, 12))], [np.zeros((5, 4)), B1, np.zeros((5, 10))], [np.zeros((5, 6)), B1, np.zeros((5, 8))], [np.zeros((4, 8)), B2, np.zeros((4, 6))], [np.zeros((4, 10)), B3, np.zeros((4, 4))], [np.zeros((3, 12)), B4, np.zeros((3, 2))], [np.zeros((4, 14)), B5]])
bi = np.array([8, 8, 6, 2, 2, 8, 8, 6, 2, 2, 8, 8, 6, 2, 2, 8, 8, 6, 2, 2, 4, 19, -4, -7, -2, 18, 8, -12, -18, 27, 0, -10, -4, 30, -6])

I = -I
gI = []
bi = -bi


quad = cl.quad(E, gE, be, I, gI, bi)

para = cl.para(0.0001, 0.19, 0, parameters, quad, 0, 0)
pr = cl.funct('', '', '', input, para, 1)

QP.QP(pr)
