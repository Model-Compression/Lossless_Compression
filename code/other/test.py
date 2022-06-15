import math

import matplotlib.pyplot as plt
import numpy as np
from utils.expect_calculate import expect_calcu


def custome_activation_analysis(name):
    if name == 'binary_last_final':
        zero_order = lambda s1, s2, b1, b2, b3, tao_last: (b1 * (math.erf(s1 / (tao_last * pow(
            2, 1 / 2))) + 1) + b3 * math.erfc(s2 / (tao_last * pow(2, 1 / 2)))) / 2 + b2 * (math.erf(s2 / (tao_last * pow(
                2, 1 / 2))) - math.erf(s1 / (tao_last * pow(2, 1 / 2)))) / 2

        first_order = lambda s1, s2, b1, b2, b3, tao_last: (b3 * math.exp(-pow(s2 / tao_last, 2) / 2) -
        b1 * math.exp(-pow(s1/tao_last , 2) / 2) + \
            b2 * (math.exp(-pow(s1 / tao_last, 2) / 2) - math.exp(-pow(s2/tao_last , 2) / 2) ) ) \
                / (tao_last * pow(2 * math.pi,1/2))

        second_order = lambda s1, s2, b1, b2, b3, tao_last: (s2 * b3 * math.exp(-pow(s2/tao_last, 2) / 2) -
        s1 * b1 * math.exp(-pow(s1/tao_last , 2) / 2) + \
            b2 * (s1 * math.exp(-pow(s1 / tao_last, 2) / 2) - s2 * math.exp(-pow(s2/tao_last , 2) / 2) ) \
        )/(pow(tao_last,3) * pow(2 *math.pi,1/2))

        square_second_order = lambda s1, s2,  b1, b2, b3, tao_last: (s2 * b3**2 * math.exp(-pow(s2/tao_last, 2) / 2) -
        s1 * b1**2 * math.exp(-pow(s1/tao_last , 2) / 2) + \
            b2**2 * (s1 * math.exp(-pow(s1 / tao_last, 2) / 2) - s2 * math.exp(-pow(s2/tao_last , 2) / 2) ) \
        )/(pow(tao_last,3) * pow(2 *math.pi,1/2)) - 2 * zero_order(s1, s2, b1, b2, b3, tao_last) * second_order(s1, s2, b1, b2, b3, tao_last)

        tau = lambda s1, s2, b1, b2, b3,  tao_last: (b1**2  * (math.erf(s1 / (tao_last * pow(2 , 1/2)) )+ 1) + b3**2  * math.erfc(s2 / (tao_last * pow(2, 1/2)))) /2 + \
                                                    b2**2 * (math.erf((s2 / tao_last) / math.sqrt(2)) - math.erf( (s1/tao_last) / math.sqrt(2)))/2 - \
                                                    (zero_order(s1, s2, b1, b2, b3, tao_last))**2

        return zero_order, first_order, second_order, square_second_order, tau

    elif name == 'binary_last_four':
        zero_order = lambda s1, s2, s3, s4, b1, b2, tao_last: b1/2 * (math.erf(s1/(pow(2, 1/2) * tao_last)) - math.erf(s4/(pow(2, 1/2) * tao_last))) + b1  + \
                                                        b2/2 * (math.erf(s3/(pow(2, 1/2) * tao_last)) - math.erf(s2/(pow(2, 1/2) * tao_last)))

        first_order = lambda s1, s2, s3, s4, b1, b2, tao_last: b1 * (math.exp(-pow(s4/tao_last, 2)/2) - math.exp(-pow(s1/tao_last, 2)/2)) / (pow(2*math.pi,1/2) * tao_last) + \
                                                    b2 * (math.exp(-pow(s2/tao_last, 2)/2) - math.exp(-pow(s3/tao_last, 2)/2)) / (pow(2*math.pi,1/2) * tao_last)

        second_order = lambda s1, s2, s3, s4, b1, b2, tao_last: b1 * (s4*math.exp(-pow(s4/tao_last, 2)/2) - s1*math.exp(-pow(s1/tao_last, 2)/2)) / (pow(2*math.pi,1/2) * pow(tao_last,3)) + \
                                                    b2 * (s2*math.exp(-pow(s2/tao_last, 2)/2) - s3*math.exp(-pow(s3/tao_last, 2)/2)) / (pow(2*math.pi,1/2) * pow(tao_last,3))

        square_second_order = lambda s1, s2, s3, s4, b1, b2, tao_last: b1**2 * (s4*math.exp(-pow(s4/tao_last, 2)/2) - s1*math.exp(-pow(s1/tao_last, 2)/2)) / (pow(2*math.pi,1/2) * pow(tao_last,3)) + \
                                                    b2**2 * (s2*math.exp(-pow(s2/tao_last, 2)/2) - s3*math.exp(-pow(s3/tao_last, 2)/2)) / (pow(2*math.pi,1/2) * pow(tao_last,3)) - \
                                                    2 * zero_order(s1, s2, s3, s4, b1, b2, tao_last) * second_order(s1, s2,s3, s4, b1, b2, tao_last)

        tau = lambda s1, s2, s3, s4, b1, b2, tao_last: b1**2 / 2 * (math.erf(s1/(pow(2, 1/2) * tao_last)) - math.erf(s4/(pow(2, 1/2) * tao_last))) + b1  + \
                                                        b2**2 / 2 * (math.erf(s3/(pow(2, 1/2) * tao_last)) - math.erf(s2/(pow(2, 1/2) * tao_last))) - \
                                                            (zero_order(s1, s2, s3, s4, b1, b2, tao_last))**2
        # return zero_order, first_order, second_order, square_second_order, tau

        return zero_order, first_order, second_order, square_second_order, tau


if __name__ == "__main__":
    # s1, s2, b1 , b2, tau= -1.16799936, -1.35270477, -5.66409632, -3.17210397, 0.67725275
    # a, b , c, d, e = expect_calcu('Binary_Last', s1=s1, s2=s2, b1=b1, b2=b2)
    # print(a(tau)[0], b(tau)[0], c(tau)[0], d(tau)[0], e(tau)[0], '\n')
    # # )
    # s1, s2, b1, b2, b3, tao_last = 1, 0, 2, 0, 5,  0.6
    # s1, s2, s3, s4, b1 , b2, tao_last = -0.43, -0.43, 1.14, 1.14, -0.3, 1.13, 0.6657
    # # s1, s2, b1 , b2, tau= -1.16799936, -1.35270477, -5.66409632, -3.17210397, 0.67725275
    # a11, b11, c11, d11, e11 = custome_activation_analysis('binary_last_four')
    # print(a11(s1, s2, s3, s4, b1, b2 , tao_last), b11(s1, s2, s3, s4, b1, b2 , tao_last)**2, c11(s1, s2, s3, s4, b1, b2 , tao_last)**2, d11(s1, s2, s3, s4, b1, b2 , tao_last), e11(s1, s2, s3, s4, b1, b2 , tao_last))
    # )
    # plt.figure(1)
    # plt.plot((1,1,1), (2,4,5))
    # # plt.show())

    # # plt.hold(True)
    # plt.figure(2)
    # plt.plot((1,6,5), (2,4,5))
    # plt.show()
    s1, s2, s3, s4, b1, b2, tao_last = 1, 2, 3, 4, 5, 6, 8
    zero_order = lambda s1, s2, s3, s4, b1, b2, tao_last: (b1/2) * (math.erf(s1/(pow(2, 1/2) * tao_last)) - math.erf(s4/(pow(2, 1/2) * tao_last))) + b1  + \
                                                        (b2/2) * (math.erf(s3/(pow(2, 1/2) * tao_last)) - math.erf(s2/(pow(2, 1/2) * tao_last)))

    tau = lambda s1, s2, s3, s4, b1, b2, tao_last: (b1**2 / 2) * (math.erf(s1/(pow(2, 1/2) * tao_last)) - math.erf(s4/(pow(2, 1/2) * tao_last))) + b1**2  + \
                                                    (b2**2 / 2) * (math.erf(s3/(pow(2, 1/2) * tao_last)) - math.erf(s2/(pow(2, 1/2) * tao_last))) - \
                                                        (zero_order(s1, s2, s3, s4, b1, b2, tao_last))**2
    print(tau(s1, s2, s3, s4, b1, b2, tao_last), zero_order(s1, s2, s3, s4, b1, b2, tao_last),
          (zero_order(s1, s2, s3, s4, b1, b2, tao_last) - b1) * (zero_order(s1, s2, s3, s4, b1, b2, tao_last) - b2))
