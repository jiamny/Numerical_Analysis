
print('------------------------------------------------------')
print('Decimal to binary')
print('------------------------------------------------------')
def toBinary(d, stop = 52):
    BinaryIntPart = []
    BinaryFractPart = []

    if isinstance(d, float):
        s = str(d)
        Int_part, _ = s.split('.')

        Int_part = int(Int_part)

        if Int_part > 0:
            BinaryIntPart = []
            R = Int_part % 2
            Rem = Int_part // 2
            BinaryIntPart.append(R)

            while Rem > 1:
                R = Rem % 2
                Rem = Rem // 2
                BinaryIntPart.append(R)
            BinaryIntPart.append(Rem)

            BinaryIntPart = list(reversed(BinaryIntPart))

        Fract_part = d - Int_part

        if Fract_part > 0:
            BinaryFractPart = []
            R = 0
            if Fract_part * 2 >= 1.:
                Rem = Fract_part * 2 - 1.
                R = 1
            else:
                Rem = Fract_part * 2

            BinaryFractPart.append(R)
            count = 0
            while Rem > 0.:
                R = 0
                if Rem * 2 >= 1.:
                    Rem = Rem * 2 - 1.
                    R = 1
                else:
                    Rem = Rem * 2

                BinaryFractPart.append(R)
                count += 1
                if count > stop:
                    break
    return ''.join(map(str, BinaryIntPart)) + '.' + ''.join(map(str, BinaryFractPart))


d = 512.5
bstr = toBinary(d)
print('toBinary(d) = ', bstr)

print('------------------------------------------------------')
print('Binary to decimal')
print('------------------------------------------------------')
def toDecimal(b):
    Int_part, Fract_part = bstr.split('.')

    Int_P = list(map(int, [x for x in Int_part]))
    Fract_P = list(map(int, [x for x in Fract_part]))

    tInt = 0
    ln = len(Int_P)
    for index in list(reversed(range(len(Int_P)))):
        idx = ln - index - 1
        tInt += Int_P[idx] * 2 ** index

    tFact = 0
    ln = len(Fract_P)
    for idx in range(len(Fract_P)):
        tFact += Fract_P[idx] * 0.5 ** (idx + 1)

    return (tInt + tFact)

print('toDecimal(bstr) = ', toDecimal(bstr))

d = 53.7
print( "d = ", d)
bstr = toBinary(d)
print('toBinary(d) = ', bstr)
print('toDecimal(bstr) = ', toDecimal(bstr))