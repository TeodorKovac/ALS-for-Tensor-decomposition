import numpy as np
import matplotlib.pyplot as plt


def nasob(n):
    # Metoda generující Tenzor reprezentující maticové násobení dvou matic o rozměrech n x n
    # INPUT:
    #           n ... přirozené číslo
    # OUTPUT:
    #           tenzor o rozměrech n^2 x n^2 x n^2
    t = np.zeros([n ** 2, n ** 2, n ** 2])
    for i in range(1, n+1):
        ii = (i - 1) * n
        for j in range(1, n+1):
            ij = ii + j
            a = np.zeros([n, n])
            a[i - 1, j - 1] = 1
            for k in range(1, n+1):
                kk = (k - 1) * n
                for l in range(1, n+1):
                    b = np.zeros([n, n])
                    b[k - 1, l - 1] = 1
                    c = a.dot(b)
                    c = np.reshape(c, n ** 2, order='F')
                    for r in range(1, n ** 2+1):
                        t[ij-1, kk + l-1, r - 1] = c[r - 1]
    return t

######## KRB ######
def krb(A,B):
    # Funkce Khatri-Rao produkt dvou matic A,B
    # INPUT:
    #           A ... vstupní matice n x r
    #           B ... vstupní matice m x r
    # OUTPUT:
    #           matice n*m x r
    rowsA = len(A) # Počet řádků
    rowsB = len(B)
    columnsA = len(A[0]) # Počet sloupců
    columnsB = len(B[0])
    if columnsA != columnsB: # Případ špatně zadaných argumentů
        raise Exception("Error - matice musi mit shodny pocet sloupcu")
    nasobek = rowsA * rowsB
    ab = np.zeros([nasobek, columnsA])
    for i in range(0, columnsA):
        meziMatice = np.zeros([rowsB, rowsA])
        for j in range(0, rowsA):
            for k in range(0, rowsB):
                meziMatice[k][j] = A[j][i]*B[k][i]
        ab[:, i] = meziMatice.flatten('F')
    return ab


def JakobA(A, B, C):
    # Funkce pro výpočet jakobiánu účelové funkce podle faktorové matice A
    # INPUT:
    #       A,B,C ... faktorové matice
    # OUTPUT:
    #       Jakobiho matice účelové funkce podle faktorové matice A
    Ia1 = len(A)
    Ja = np.kron(krb(C, B), np.identity(Ia1))
    return Ja

def JakobB(A, B, C):
    # Funkce pro výpočet jakobiánu účelové funkce podle faktorové matice B
    # INPUT:
    #       A,B,C ... faktorové matice
    # OUTPUT:
    #       Jakobiho matice účelové funkce podle faktorové matice B
    Ia1 = len(A)
    Ib1 = len(B)
    Ib2 = len(B[0])
    Ic1 = len(C)
    Jb = np.ones(((Ia1*Ib1*Ic1), Ib1*Ib2))
    for i in range(1, Ib2+1):
        a = (i-1) * Ib1
        b = i * Ib1
        Jb[:, a:b] = (np.kron(C[:, (i-1)], (np.kron(np.identity(Ib1), A[:, (i-1)])))).transpose()
    return Jb

def JakobC(A, B, C):
    # Funkce pro výpočet jakobiánu účelové funkce podle faktorové matice C
    # INPUT:
    #       A,B,C ... faktorové matice
    # OUTPUT:
    #       Jakobiho matice účelové funkce podle faktorové matice C
    Ia1 = len(A)
    Ib1 = len(B)
    Ic2 = len(C[0])
    Ic1 = len(C)
    Jc = np.ones(((Ia1 * Ib1 * Ic1), Ic1 * Ic2))
    for i in range(1, Ic2+1):
        a = (i-1) * Ic1
        b = i * Ic1
        Jc[:, a:b] = (np.kron(np.identity(Ic1), np.kron(B[:, (i-1)], A[:, (i-1)]))).transpose()
    return Jc

def ALS(X, rank, numit):
    # Alternating Least Squares algoritmus pro hledání CPD rozkladu tenzoru třetího řádu
    # INPUT:
    #           X ... tenzor k hledání CPD rozkladu
    #           rank ... požadovaná hodnost kanonického rozkladu
    #           mnumit ... počet iterací algoritmu
    # OUTPUT:
    #           [A, B, C, error] ... faktorové matice [[A,B,C]], chyba aproximace
    Ia = len(X[:, 1, 1]) # Rozměry zadaného tenzoru
    Ib = len(X[1, :, 1])
    Ic = len(X[1, 1, :])
    X = np.reshape(X, (Ia, Ib * Ic), order='F') # Matrizace zadaného tenzoru
    A = np.random.rand(Ia, rank) # Inicializace náhodnými čísly
    B = np.random.rand(Ib, rank)
    C = np.random.rand(Ic, rank)
    error = np.zeros(numit-1)
    for it in range(1, numit):
        # UPDATE MATICE A
        Ja = JakobA(A, B, C)
        invJ = np.linalg.inv(np.matmul((Ja.transpose()), Ja))
        theta = np.matmul(invJ, (np.matmul(Ja.transpose(), X.flatten('F'))))
        A = np.reshape(theta, (Ia, rank), order='F')
        # UPDATE MATICE B
        Jb = JakobB(A, B, C)
        invJ = np.linalg.inv(np.matmul((Jb.transpose()), Jb))
        theta = np.matmul(invJ, (np.matmul(Jb.transpose(), X.flatten('F'))))
        B = np.reshape(theta, (Ib, rank), order='F')
        # UPDATE MATICE C
        Jc = JakobC(A, B, C)
        invJ = np.linalg.inv(np.matmul((Jc.transpose()), Jc))
        theta = np.matmul(invJ, (np.matmul(Jc.transpose(), X.flatten('F'))))
        C = np.reshape(theta, (Ic, rank), order='F')
        error[it-1] = chyba(X, A, B, C)

        print(error[it-1])
    return [A, B, C, error]

def chyba(X, A, B, C):
    # Funkce pro výpočet chyby aproximace, jako frobeniova norma rozdílu
    # INPUT:
    #           X ... rozkládaný tenzor
    #           A,B,C ... faktorové matice rozkladu
    # OUTPUT:
    #           chyba aproximace kanonickým rozkladem
    Y = X - (np.matmul(A, (krb(C, B)).transpose()))
    err = np.linalg.norm(Y, 'fro')
    return err
if __name__ == "__main__":
# Příklad s tenzorem 3.řádu reprezentující násobení matic dvou matic 3x3
    X = nasob(3)
    Hodnost = 25
    numit = 50
    Vysledek = ALS(X, Hodnost, numit)
    print('CPD rozklad s chybou: \n')
    error = Vysledek[3]
    plt.plot(error)
    plt.ylabel('chyba rozkladu')
    plt.xlabel('iterace')
    plt.title('Graf chyby aproximace')
    plt.show()
    print(Vysledek[3])
else:
    print("ALS bylo naimportovano")


