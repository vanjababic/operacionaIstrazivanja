# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import math

#%%

def balansiranje(ponuda, potraznja, cene, penali = None):
    ponudaSum = sum(ponuda)
    potraznjaSum = sum(potraznja)
    
    if ponudaSum < potraznjaSum:
        if penali is None:
            raise Exception('Neophodni penali.')
        novaPonuda = ponuda + [potraznjaSum - ponudaSum]
        noveCene = cene + [penali]
        return novaPonuda, potraznja, noveCene
    if ponudaSum > potraznjaSum:
        novaPotraznja = potraznja + [ponudaSum - potraznjaSum]
        noveCene = cene + [[0 for _ in potraznja]]
        return ponuda, novaPotraznja, noveCene
    return ponuda, potraznja, cene

cene = [[3, 4],
        [2, 4]]
ponuda = [40, 30]
potraznja = [30, 50]
penali = [3, 1]
ponudaB, potraznjaB, ceneB = balansiranje(ponuda, potraznja, cene, penali)
print(ponudaB, potraznjaB, ceneB)

#%%
    
def severozapadni_ugao(ponuda, potraznja):
    i = 0
    j = 0
    pocetnoResenje = []
    ponudaCopy = ponuda.copy()
    potraznjaCopy = potraznja.copy()
    while len(pocetnoResenje) < len(ponuda) + len(potraznja) - 1:
        pon = ponudaCopy[i]
        potr = potraznjaCopy[j]
        v = min(pon, potr)
        ponudaCopy[i] -= v
        potraznjaCopy[j] -= v
        pocetnoResenje.append(((i, j), v))
        if ponudaCopy[i] == 0 and i < len(ponuda) - 1:
            i += 1
        elif potraznjaCopy[j] == 0 and j < len(potraznja) - 1: 
            j += 1
    return pocetnoResenje

cene = [[3, 4],
        [2, 4],
        [3, 1]]
ponuda = [40, 30, 10]
potraznja = [30, 50]
pocetnoResenje = severozapadni_ugao(ponuda, potraznja)
print(pocetnoResenje)

#%%

def mincena(ponuda, potraznja, cene):
    i = 0
    j = 0
    ponudaCopy = ponuda.copy()
    potraznjaCopy = potraznja.copy()
    pocetnoResenje=[]
    ceneCopy = cene.copy()
    
    while len(pocetnoResenje) != len(ponuda) * len(potraznja):
        minimum = np.min(ceneCopy, axis=None)
        for ii, red in enumerate(ceneCopy):
            for jj, cena in enumerate(red):
                if cena == minimum:
                    i,j = ii, jj
        
        minVrednost = min(ponudaCopy[i], potraznjaCopy[j])
        pocetnoResenje.append(((i, j), minVrednost))
        ponudaCopy[i] -= minVrednost
        potraznjaCopy[j] -= minVrednost
        
        ceneCopy[i][j] = math.inf
    
    
    pocetnoResenjeA = []
    for p, v in enumerate(pocetnoResenje):
        if(v[1] != 0):
            pocetnoResenjeA.append(v)
            
    return pocetnoResenjeA

cene = [[3, 4],
        [2, 4],
        [3, 1]]
ponuda = [40, 30, 10]
potraznja = [30, 50]
pocetnoResenje = mincena(ponuda, potraznja, cene)
print(pocetnoResenje)

#%%vogel

def redPenali(cene, potraznja):
    cene1 = np.transpose(cene)  
    cene1 = [sorted(i) for i in cene1] 
    red_penal=[] 
    for i in range(len(cene1)):
            if(potraznja[i] != 0):
                red_penal.append(cene1[i][1] - cene1[i][0])
            else:
                red_penal.append(0)
    return red_penal

def kolonaPenali(cene, ponuda):

    cene1=[sorted(i) for i in cene]
    kolona_penal=[]
    for i in range(len(cene1)):
            if(ponuda[i] != 0):
                kolona_penal.append(cene1[i][1]-cene1[i][0])
            else:
                kolona_penal.append(0)
    return kolona_penal

def nadjiIndeks(kolona_penal, red_penal):
    max_red = max(red_penal)
    max_kolona = max(kolona_penal)
    red=False
    if(max_red > max_kolona):
        index_max = red_penal.index(max_red)   
        red=True
    else:
        index_max=kolona_penal.index(max_kolona)        
    return index_max,red
    
def Uzmi_minimum_u_matrici_red(index_max, ponuda, potraznja, red, cene):
    ponuda_kopija = ponuda.copy()
    potraznja_kopija = potraznja.copy()
    i=0
    minimalni_index = i
    if(red == True):
        l = []
        for ind1 in range(0, len(cene)):
           l.append(cene[ind1][index_max])
        lm = min(l)

        minimalni_index = l.index(min(l))
        return minimalni_index,index_max
    else:
        print(cene[index_max])
        lm=min(cene[index_max])
        l=cene[index_max]
        minimalni_index = l.index(lm)
       
        return index_max, minimalni_index


def nadjiPocetnoResenje(pocetnoResenje, cene, ponuda, potraznja, i, j, red):
   
    if(ponuda[i] > potraznja[j]):
        pocetnoResenje[i][j]=potraznja[j]
        ponuda[i]=ponuda[i]-potraznja[j]
        potraznja[j]=0
        for ind in range(0,len(ponuda)):
          cene[ind][j]=99999

    else:
        pocetnoResenje[i][j]=ponuda[i]
        potraznja[j]=potraznja[j]-ponuda[i]
        ponuda[i]=0
        for ind in range(0, len(ponuda)):
            cene[i][ind] = 99999    
    return pocetnoResenje,cene

cene = [[8,3,5,5,4],
        [2,5,3,6,8],
        [4,2,8,3,6],
        [3,6,9,5,3]]
ponuda = [65,50,40,45]
potraznja = [60,50,30,20,40]
pocetnoResenje=np.zeros((4,5))


while(max(potraznja)!=0):

    potraznjaA = [i for i, v in enumerate(potraznja) if v == 0]
    print(potraznjaA)
    if len(potraznjaA)==len(potraznja)-1:
        # Ovdje je dio kada ostane ne kraju jedna kolona
        potraznja0 = [i for i, element in enumerate(potraznja) if element != 0]

        red=True
        index_max=potraznja0[0]
        for k in range(len(ponuda)):
            pocetnoResenje[k][index_max]+=ponuda[k]
        potraznja[index_max]=0


    else:
        kolona_penal = kolonaPenali(cene,ponuda)
        red_penal = redPenali(cene,potraznja)
        index_max,red = nadjiIndeks(kolona_penal,red_penal)
        i,j=Uzmi_minimum_u_matrici_red(index_max,ponuda,potraznja,red, cene)

        pocetnoResenje,cene=nadjiPocetnoResenje(pocetnoResenje,cene,ponuda,potraznja,i,j,red)
  
    
print(pocetnoResenje)

p=[]
for i, red in enumerate(pocetnoResenje):
        for j, vrednost in enumerate(red):
            if(vrednost!=0):
                p.append(((i,j),vrednost))
                
#%%
    
def potencijali(pocetnoResenje, cene):
    u = [None] * len(cene) 
    v = [None] * len(cene[0]) 
    u[0] = 0
    cena=0
    pocetnoResenjeCopy = pocetnoResenje.copy()
    while len(pocetnoResenjeCopy) > 0:
        for indeksi, resenje in enumerate(pocetnoResenjeCopy):
            i, j = resenje[0]
            if u[i] is None and v[j] is None: continue
        
            cena = cene[i][j]
            if u[i] is None:
                u[i] = cena - v[j]
            else: 
                v[j] = cena - u[i]
            pocetnoResenjeCopy.pop(indeksi)
            break
    return u, v

cene = [[3, 4],
        [2, 4],
        [3, 1]]
ponuda = [40, 30, 10]
potraznja = [30, 50]
pocetnoResenje = severozapadni_ugao(ponuda, potraznja)
u,v = potencijali(pocetnoResenje, cene)
print(u,v)

#%%

def tezine(pocetnoResenje, cene, u, v):
    tezine = []
    for i, red in enumerate(cene):
        for j, cena in enumerate(red):
            nijeBazno = all([index[0] != i or index[1] != j for index,vrednost in pocetnoResenje])
            if nijeBazno:
                tezine.append(((i, j), cena - u[i] - v[j]))
    return tezine

cene = [[3, 4],
        [2, 4],
        [3, 1]]
ponuda = [40, 30, 10]
potraznja = [30, 50]
pocetnoResenje = severozapadni_ugao(ponuda, potraznja)
u,v = potencijali(pocetnoResenje, cene)
t = tezine(pocetnoResenje, cene, u, v)
print(t)

#%%
    
def optimizovati(tezine):
    for index, vrednost in tezine:
        if vrednost < 0: 
            return True
    return False

cene = [[3, 4],
        [2, 4],
        [3, 1]]
ponuda = [40, 30, 10]
potraznja = [30, 50]
pocetnoResenje = severozapadni_ugao(ponuda, potraznja)
u,v = potencijali(pocetnoResenje, cene)
t = tezine(pocetnoResenje, cene, u, v)
o = optimizovati(t)
print(o)

#%%

def pocetnoPolje(tezine):
    minVrednosti = []
    for index, vrednost in tezine:
        minVrednosti.append(vrednost)
    minVrednost = min(minVrednosti)
    for index, vrednost in tezine:
        if(vrednost == minVrednost):
            return index
cene = [[3, 4],
        [2, 4],
        [3, 1]]
ponuda = [40, 30, 10]
potraznja = [30, 50]
pocetnoResenje = severozapadni_ugao(ponuda, potraznja)
u,v = potencijali(pocetnoResenje, cene)
t = tezine(pocetnoResenje, cene, u, v)
najnegativnije = pocetnoPolje(t)
print(najnegativnije)

#%%
    
def potencijalniCvorovi(petlja, nijePosecen):
    poslednjiCvor = petlja[-1]
    cvoroviURedu = [n for n in nijePosecen if n[0] == poslednjiCvor[0]]
    cvoroviUKoloni = [n for n in nijePosecen if n[1] == poslednjiCvor[1]] 
    if len(petlja) < 2:
        return cvoroviURedu + cvoroviUKoloni
    else:
        prepre_cvor = petlja[-2] 
        potez_red = prepre_cvor[0] == poslednjiCvor[0]
        if potez_red: 
            return cvoroviUKoloni
        return cvoroviURedu
    
    
    
def pronadjiPetlju(pocetniIndexi, pocetnoPolje, petlja):
    if len(petlja) > 3:
        kraj = len(potencijalniCvorovi(petlja, [pocetnoPolje])) == 1 
        if kraj: 
            return petlja
        
    nisuPoseceni = list(set(pocetniIndexi) - set(petlja))
    moguciCvorovi = potencijalniCvorovi(petlja, nisuPoseceni)
    for sledeci_cvor in moguciCvorovi: 
        sledecaPetlja = pronadjiPetlju(pocetniIndexi, pocetnoPolje, petlja + [sledeci_cvor])
        if sledecaPetlja: 
            return sledecaPetlja

cene = [[3, 4],
        [2, 4],
        [3, 1]]
ponuda = [40, 30, 10]
potraznja = [30, 50]
pocetnoResenje = severozapadni_ugao(ponuda, potraznja)
u,v = potencijali(pocetnoResenje, cene)
t = tezine(pocetnoResenje, cene, u, v)
najnegativnije = pocetnoPolje(t)
petlja = pronadjiPetlju([p for p,v in pocetnoResenje], najnegativnije, [najnegativnije])
print(petlja)

#%%

def novoResenje(pocetnoResenje, petlja):
    plus = petlja[0::2]
    minus = petlja[1::2]
    get_bv = lambda pos: next(v for p, v in pocetnoResenje if p == pos)
    minIndeks = sorted(minus, key=get_bv)[0]
    minVrednost = get_bv(minIndeks)
    
    novoResenje = []
    for p, v in [p for p in pocetnoResenje if p[0] != minIndeks] + [(petlja[0], 0)]:
        if p in plus:
            v += minVrednost
        elif p in minus:
            v -= minVrednost
        novoResenje.append((p, v))
        
    return novoResenje

cene = [[3, 4],
        [2, 4],
        [3, 1]]
ponuda = [40, 30, 10]
potraznja = [30, 50]
pocetnoResenje = severozapadni_ugao(ponuda, potraznja)
u,v = potencijali(pocetnoResenje, cene)
t = tezine(pocetnoResenje, cene, u, v)
najnegativnije = pocetnoPolje(t)
petlja = pronadjiPetlju([p for p,v in pocetnoResenje], najnegativnije, [najnegativnije])
novo = novoResenje(pocetnoResenje, petlja)
print(novo)

#%%

def transportniProblem(ponuda, potraznja, cene, penali):
    balansiranaPonuda, balansiranaPotraznja, balansiraneCene = (balansiranje(ponuda, potraznja, cene, penali))
    pocetnoResenje = severozapadni_ugao(balansiranaPonuda, balansiranaPotraznja)
    optimalnoResenje = optimalno(pocetnoResenje, balansiraneCene)

    matrica = np.zeros((len(balansiraneCene), len(balansiraneCene[0])))
    for (i, j), v in optimalnoResenje:
        matrica[i][j] = v
    trosakTransporta = odrediTrosak(balansiraneCene, matrica)
    return matrica, trosakTransporta

def optimalno(pocetnoResenje, cene):
    u,v = potencijali(pocetnoResenje, cene)
    cij = tezine(pocetnoResenje, cene, u, v)

    if(optimizovati(cij)):
        najnegativnije = pocetnoPolje(cij)
        petlja = pronadjiPetlju([p for p,v in pocetnoResenje], najnegativnije, [najnegativnije])
        return optimalno(novoResenje(pocetnoResenje, petlja), cene)
    return pocetnoResenje

def odrediTrosak(cene, resenje):
    suma = 0
    for i, red in enumerate(cene):
        for j, cena in enumerate(red):
            suma += cena*resenje[i][j]
    return suma

cene = [[3, 4],
        [2, 4]]
ponuda = [40, 30]
potraznja = [30, 50]
penali = [3, 1]
transport, cena = transportniProblem(ponuda, potraznja, cene, penali)
print('\n\nMatrica:\n',transport,'\n\nCena transporta:' ,cena)

#%%

cene = [
    [ 10, 12, 0],
    [8, 4, 3],
    [6, 9, 4],
    [7, 8, 5]]
ponuda = [20, 30, 20, 10]
potraznja = [10, 40, 10]

penali=[]
transport, cena = transportniProblem(ponuda, potraznja, cene, penali)
print('\n\nMatrica:\n',transport,'\n\nCena transporta:' ,cena)

#%%
