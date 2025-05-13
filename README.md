# Projekt: Pridobivanje črk in značilk iz ročno napisanih tabel

## Namen naloge

Namen projekta je razviti Python skripto, ki samodejno obdeluje slike ročno napisanih črk v tabelah. Cilji naloge so:
- zaznati in izrezati posamezne črke iz slik (vseh vrst: velike/male tiskane in pisane),
- klasificirati tip črk (velika/mala, tiskana/pisana),
- za vsako črko izračunati značilke s pomočjo razdelitve slike na kvadrante,
- vse rezultate shraniti v CSV datoteko, pripravljeno za strojno učenje.

---

## Funkcionalnosti

- **Izrez posameznih črk** iz tabele s pomočjo zaznave mreže (OpenCV).
- **Klasifikacija tipa črk** na podlagi velikosti in kompleksnosti (št. kontur).
- **Shranjevanje izrezanih črk** v ločene mape glede na tip pisave.
- **Pridobivanje značilk** (štetje temnih pikslov in razmerje) za vsako črko po kvadrantih.
- **Shranjevanje značilk** v `.csv` datoteko za nadaljnjo analizo ali strojno učenje.

## Zahteve
Za uspešno izvajanje skript potrebujete:
- **Python 3.x**
- **OpenCV** – za obdelavo slik  
- **NumPy** – za delo s številskimi matrikami
- **csv** – za izvoz podatkov (vgrajeno v Python)

## Zagon

Za zagon projekta sledi naslednjim korakom:

### 1. Izreži črke iz tabelnih slik

Zaženi funkcijo, ki pregleda vse podmape (npr. `new abeceda`) in izreže posamezne črke v ustrezne mape:

```python
process_all_from_subfolders("new abeceda", "izhod_abeceda")
```

### 2. (Neobvezno) Izbriši prve vzorce vsake črke (_01)

Če želiš odstraniti začetne primere (npr. A_01, B_01...), lahko uporabiš:

```python
delete_images_ending_with_01("izhod_abeceda")
```

### 3. Izračun značilk in izvoz v CSV

Za vsako izrezano sliko črke izračunaj značilke po kvadrantih (npr. 4×4 razdelitev slike → 16 blokov → 32 značilk: število temnih pikslov + razmerje):

```python
extract_features_from_images("izhod_abeceda", grid_size=4, output_csv="znacilke.csv")
```
- grid_size=4 pomeni, da bo vsaka slika razdeljena na 16 kvadrantov (4 x 4).
- znacilke.csv bo vseboval vse značilke za nadaljnjo obdelavo oz. strojno učenje.

# Projekt: Pridobivanje črk in značilk iz ročno napisanih tabel

## Namen naloge

Namen projekta je razviti Python skripto, ki samodejno obdeluje slike ročno napisanih črk v tabelah. Cilji naloge so:
- zaznati in izrezati posamezne črke iz slik (vseh vrst: velike/male tiskane in pisane),
- klasificirati tip črk (velika/mala, tiskana/pisana),
- za vsako črko izračunati značilke s pomočjo razdelitve slike na kvadrante,
- vse rezultate shraniti v CSV datoteko, pripravljeno za strojno učenje.

---

## Funkcionalnosti

- **Izrez posameznih črk** iz tabele s pomočjo zaznave mreže (OpenCV).
- **Klasifikacija tipa črk** na podlagi velikosti in kompleksnosti (št. kontur).
- **Shranjevanje izrezanih črk** v ločene mape glede na tip pisave.
- **Pridobivanje značilk** (štetje temnih pikslov in razmerje) za vsako črko po kvadrantih.
- **Shranjevanje značilk** v `.csv` datoteko za nadaljnjo analizo ali strojno učenje.

## Zahteve
Za uspešno izvajanje skript potrebujete:
- **Python 3.x**
- **OpenCV** – za obdelavo slik  
- **NumPy** – za delo s številskimi matrikami
- **csv** – za izvoz podatkov (vgrajeno v Python)

## Zagon

Za zagon projekta sledi naslednjim korakom:

### 1. Izreži črke iz tabelnih slik

Zaženi funkcijo, ki pregleda vse podmape (npr. `new abeceda`) in izreže posamezne črke v ustrezne mape:

```python
process_all_from_subfolders("new abeceda", "izhod_abeceda")
```

### 2. (Neobvezno) Izbriši prve vzorce vsake črke (_01)

Če želiš odstraniti začetne primere (npr. A_01, B_01...), lahko uporabiš:

```python
delete_images_ending_with_01("izhod_abeceda")
```

### 3. Izračun značilk in izvoz v CSV

Za vsako izrezano sliko črke izračunaj značilke po kvadrantih (npr. 4×4 razdelitev slike → 16 blokov → 32 značilk: število temnih pikslov + razmerje):

```python
extract_features_from_images("izhod_abeceda", grid_size=4, output_csv="znacilke.csv")
```
- grid_size=4 pomeni, da bo vsaka slika razdeljena na 16 kvadrantov (4 x 4).
- znacilke.csv bo vseboval vse značilke za nadaljnjo obdelavo oz. strojno učenje.

## Oblika CSV datoteke

Vsaka vrstica v datoteki `znacilke.csv` predstavlja eno izrezano črko.

Primer vrstice:
```csv
sc032,mala_pisana,A,02,203,0.79,240,0.94,...,196,0.76
```
Stolpci:
- ime_slike: ime originalne slike (npr. sc032)
- tip_crke: vrsta črke (npr. mala_pisana)
- crka: katera črka je (npr. A)
- stevilka: zaporedna številka primerka črke
- temni_i: število temnih pikslov v kvadrantu i
- razmerje_i: razmerje temnih pikslov v kvadrantu i


