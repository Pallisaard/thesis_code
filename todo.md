# Projektplan fremadrettet

## Hvad skal vi nå?

- [ ] Klargjort eksperimentdetaljer for pre-training af VAE (Begge) !!!
- [ ] Vi skal have overblik over hvad hvilke modeller vi har/hvad vi skal bruge (Begge)
  - [x] Undersøg hvordan M3VAE fungerer og hvad deres implementation af beta[ ]VAE gør. (Pallisgaard)
    - [x] Pænt nej tak.
  - [x] Undersøg om der er alternativer til M3VAE-paper der viser resultater på syntetisk data (Jensen)
  - [ ] Vi benytter beta-VAE på 3d med 64, 256 opløsning.
    - [ ] Vi benytter nogle GANs, e.g. alpha WGAN-GP fra Kwon et al, HAGAN har en pre-trained model, den snupper vi.
  - [ ] Hvilke parametre vælger vi og hvilke træningsmetoder benytter vi.
  - [ ] Hvilke målinger skal vi lave undervejs.
  - [ ] en klargørelse for modelvalg.

- [ ] Pre-training af VAE (Begge)
  - [ ] Vi skal have træningsscriptet gjort klar så vi kan bruge det på en vilkårlig model (Er gjort)
  - [ ] Vi skal have gjort VAE klar til træning
    - [ ] Venter på klargørelse
  - [ ] Vi skal køre tests for at finde optimale GPU indstillinger og hypeparametre (antal GPUer, batch size, modelstørrelse)
  - [ ] Vi skal vælge tidsremme (epochs, uger)
  - [ ] Vi skal løbende holde øje med hvordan modellen træner

- [ ] Diskuteret og implementeret metrikker (Begge)
  - [ ] Vi skal have diskuteret pros/cons for SSIM
    - [ ] Hvad ønsker vi at måle med SSIM?
  - [ ] Vi skal have diskuteret pros/cons for FastSurfer behandling
    - [ ] Skal vi overhovedet gøre det her?
    - [ ] Skal vi måle varians over hver vektorkoordinat eller på alle vektorer rent.

- [ ] Fine-tuning af VAE (Jensen)
  - [ ] Vi skal have en opskrivning af hvordan fine tuning processen foregår.
  - [ ] Vi skal have udvalgt struktur for hvordan man gør sådan et forsøg.
  - [ ] Vi skal have klare definitioner af mål og metode.
  - [ ] Vi skal teste fine-tuningsprocessen og sikre os at den kører korrekt, træner og tilføjer støj på korrekt vis.
  - [ ] Vi skal eksperimentere med GPU setup så modellen kører bedst.
  - [ ] Vi skal igangsætte fine-tuning.
  - [ ] Vi skal løbende observere modellens træning så vi kan identificere om der er fejl / om forsøget skal gentages med andre parametre.

- [ ] Fine-tuning data (Pallisgaard)
  - [ ] Vi skal have gennemgået ADNI for at finde en masse eksempler på Alzheimers som vi kan bruge til fine tunene
  - [ ] Den data skal efterfølgende igennem vores data pipeline.
  - [ ] Den skal placeres så den er klar til brug
  - [ ] Vi skal have ryddet op i datoen på cluster så vi har dataet opdelt efter træningsmetode og formål.


