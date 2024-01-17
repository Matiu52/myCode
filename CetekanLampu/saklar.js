let lamp = []
let idSaklarLamp = []

function saklar(idSaklar, idLampu) {
    let toggle = document.getElementById(idSaklar)
    let lampu = document.getElementById(idLampu)

    if (toggle.checked) {
        lampu.src = 'assets/images/on.gif';
    } else {
        lampu.src = 'assets/images/off.gif';
    }
}

function saklarRuangan(idSaklar, lamp, idSaklarLamp) {
    let toggleRuang = document.getElementById(idSaklar)
    let noLamp = lamp
    let noSaklarLamp = idSaklarLamp
    if (toggleRuang.checked) {
        for (i = 0; i < noLamp.length; i++) {
            document.getElementById(noLamp[i]).src = 'assets/images/on.gif';
            document.getElementById(noSaklarLamp[i]).checked = true;
        }
    } else {
        for (i = 0; i < noLamp.length; i++) {
            document.getElementById(noLamp[i]).src = 'assets/images/off.gif';
            document.getElementById(noSaklarLamp[i]).checked = false;
        }
    }

}