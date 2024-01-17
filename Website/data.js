const nama = "matius andreatna";
let usia = 24;
console.log(`nama saya adalah ${nama} dan usia saya ${usia} tahun.`);
//console.log('nama saya adalah', nama, 'usia saya adalah', usia);

let biodata = document.getElementById('biodata')
console.log(biodata)

function repeatName(name, usia) {
    let generasi;

    if (usia > 10 && usia < 18) {
        generasi = "generasi remaja"
    }
    else if (usia >= 18 && usia < 30) {
        generasi = "generasi dewasa"
    }
    else if (usia >= 30) {
        generasi = "generasi tua"
    }
    else if (usia >= 2 && usia <= 10) {
        generasi = "generasi anak-anak"
    }
    else {
        generasi = "generasi bayi"
    }

    return biodata.innerHTML = generasi
}

repeatName(nama, usia)