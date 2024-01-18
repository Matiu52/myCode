let i = 0
const mahasiswa = [
    {
        nama: "matius andreatna",
        alamat: "kampung inggris",
        usia: 24,
        semester: 3
    }, {
        nama: "dea",
        alamat: "kampung belanda",
        usia: 30,
        semester: 8
    }
]

//standar function
function getDetailHuman() {
    i += 1
    if (i > 5) {
        console.log('lebih dari 5x diklik')
    } else {
        console.log('jatah klik masih ada')
    }
}

//arrow function
const getDetailHuman2 = () => {
    i += 1
    i > 5 ? console.log('lebih dari 5x bro human2') : console.log('jatah masih ada human2')
}

function getDetailData() {
    mahasiswa.map(function(result, i){
        console.table(result)
    })
    mahasiswa.forEach(result=> {
        console.log(result)
    })
    console.log(`data yang anda cari adalah: ${mahasiswa}`)
}

class Hewan {
    warna
    keahlian
    constructor(nama) {
        this.nama = nama
    }

    set newColor(color) {
        this.warna = color
    }

    set newSkill(skill) {
        this.keahlian = skill
    }

    get detail() {
        return `Hi saya ${this.nama}, saya berwarna ${this.warna} dan keahlian saya ${this.keahlian}`
    }
}

const kucing = new Hewan('bread')
kucing.newColor = "red"
kucing.newSkill = "mengeong"
console.log(kucing.detail)