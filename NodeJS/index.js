const http = require('http');
const rupiah = require('rupiah-format');
const fs = require('fs');
const os = require('os');
const host = 'localhost'
const port = 3002

// request adalah data masuk dari keluar
// response adalah data keluar dari sistem
const server = http.createServer(function (request, response) {
    const nama = "Matius Andreatna";
    let uang = 500000;
    let jajan = 150000;
    let sisa = uang - jajan;

    uang = rupiah.convert(uang);
    jajan = rupiah.convert(jajan);
    sisa = rupiah.convert(sisa);

    fs.appendFile('sisaUang.txt', sisa, ()=> {
        console.log('data uang berhasi disimpan')
    });

    const sisaRAM = os.freemem();
    const jumlahCPU = os.cpus();

    function checkCPU(){
        let myCPU = [];
        jumlahCPU.map((cpu, i) => {
            myCPU.push(cpu.model)
        })
        return myCPU[0]
    }

    const hasil = `
    <head>
        <title>${nama}</title>
    </head>
    <body>
        <h1 style='background:black;color:white;padding:20px;text-align:center'>NODE JS UANG JAJAN</h1>
        <p>Halo nama saya ${nama}. Saya jajan sebanyak ${jajan}, uang saya tadinya ${uang} sekarang menjadi ${sisa}. 😁 </p>
        <h5>sisa RAM Laptop saya: ${sisaRAM}</h5>
        <h5>Merek CPU Laptop saya: ${checkCPU()}</h5>
    </body>
    `

    response.statusCode = 200;
    response.end(hasil);
});

server.listen(port, host, function () {
    console.log(`server menyala di ${host}:${port} 🎈`);
}) 