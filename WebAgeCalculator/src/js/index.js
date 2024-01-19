/*? no js js needed from me */

function checkAge(){
    const birthdayInput = document.getElementById("birthday");
    const birthday = birthdayInput.value;
    
    const now = new Date().getFullYear(); //bernilai tahun saat ini secara otomatis
    const age = now - birthday;
    
    const resultView = document.getElementById("result")
    resultView.innerHTML = `Usia Anda saat ini adalah ${age} tahun.`
}