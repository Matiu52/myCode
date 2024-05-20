const btn = document.getElementById("btn")
const trailer = document.getElementById("trailer")

function triggerBtn(){
    if(trailer.paused){
        trailer.play()
        btn.innerHTML = "PAUSE"
    }else{
        trailer.pause()
        btn.innerHTML = "PLAY"
    }
}