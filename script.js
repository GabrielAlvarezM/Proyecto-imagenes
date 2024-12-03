function postData() {
    $.ajax({
        url: 'http://localhost:8000/run-script',
        method: 'GET',
        dataType: 'json',
        success: function(data) {
          console.log(data);
        },
        error: function(xhr, status, error) {
          console.error('Error:', error);
        }
      });      
}

const nImages = 3;

function buttonPress(id){
    const toggleButton = document.getElementById("button"+id);
    let imagenoriginal = document.getElementById("imagenoriginal");
    let imagenprocesada = document.getElementById("imagenprocesada");
    let imagendefectos = document.getElementById("imagendefects");
    
    toggleButton.classList.toggle("active");
    for(let i = 1; i <= nImages; i++){
        if(i != id && document.getElementById("button"+i).classList.contains("active")){
            document.getElementById("button"+i).classList.toggle("active");
        }
    }

    //toggleButton.textContent = toggleButton.classList.contains("active") ? "ON" : "OFF";
    imagenoriginal.style.background = "url('imagen_"+id+".png') no-repeat center center / cover";
    imagenprocesada.style.background = "url('./directorio_de_salida/imagen_"+id+"_processed.png') no-repeat center center / cover";
    imagendefectos.style.background = "url('./directorio_de_salida/imagen_"+id+"_defects.png') no-repeat center center / cover";
    console.log("pressed");
}

function callbackFunc(response) {
    console.log(response);
}

postData();