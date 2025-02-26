const socket = new WebSocket(`ws://${window.location.hostname}:8005`);

socket.addEventListener("open", () => {
    console.log("Connected to Python");
});

var cnt=0;
var face_v=0;
var face_percent=0;
socket.onmessage=function(event){
    var message=JSON.parse(event.data);
    
    if(message.nextEvent=='endAct'){
        socket.close();
        document.getElementById('heading').innerText='Self-Care Activities are: 😊';
        document.getElementById('description').innerText=message.ans;
        
    }
    else if(message.nextEvent=='endPres'){
        socket.close();
        document.getElementById('heading').innerText='Medical Prescription is: 😊';
        document.getElementById('description').innerText=message.ans;
       
    } 
}

/**
 * Generates option buttons dynamically based on the given options array.
 */
function generateOptionButtons(options) {
    var optionsContainer = document.getElementById('options-container');
    console.log('#####################'+options);

    optionsContainer.innerHTML = '';
    document.getElementById('nextoptions-container').innerText='';
    document.getElementById('speak').innerText='';

    if(options.length !== 0) {
        var messageParagraph = document.createElement('p');
        messageParagraph.textContent = "Choose an option:";
        optionsContainer.appendChild(messageParagraph);

    
        for (var i = 0; i < options.length; i++) {
            var button = document.createElement('button');
            button.textContent = options[i];
            button.onclick = function() {
                selectOption(this.textContent);
            };
            optionsContainer.appendChild(button);
        }
    }else{

    }
}


/**
 * Selects an option by setting its text as the user input value.
 */
function selectOption(optionText) {
    document.getElementById('user-input').value = optionText;
}

var exampleOptions = ['Medical Prescription', 'Depression Test'];
generateOptionButtons(exampleOptions);

var nextOptions = ['Next'];


/**
 * Generates the next option buttons for navigation.
 */
function generateNextOptionButtons(options) {
    var optionsContainer = document.getElementById('nextoptions-container');
    console.log('#####################'+options);

    optionsContainer.innerHTML = '';
    document.getElementById('speak').innerText='';

    if(options.length !== 0) {
        var messageParagraph = document.createElement('p');
        optionsContainer.appendChild(messageParagraph);

    
        for (var i = 0; i < options.length; i++) {
            var button = document.createElement('button');
            button.textContent = options[i];
            optionsContainer.appendChild(button);
            button.onclick = function(){
                window.location.href = "self.html";
            };
        }
    }else{

    }
}

/**
 * Handles incoming messages from the WebSocket connection and updates the UI accordingly.
 */
socket.addEventListener("message", (event) => {
    console.log('Breakpoint')
    const response = JSON.parse(event.data);
    console.log("Received response from Python:", response);
    updatedUI(response);
});

/**
 * Sends data through the WebSocket connection.
 */
function pyconnect(data) {
    socket.send(data);
}
var nextEvent = '';
var bool=true;

/**
 * Handles the sending of user messages and processes the response accordingly.
 */
function sendMessage() {
    var data = document.getElementById('user-input').value;
    console.log(nextEvent);
    var age = localStorage.getItem('age');
    var gender = localStorage.getItem('gender');
    
    data += "==" + age + "==" +gender;


    var buttonClickPromise = new Promise(function(resolve, reject) {
        document.getElementById('user-input').addEventListener('input', function() {
            resolve();
        });
    });

    if (exampleOptions[0] == 'Medical Prescription') {
        data += '==' + 'process_data'
    } else { 
        buttonClickPromise.then(function() {
            console.log('Button clicked');
            data=document.getElementById('user-input').value;            
        });
        console.log('#####');
        console.log(data);
        console.log(nextEvent);
        console.log('#####');
        data += '==' + nextEvent;
    }

    console.log(data);
    console.log('Data being sent');
    if(bool)
    {
        pyconnect(data);
        bool=false;
    }
    else{
        bool=true;
    }
    
    document.getElementById('user-input').value = '';
}

/**
 * Reads aloud the content of the given HTML element.
 */
function updatedUI(response) {
    console.log('+++++++'+'}'+response.cnt);
    console.log('+++++++'+'}'+response.nextEvent);
    if(response.nextEvent=='endMed')
    {
        socket.close();
        document.getElementById('chat-body').innerText='Problem solved! 😊';
        document.getElementById('end-display').innerHTML= "<p>Thank you for using Wellness Chatbot.<p>";
        document.getElementById('ipms').innerText=response.ans;
        document.getElementById('nextoptions-container').innerText='';
        document.getElementById('voice_n').innerText='';
    }
    else if(response.nextEvent=='endDep')
    {
        document.getElementById('chat-body').innerText='Depression Detection test completed!! 😊';
        document.getElementById('end-display').innerHTML= "<p>Thank you for using Wellness Chatbot.<p>";

        cnt*=2;
        localStorage.setItem('cnt',cnt);

        document.getElementById('result').innerText="DASS score is:"+cnt;
        if(cnt>=28)
        {
            document.getElementById('ipms').innerText = 'You are experiencing extremely severe Depression';
        }else if(cnt>=21 && cnt<=27)
        {
            document.getElementById('ipms').innerText = 'You are experiencing severe Depression';
        }
        else if(cnt>=14 && cnt<=20)
        {
            document.getElementById('ipms').innerText = "You are experiencing moderate Depression";
        }else if(cnt>=10 && cnt<=13)
        {
            document.getElementById('ipms').innerText = "You are experiencing mild Depression";
        }else{
            document.getElementById('ipms').innerText = "Your are experiencing normal Depression";
        }

        if(cnt>=20)
            {
                document.getElementById('ipms').innerText += ', extremely severe Anxiety';
            }else if(cnt>=15 && cnt<=19)
            {
                document.getElementById('ipms').innerText += ', severe Anxiety';
            }
            else if(cnt>=10 && cnt<=14)
            {
                document.getElementById('ipms').innerText += ", moderate Anxiety";
            }else if(cnt>=8 && cnt<=9)
            {
                document.getElementById('ipms').innerText += ", mild Anxiety";
            }else{
                document.getElementById('ipms').innerText += ", normal Anxiety";
            }

            if(cnt>=34)
                {
                    document.getElementById('ipms').innerText += '& extremely severe Stress.';
                }else if(cnt>=26 && cnt<=33)
                {
                    document.getElementById('ipms').innerText += '& severe Stress.';
                }
                else if(cnt>=19 && cnt<=25)
                {
                    document.getElementById('ipms').innerText += "& moderate Stress.";
                }else if(cnt>=15 && cnt<=18)
                {
                    document.getElementById('ipms').innerText += "& mild Stress.";
                }else{
                    document.getElementById('ipms').innerText += "& normal Stress.";
                }
        
        face_percent=face_v/21;
        document.getElementById('percent').innerText="Depression score is :"+ parseFloat(face_percent.toFixed(2))+ '%';
        document.getElementById('options-container').innerText='';
        generateNextOptionButtons(nextOptions);
        document.getElementById('speak').innerText='';
        document.getElementById('voice_n').innerText='';

    }
    else{
        console.log('Printing msg :'+response.msg);
        console.log('Printing options : '+response.option)
        document.getElementById('chat-body').innerText = response.msg;
        exampleOptions = response.option;
        generateOptionButtons(exampleOptions);
        document.getElementById('user-input').value = '';
        cnt+=response.cnt -1;
        face_v+=response.face_value;
        console.log('--------------'+cnt+response.cnt);
        console.log(cnt);
        console.log('--------------');
        nextEvent=response.nextEvent;
        sendMessage();
        document.getElementById('voice_n').innerText='';
    }
}

function sendMessage_Act() {
    var data ='SelfCareActivities';
    console.log(nextEvent);
    var age = localStorage.getItem('age');
    var gender = localStorage.getItem('gender');
    var cnt_n = localStorage.getItem('cnt');
    
    data += "==" + age + "==" +gender + "==" + cnt_n;

    pyconnect(data);
}

function sendMessage_presc(){
    var data='prescription';
    var age = localStorage.getItem('age');
    var gender = localStorage.getItem('gender');
    var cnt_n = localStorage.getItem('cnt');
    
    data += "==" + age + "==" +gender + "==" + cnt_n;

    pyconnect(data);
}



function stopSpeech() {
    // Stop speech synthesis
    window.speechSynthesis.cancel();
}

function sendMessage_voice(){
    var data='voice based';
    data += "==" + "process_data";

    pyconnect(data);
}

function speakText(id_n) {
    // Get the text content of the div with id 'percent'
    var textToSpeak = document.getElementById(id_n).textContent;

    // Create a new SpeechSynthesisUtterance object
    var speech = new SpeechSynthesisUtterance();

    // Set the text to be spoken
    speech.text = textToSpeak;

    // Use the default speech synthesizer
    window.speechSynthesis.speak(speech);
}

/**
 * Toggles reading aloud the description text.
 */
var flag=false;
function readAloud(){
    if(flag==false){
        speakText('description');
        flag=true;
    }
    else{
        stopSpeech();
        flag=false;
    }
}