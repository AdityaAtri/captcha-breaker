function ChangeCaptcha() {
    var chars = "12345679ACEFGHIJKLMNOPQRSTUVWXYZ"; 
    var string_length = 4; 
    var ChangeCaptcha = '';
    var max= chars.length -1;
    var min=0;
    for (var i=0; i<string_length; i++) 
    {
        var random = Math.floor(Math.random() * (+max - +min) + +min);
        ChangeCaptcha = ChangeCaptcha + chars.substring(random,random+1); // single character is selected from the string 
    }
    
    document.getElementById('randomfield').value = ChangeCaptcha;
}

function check() 
{ // Function which checks if the entered value is matching the Captcha
    if(document.getElementById('CaptchaEnter').value == document.getElementById('randomfield').value ) 
    {
        window.open('https://www.google.co.in','_self');    
    }
    else 
    {
        alert('Please re-check the captcha');  
    }
}


