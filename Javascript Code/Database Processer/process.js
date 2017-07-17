let jsonObj;
let dbArr = [];
let dbObj = {};
let props = [];

let DATABASE_FILE_SIZE = 999;


function readJSON(file) {
    let rawFile = new XMLHttpRequest();
    rawFile.overrideMimeType("application/json");
    rawFile.open("GET", file, true);
    rawFile.onreadystatechange = function () {
        if (rawFile.readyState === 4 && rawFile.status == "200") {
            jsonObj = JSON.parse(rawFile.responseText);
        }
    }
    rawFile.send(null);
}

function dataToArray(showTest) {
    for (let i = 1; i <= DATABASE_FILE_SIZE; i++)
        dbArr.push(jsonObj[i + ""]);
        
    processDatabase(showTest); //move on to process this array
}

function processDatabase(showTest) {
    for (let i = 0; i < DATABASE_FILE_SIZE; i++) {
        let form = dbArr[i].numbers;
        let formStr = form.toString();

        if (!props.includes(formStr)) {
            props.push(formStr);
            dbObj[formStr] = [dbArr[i]];
        } else {
            dbObj[formStr].push(dbArr[i]);
        }
    }

    if (showTest)
        test(); 
}

function test() {
    for (var property in dbObj) {
        if (dbObj.hasOwnProperty(property)) {
            console.log(dbObj[property]);
        }
    }
}
