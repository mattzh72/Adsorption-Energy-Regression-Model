let moleculeObj;
let energies;

let insignificantPropertiesArr = [];
let allPropertiesArr = [];

//set to -1 to turn off rounding
let ENERGY_VAL_SENSITIVITY = -1;

function processMoleculeArr(moleculeArr) {
    moleculeObj = {};
    energies = [];

    if (ENERGY_VAL_SENSITIVITY === -1)
        ENERGY_VAL_SENSITIVITY = 16;

    for (let i = 0; i < moleculeArr.length; i++) {
        let molecule = moleculeArr[i];
        let energy = molecule.energy.toFixed(ENERGY_VAL_SENSITIVITY);;

        if (!energies.includes(energy)) {
            energies.push(energy);
            moleculeObj[energy] = [moleculeArr[i]];
        } else {
            moleculeObj[energy].push(moleculeArr[i]);
        }
    }

    test();
}

function test() {
    let currentMolecule;

    console.error("Starting analysis:");
    for (let property in moleculeObj) {
        currentMolecule = moleculeObj[property][0].numbers;
        if (moleculeObj.hasOwnProperty(property)) {
            console.warn("Found " + moleculeObj[property].length + " molecules with adsorption energy " + moleculeObj[property][0].energy.toFixed(ENERGY_VAL_SENSITIVITY));

            if (moleculeObj[property].length > 1) {
                insignificantPropertiesArr = compareMolecules(moleculeObj[property]);
                allPropertiesArr = Object.getOwnPropertyNames(moleculeObj[property][0]);
                let significantPropertiesArr = allPropertiesArr.filter(checkDuplicates);
                console.log("The above dataset suggests that [" + significantPropertiesArr + "] are significant in determining adsorption energy");
            }
        }
    }
    console.error("Ending analysis for " + currentMolecule);
    console.log("");
    console.log("");
}

function checkDuplicates(property) {
    return !insignificantPropertiesArr.includes(property);
}

function compareMolecules(moleculeArray) {
    let insignificantProperties = [];

    for (let i = 0; i < moleculeArray.length - 1; i++) {
        let mol1 = moleculeArray[i];
        let mol2 = moleculeArray[i + 1];

        for (let property in mol1) {
            if (mol1.hasOwnProperty(property)) {
                let property1 = mol1[property];
                let property2 = mol2[property];

                if (property1 instanceof Array) {
                    if (!(equalArrays(property1, property2)))
                        insignificantProperties.push(property);
                }
                else if (typeof property1 === 'string' || property1 instanceof String) {
                    if (property1 !== property2)
                        insignificantProperties.push(property);
                }
                else if (typeof property1 === 'number') {
                    if (property1 !== property2)
                        insignificantProperties.push(property);
                }
                else{
                    for (let prop in property1) {
                        let propertyVal1 = property1[prop];
                        let propertyVal2 = property2[prop];
                        
                        if (propertyVal1 instanceof Array) {
                            if (!(equalArrays(propertyVal1, propertyVal2))){
                                insignificantProperties.push(property);
                                break;
                            }
                        }
                        else if (typeof propertyVal1 === 'string' || propertyVal1 instanceof String) {
                            if (propertyVal1 !== propertyVal2){
                                insignificantProperties.push(property);
                                break;
                            }
                        }
                        else if (typeof propertyVal1 === 'number') {
                            if (propertyVal1 !== propertyVal2){
                                insignificantProperties.push(property);
                                break;
                            }
                        }
                        
                    }
                }
            }
        }
    }

    return insignificantProperties;
}

function equalArrays(array, array2) {
    // if the other array is a falsy value, return
    if (!array)
        return false;

    // compare lengths - can save a lot of time 
    if (array2.length != array.length)
        return false;

    for (var i = 0, l = array2.length; i < l; i++) {
        // Check if we have nested arrays
        if (this[i] instanceof Array && array[i] instanceof Array) {
            // recurse into the nested arrays
            if (!this[i].equals(array[i]))
                return false;
        } else if (this[i] != array[i]) {
            // Warning - two different object instances will never be equal: {x:20} != {x:20}
            return false;
        }
    }
    return true;
}
