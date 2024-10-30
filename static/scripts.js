const patients = {{ patients | tojson }};

function showpatientDetails(patientId) {
    const patient = patients.find(c => c.id == patientId);
    if (patient) {
        const detailsDiv = document.getElementById("patient-details");
        detailsDiv.innerHTML = `
            <img src="/static/uploads/${patient.photo}" alt="${patient.name}">
            <p><strong>Name:</strong> ${patient.name}</p>
            <p><strong>Patient ID:</strong> ${patient.id}</p>
            <p><strong>Sex:</strong> ${patient.sex}</p>
            <p><strong>Age:</strong> ${patient.age}</p>
            <p><strong>Address:</strong> ${patient.address}</p>
            <p><strong>Previous Diagnosis:</strong> ${patient.diagnosis}</p>
            <button class="take-questionnaire-btn" onclick="takeQuestionnaire(${patient.id})">Take Questionnaire</button>
            <button class="edit-patient-btn" onclick="editpatient(${patient.id})">Edit Info</button>
        `;
    }
}

function takeQuestionnaire(patientId) {
    window.location.href = `/select/${patientId}`;
}

function editpatient(patientId) {
    const patient = patients.find(c => c.id == patientId);
    if (patient) {
        const detailsDiv = document.getElementById("patient-details");
        detailsDiv.innerHTML = `
            <h3>Edit patient Information</h3>
            <form id="edit-patient-form" action="/edit_patient/${patientId}" method="POST" enctype="multipart/form-data">
                <label for="name">Name:</label>
                <input type="text" id="name" name="name" value="${patient.name}" required>
                
                <label for="sex">Sex:</label>
                <select id="sex" name="sex" required>
                    <option value="Male" ${patient.sex === 'Male' ? 'selected' : ''}>Male</option>
                    <option value="Female" ${patient.sex === 'Female' ? 'selected' : ''}>Female</option>
                </select>
                
                <label for="age">Age:</label>
                <input type="number" id="age" name="age" value="${patient.age}" required>
                
                <label for="address">Address:</label>
                <input type="text" id="address" name="address" value="${patient.address}" required>
                
                <label for="photo">Photo:</label>
                <input type="file" id="photo" name="photo" accept="image/*">
                
                <button type="submit" class="take-questionnaire-btn">Save Changes</button>
            </form>
        `;
    }
}

function addNewpatient() {
    const detailsDiv = document.getElementById("patient-details");
    detailsDiv.innerHTML = `
        <h3>Add New patient</h3>
        <form id="add-patient-form" action="/add_patient" method="POST" enctype="multipart/form-data">
            <label for="name">Name:</label>
            <input type="text" id="name" name="name" required>
            
            <label for="sex">Sex:</label>
            <select id="sex" name="sex" required>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
            </select>
            
            <label for="age">Age:</label>
            <input type="number" id="age" name="age" required>
            
            <label for="address">Address:</label>
            <input type="text" id="address" name="address" required>
            
            <label for="photo">Photo:</label>
            <input type="file" id="photo" name="photo" accept="image/*">
            
            <button type="submit" class="take-questionnaire-btn">Save patient</button>
        </form>
    `;
}

function deletepatient(patientId) {
    if (confirm("Are you sure you want to delete this patient?")) {
        fetch(`/delete_patient/${patientId}`, {
            method: 'POST'
        }).then(() => {
            window.location.reload();
        });
    }
}
