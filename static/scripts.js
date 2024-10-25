const clients = {{ clients | tojson }};

function showClientDetails(clientId) {
    const client = clients.find(c => c.id == clientId);
    if (client) {
        const detailsDiv = document.getElementById("client-details");
        detailsDiv.innerHTML = `
            <img src="/static/uploads/${client.photo}" alt="${client.name}">
            <p><strong>Name:</strong> ${client.name}</p>
            <p><strong>Patient ID:</strong> ${client.id}</p>
            <p><strong>Sex:</strong> ${client.sex}</p>
            <p><strong>Age:</strong> ${client.age}</p>
            <p><strong>Address:</strong> ${client.address}</p>
            <p><strong>Previous Diagnosis:</strong> ${client.diagnosis}</p>
            <button class="take-questionnaire-btn" onclick="takeQuestionnaire(${client.id})">Take Questionnaire</button>
            <button class="edit-client-btn" onclick="editClient(${client.id})">Edit Info</button>
        `;
    }
}

function takeQuestionnaire(clientId) {
    window.location.href = `/select/${clientId}`;
}

function editClient(clientId) {
    const client = clients.find(c => c.id == clientId);
    if (client) {
        const detailsDiv = document.getElementById("client-details");
        detailsDiv.innerHTML = `
            <h3>Edit Client Information</h3>
            <form id="edit-client-form" action="/edit_client/${clientId}" method="POST" enctype="multipart/form-data">
                <label for="name">Name:</label>
                <input type="text" id="name" name="name" value="${client.name}" required>
                
                <label for="sex">Sex:</label>
                <select id="sex" name="sex" required>
                    <option value="Male" ${client.sex === 'Male' ? 'selected' : ''}>Male</option>
                    <option value="Female" ${client.sex === 'Female' ? 'selected' : ''}>Female</option>
                </select>
                
                <label for="age">Age:</label>
                <input type="number" id="age" name="age" value="${client.age}" required>
                
                <label for="address">Address:</label>
                <input type="text" id="address" name="address" value="${client.address}" required>
                
                <label for="photo">Photo:</label>
                <input type="file" id="photo" name="photo" accept="image/*">
                
                <button type="submit" class="take-questionnaire-btn">Save Changes</button>
            </form>
        `;
    }
}

function addNewClient() {
    const detailsDiv = document.getElementById("client-details");
    detailsDiv.innerHTML = `
        <h3>Add New Client</h3>
        <form id="add-client-form" action="/add_client" method="POST" enctype="multipart/form-data">
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
            
            <button type="submit" class="take-questionnaire-btn">Save Client</button>
        </form>
    `;
}

function deleteClient(clientId) {
    if (confirm("Are you sure you want to delete this client?")) {
        fetch(`/delete_client/${clientId}`, {
            method: 'POST'
        }).then(() => {
            window.location.reload();
        });
    }
}
