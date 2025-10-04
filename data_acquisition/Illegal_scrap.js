(() => {
    const grid = document.querySelector('.MuiDataGrid-virtualScroller');
    const totalHeight = grid.scrollHeight;
    const totalWidth = grid.scrollWidth;
    const stepY = 200; // scroll vertical
    const stepX = 300; // scroll horizontal
    let currentY = 0;
    let currentX = 0;

    let data = [];

    const scrollInterval = setInterval(() => {
        // Scroll horizontal
        if (currentX < totalWidth) {
            grid.scrollTo(currentX, currentY);
            currentX += stepX;
        } else {
            // Une fois à droite, revenir à gauche et descendre
            currentX = 0;
            currentY += stepY;
            grid.scrollTo(currentX, currentY);
        }

        // Récupérer les lignes visibles
        const rows = document.querySelectorAll('.MuiDataGrid-row');
        rows.forEach(r => {
            const rowData = Array.from(r.querySelectorAll('.MuiDataGrid-cell')).map(c => c.innerText);
            if (!data.some(d => JSON.stringify(d) === JSON.stringify(rowData))) {
                data.push(rowData);
            }
        });

        // Fin du scroll
        if (currentY >= totalHeight) {
            clearInterval(scrollInterval);
            console.log('Toutes les lignes ont été chargées :');
            console.log(JSON.stringify(data, null, 2));

            // Télécharger le JSON
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "tableau.json";
            a.click();
            URL.revokeObjectURL(url);
        }
    }, 300);
})();
