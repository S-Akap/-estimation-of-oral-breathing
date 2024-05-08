var labels = [];
var data = [];
const canvasElm = document.querySelector("#graph-canvas");
const loaderElm = document.querySelector(".loader");
var resultLineChart
fetch('/analysis')
    .then(response => {
        return response.json();
    })
    .then(jsonData => {
        loaderElm.classList.add("fin")

        console.log('オブジェクト形式に変換したJSONデータ:', jsonData); // パースされたJSONデータを出力
        labels = jsonData["labels"]
        data = jsonData["data"]
        resultLineChart = new Chart(canvasElm, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: '呼吸流量',
                    data: data,
                    borderColor: "rgba(0,0,255,1)",
                    lineTension: 0.3,
                    borderWidth: 3,
                    pointRadius: 0,
                }],
            },
            options: {
                scales: {
                    x:{
                        display: true,
                        title:{
                            display: true,
                            text: '時刻[s]'
                        }
                    },
                    y:{
                        display: true,
                        min: 0,
                        title:{
                            display: true,
                            text: '正規化された推定値'
                        },
                    }
                },
            }
        })
    })
    .catch(error => {
        console.error('An error occurred:', error);
    })

document.querySelector("#download").addEventListener("click", function () {
    let bom = new Uint8Array([0xEF, 0xBB, 0xBF]);
    let csvContent = "";
    var baseData = [labels,data]
    baseData.forEach(function (row) {
        csvContent += row.join(",") + "\n";
    });

    let blob = new Blob([bom, csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "result.csv";
    a.click();
    URL.revokeObjectURL(url);
});