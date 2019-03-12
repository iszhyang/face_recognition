 var form = new FormData();
        var url = '';
        function selectFile(){
            var files = document.getElementById('pic').files;
            console.log(files[0]);
            if(files.length == 0){
                return;
            }
            var file = files[0];
            var reader = new FileReader();
            console.log(reader);
            reader.readAsBinaryString(file);
            reader.onload = function(f){
                var result = document.getElementById("result");
                var src = "data:" + file.type + ";base64," + window.btoa(this.result);
                result.innerHTML = '<img src ="'+src+'" style="width: auto;height: auto;max-width: 100%;max-height: 100%;"/>';
            };
            console.log('file',file);
            
            form.append('file',file);
            console.log(form.get('file'));
        }
	 
