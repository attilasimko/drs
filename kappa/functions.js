function dephase_step(alpha, dephase, rfon, t2) 
{
    var delta = dephase / (t2 * 10)
    var noise = (Math.random() - 0.5) / 40;
    if (rfon)
    {
      var value = alpha + Math.sign(delta) * (Math.PI / 16) + noise;
      if (Math.abs(value%(Math.PI*2)) < Math.abs(alpha - value) * 1.1)
          return 0
      else
          return value%(Math.PI*2);
    }
    else
    {
      var value = alpha - delta + noise;
      return value%(Math.PI*2);
    }
}

function degrees_to_radians(degrees)
{
  var pi = Math.PI;
  return degrees * (pi/180);
}
  
function is_alpha_synced(alpha)
{
    for (let i = 0; i < alpha.length; i++)
    {
        if (Math.abs(alpha[i]) > 0.2) 
            return false
    }
    return true
}

function canvas_arrow(context, fromx, fromy, tox, toy, headlen) {
    var dx = tox - fromx;
    var dy = toy - fromy;
    var angle = Math.atan2(dy, dx);
    context.moveTo(fromx, fromy);
    context.lineTo(tox, toy);
    context.stroke();
    context.beginPath();
    context.moveTo(tox, toy);
    context.lineTo(tox - headlen * Math.cos(angle - Math.PI / 6), toy - headlen * Math.sin(angle - Math.PI / 6));
    context.lineTo(tox - headlen * Math.cos(angle + Math.PI / 6), toy - headlen * Math.sin(angle + Math.PI / 6));
    context.fill();
    context.closePath()
    context.stroke();
}


class mri_contrast{
    constructor (canvasId, teId, trId) {
        this.canvas = document.getElementById(canvasId);
        if (this.canvas == null)
            return

        this.teId = teId;
        this.trId = trId;
        this.old_time = 0;
        this.old_te = 0;
        this.old_tr = 0;
        
        var slider = document.getElementById(this.teId);
        slider.tag = this;
        slider.oninput = function() 
        {
            this.tag.tick();
        }
        
        var slider = document.getElementById(this.trId);
        slider.tag = this;
        slider.oninput = function() 
        {
            this.tag.tick();
        }
        this.tick();

    }

    draw = (te, tr) => {
        var ctx = this.canvas.getContext("2d");
    
        let newImage = new Image();
        newImage.onload = () => {ctx.drawImage(newImage, 0, 0, 300, 300);}
        newImage.src = 'https://mlsatellite.com/wp-content/uploads/2022/09/' + te + '_' + tr + '.png';
        return true;
    }

    
    tick()
    {
        var slider_te = document.getElementById(this.teId);
        var te = slider_te.value;
        var slider_tr = document.getElementById(this.trId);
        var tr = slider_tr.value;
        if (this.draw(te, tr))
        {
            this.old_te = te;
            this.old_tr = tr;
        }
    }
}


class mri_rf{
    constructor(canvasId, b0Id, rfId){
        this.canvas = document.getElementById(canvasId);
        if (this.canvas == null)
            return
    
        this.b0Id = b0Id;
        this.rfId = rfId;
        this.phase = 0;
        this.old_time = 0;
        this.alpha = Math.PI / 12;

        window.requestAnimationFrame(this.tick);

    }


    draw = (canvas, phase, alpha, rfon) => {
        
        var ctx = canvas.getContext("2d");

        ctx.resetTransform();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = "#FFFFFF";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = "#000000";

        var slider = document.getElementById("b0");

        ctx.strokeStyle = "#000000";
        var xcenter = 165;
        var ycenter = 230;
        var length = 130;
        var ellw = length * Math.sin(alpha);
        var ellh = 0.05 * length;// * Math.cos(alpha);
        let x0 = xcenter;
        let y0 = ycenter - length * Math.cos(alpha);
 
        let x = x0 + ellw * Math.sin(phase);
        let y = y0 + ellh * Math.cos(phase);
            
        ctx.beginPath();
        ctx.lineWidth = 1;
        canvas_arrow(ctx, xcenter, ycenter + 30, xcenter, ycenter - 160, 5)
        ctx.stroke();

        ctx.beginPath();
        ctx.lineWidth = 3;
        ctx.setLineDash([10, 5]);
        ctx.ellipse(x0, y0, ellw, ellh, 0, 0, 2 * Math.PI);
        ctx.stroke();

        ctx.beginPath();
        ctx.setLineDash([]);
        ctx.lineWidth = 2;
        canvas_arrow(ctx, xcenter, ycenter, x, y, 7)

        ctx.beginPath();
        ctx.lineWidth = 5;
        canvas_arrow(ctx, 25, 250, 25, 200, 5)
        ctx.stroke();

        ctx.font = "14px Arial";
        ctx.fillText("B0", 0, 250);
        
        ctx.beginPath();
        ctx.lineWidth = 1;
        

        if (rfon)
        {
        ctx.font = "12px Arial";
        ctx.fillStyle = "#000000"
        ctx.fillText("RF Pulse", 50, ycenter - 10);
        ctx.beginPath();
        ctx.lineWidth = 2;
        canvas_arrow(ctx, 50, ycenter + 5, 110, ycenter + 5, 5)
        canvas_arrow(ctx, 50, ycenter, 110, ycenter, 5)
        canvas_arrow(ctx, 50, ycenter - 5, 110, ycenter - 5, 5)
        ctx.stroke();

        }



    }

    tick = (time) => {
        var dt = (time - this.old_time)/500;
        this.old_time = time;
        var slider = document.getElementById(this.b0Id);
        var freq = slider.value;
        if (this.rfId != null)
            var rf = document.getElementById(this.rfId).checked;
        else
            var rf = false;

        if (rf & this.alpha < Math.PI / 2)
          {
            this.alpha = Math.min(this.alpha + Math.PI / (2 * 15), Math.PI / 2);
          if (this.alpha == Math.PI/2 & this.rfId != null)
            document.getElementById(this.rfId).checked = false;
          }
        else if (!rf & this.alpha > Math.PI / 12)
        {
            this.alpha -= Math.max(Math.PI / (6 * 75), 0);
        }
        this.phase += freq * dt
        this.draw(this.canvas, this.phase, this.alpha, rf);

        window.requestAnimationFrame(this.tick);
    }
}

class mri_full_dephasing{
    constructor(canvasId, rfpulseId, showt1t2, t1Id, t2Id, showflip, flipId, showgrad, gradplusId, gradnegId){
        this.canvas = document.getElementById(canvasId);
        if (this.canvas == null)
            return

        this.rfpulseId = rfpulseId;
        this.t1Id = t1Id;
        this.t2Id = t2Id;
        this.flipId = flipId;
        this.gradplusId = gradplusId;
        this.gradnegId = gradnegId;
        
        this.Mz = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        this.Mxy = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        this.Mz0 = 1
        this.Mxy0 = 1
        this.dephase = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        this.alpha = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        this.dT = 0;
        this.rfon = true
        this.old_time = 0;
        this.showt1t2 = showt1t2;
        this.showflip = showflip;
        this.showgrad = showgrad;

        if (showt1t2)
        {
            var slider = document.getElementById(this.t1Id);
            slider.tag = this;
            slider.oninput = function() 
            {
                this.tag.switch_pulse(true);
            }
    
            slider = document.getElementById(this.t2Id);
            slider.tag = this;
            slider.oninput = function() 
            {
                this.tag.switch_pulse(true);
            }
        }
        
        if (showflip)
        {
            var button = document.getElementById(this.flipId);
            button.tag = this;
            button.onclick = function() 
            {
                if (showgrad)
                {
                    this.switch_gradpos(false);
                    this.switch_gradneg(false);
                }
                this.tag.switch_flip(true);
            }
        }
        

        if (showgrad)
        {
            var toggle = document.getElementById(this.rfpulseId);
            toggle.tag = this;
            toggle.onclick = function() 
            {
                this.tag.switch_gradpos(false);
                this.tag.switch_gradneg(false);
            }
            toggle = document.getElementById(this.gradplusId);
            toggle.tag = this;
            toggle.onclick = function() 
            {
                this.tag.switch_gradpos(this.checked);
            }

            toggle = document.getElementById(this.gradnegId);
            toggle.tag = this;
            toggle.onclick = function() 
            {
                this.tag.switch_gradneg(this.checked);
            }
        }
        
        
        window.requestAnimationFrame(this.tick);
    }
    
     switch_pulse = (value) => {
        document.getElementById(this.rfpulseId).checked = value;
     }
    
     switch_flip = (value) => {
        for (let i = 0; i < this.dephase.length; i++)
        {
            this.dephase[i] *= -1;
        }
        this.dT = 0;
        this.Mz[19] *= -1;
        this.Mz0 = this.Mz[19];
     }
     switch_gradpos = (value) => {
        document.getElementById(this.gradnegId).checked = false;
        if (value)
        {
            this.switch_pulse(false);
            for (let i = 0; i < this.dephase.length; i++)
            {
                this.dephase[i] *= 3;
            }
        }
    }
    switch_gradneg = (value) => {
        document.getElementById(this.gradplusId).checked = false;
        if (value)
        {
            this.switch_pulse(false);
            for (let i = 0; i < this.dephase.length; i++)
            {
                this.dephase[i] *= -1;
            }
        }
    }
     
    draw = (canvas, time, rf) => {
        if (this.showt1t2)
        {
            var slider = document.getElementById(this.t1Id);
            var T1 = slider.value / 2;
            
            var slider = document.getElementById(this.t2Id);
            var T2 = slider.value / 2;
        }
        else
        {
            if (this.showgrad | this.showflip)
            {
                var T1 = 6.0;
                var T2 = 3.3;
            }
            else
            {
                var T1 = 1.6;
                var T2 = 1.4;
            }
        }

        
        this.dT += time;
        
        if (!this.rfon)
            this.Mz_current = this.Mz0 + ((1 - this.Mz0) * (1 - Math.exp(-this.dT / T1)))
        else
            this.Mz_current = this.Mz0 * (Math.exp(-(this.dT) / 0.1))

        this.Mz.shift();
        this.Mz = this.Mz.concat(this.Mz_current)


        //if (!this.rfon)
        //    this.Mxy_current = this.Mxy0 + ((1 - this.Mxy0) * (1 - Math.exp(-this.dT / T2)))
        //else
        //    this.Mxy_current = this.Mxy0 * (Math.exp(-(this.dT) / 0.1))

        var Mx = 0;
        var My = 0;
        for (let i=0; i < this.alpha.length; i++)
        {
            Mx += Math.cos(this.alpha[i]);
            My += Math.sin(this.alpha[i]);
        }
        this.Mxy_current = Math.sqrt(Mx*Mx + My*My) / this.alpha.length;

        this.Mxy.shift();
        this.Mxy = this.Mxy.concat(this.Mxy_current * this.Mxy_current)


        var ctx = canvas.getContext("2d");

        ctx.resetTransform();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.lineWidth = 2;
        ctx.strokeStyle = "#000000";

        
        if (!this.rfon & rf)
        {
            this.rfon = true;
            this.dT = 0;
            this.Mz0 = this.Mz[19]
            this.Mxy0 = this.Mxy[19]
            for (let i=0; i < this.dephase.length; i++)
            {
                if (this.alpha[i] < 0)
                    if (Math.abs(this.alpha[i]) < Math.PI)
                        this.dephase[i] = 4;
                    else
                        this.dephase[i] = -4;
                else
                    if (Math.abs(this.alpha[i]) < Math.PI)
                        this.dephase[i] = -4;
                    else
                        this.dephase[i] = 4;
            }
            
        }
        if (this.rfon & !rf)
        {
            this.rfon = false;
            this.dT = 0;
            this.Mz0 = this.Mz[19]
            this.Mxy0 = this.Mxy[19]
            for (let i=0; i < this.dephase.length; i++)
            {
                this.dephase[i] = (Math.random() - 0.5) / 0.5;
            }
        }


        for (let i=0; i < this.alpha.length; i++)
        {
            this.alpha[i] = dephase_step(this.alpha[i], this.dephase[i], this.rfon, T2);
        }

        
        // Plotting Mz
        var x0 = 10;
        var y0 = 80;
        ctx.setLineDash([]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.strokeStyle = "#000000";
        ctx.moveTo(x0, y0 - (50 * this.Mz[0]));
        for (let i=0; i < this.Mz.length; i++)
        {
            ctx.lineTo(x0 + (i * 5), y0 - (50 * this.Mz[i]));
        }
        ctx.stroke();

        canvas_arrow(ctx, x0, y0, x0 + 130, y0, 5)
        canvas_arrow(ctx, x0, y0, x0, y0 - 60, 5)
        ctx.stroke();



        // Plotting Mxy
        x0 = 160;
        y0 = 80;
        ctx.setLineDash([]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.strokeStyle = "#000000";
        ctx.moveTo(x0, y0 - (50 * this.Mxy[0]));
        for (let i=0; i < this.Mxy.length; i++)
        {
            ctx.lineTo(x0 + (i * 5), y0 - (50 * this.Mxy[i]));
        }
        ctx.stroke();

        canvas_arrow(ctx, x0, y0, x0 + 130, y0, 5)
        canvas_arrow(ctx, x0, y0, x0, y0 - 60, 5)
        ctx.stroke();

        x0 = 150
        y0 = 170
        
        ctx.beginPath();
        ctx.lineWidth = 1;
        ctx.setLineDash([10, 5]);
        ctx.ellipse(x0, y0, 100, 30, 0, 0, 2 * Math.PI);
        ctx.stroke();

        for (let i=0; i < this.alpha.length; i++)
        {
            ctx.beginPath();
            ctx.setLineDash([]);
            ctx.lineWidth = 1;
            if (this.dephase[i] < 0)
                ctx.strokeStyle = "#e00627";
            else
                ctx.strokeStyle = "#0615e0";

            canvas_arrow(ctx, x0, y0, x0 + 100 * Math.cos(this.alpha[i]), y0 -30 * Math.sin(this.alpha[i]), 5)
            ctx.stroke();
        }

        


        


        ctx.beginPath();
        ctx.lineWidth = Math.sqrt(Math.abs(this.Mz_current)) * 7;
        ctx.strokeStyle = "#000000"
        ctx.setLineDash([]);
        canvas_arrow(ctx, x0, y0, x0, y0 - (60 * this.Mz_current), Math.sqrt(Math.abs(this.Mz_current)) * 7)
        ctx.stroke();


        if (this.rfon)
        {
        ctx.font = "12px Arial";
        ctx.fillStyle = "#000000"
        ctx.fillText("RF Pulse", 1, y0 - 15);
        ctx.beginPath();
        ctx.lineWidth = 2;
        canvas_arrow(ctx, 5, y0 - 10, 45, y0 - 10, 5)
        canvas_arrow(ctx, 5, y0, 45, y0, 5)
        canvas_arrow(ctx, 5, y0 + 10, 45, y0 + 10, 5)
        ctx.stroke();
        }

        if (this.rfon)
        {
            if (Math.abs(this.Mz_current) < 0.01 & is_alpha_synced(this.alpha))
                document.getElementById(this.rfpulseId).checked = false;
        }

        if ((this.showflip | this.showgrad) & !this.rfon & is_alpha_synced(this.alpha) & this.dT > 1)
        {
            ctx.font = "13px Arial";
            ctx.fillStyle = "#000000"
            ctx.fillText("Signal", 255, y0 - 5);
            ctx.beginPath();
            ctx.lineWidth = 2;
            canvas_arrow(ctx, 255, y0, 295, y0, 5)
            ctx.stroke();
        }
    }

    tick = (time) => {
        var rfon = document.getElementById(this.rfpulseId).checked;
        var dt = (time - this.old_time)/100;
        if (dt > 0.5)
        {
            this.old_time = time;
            this.draw(this.canvas, dt, rfon);
        }
        window.requestAnimationFrame(this.tick);
    }

}

class mri_gradients{
    constructor(canvasId, gradientId, pulseId) {
        this.canvas = document.getElementById(canvasId);
        if (this.canvas == null)
            return

        this.gradientId = gradientId;
        this.pulseId = pulseId;
        this.old_time = 0;
        this.state = 0;

        window.requestAnimationFrame(this.tick);
    }
    
    draw = (canvas, state) => {
        var ctx = canvas.getContext("2d");
        var slider = document.getElementById(this.gradientId);
        var grad = slider.value;

        var slider = document.getElementById(this.pulseId);
        var pulse = slider.value;

        
        // Create gradient
        var grd = ctx.createLinearGradient(0, 300, 0, 0);
        grd.addColorStop(0,  'rgb(' + (128 - grad) + ',' + (128 - grad) + ',' + (128 - grad) + ')');
        grd.addColorStop(1, 'rgb(' + (128 - (-grad)) + ',' + (128 - (-grad)) + ',' + (128 - (-grad)) + ')');
        ctx.fillStyle = grd;
        ctx.fillRect(0, 0, 300, 300);

        var x0 = state % 299;
        var pulse_width = 300 / ((grad + 1) / 40);
        var y0 = (300 - pulse_width) - pulse * (300 - pulse_width);

        ctx.globalAlpha = 0.1;
        ctx.fillStyle = "#ff0000";
        ctx.fillRect(0,y0,299,pulse_width);
        ctx.globalAlpha = 1.0;

        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#ff0000";
        ctx.lineTo(x0, y0 + pulse_width);
        ctx.stroke();
    }

    tick = (time) => {
        var dt = (time - this.old_time)/1000;
        this.old_time = time;
        var freq = 1000;
        this.state += freq * dt
        this.draw(this.canvas, this.state);
        window.requestAnimationFrame(this.tick);
    }

}

class mri_downsample{
    constructor(canvasId, PE, FE, labelId){
        this.canvas = document.getElementById(canvasId);
        if (this.canvas == null)
            return

        this.peId = PE;
        this.feId = FE;
        this.labelId = labelId;
        this.old_time = 0;
        this.state = 0;
        this.old_pe = 1;
        this.old_fe = 1;
        
        var slider_pe = document.getElementById(this.peId);
        slider_pe.tag = this;
        slider_pe.oninput = function() 
        {
            this.tag.tick();
            var label = document.getElementById(this.tag.labelId);
            label.innerHTML = (10 - this.value) + ' sec';
        }
        
        var slider_fe = document.getElementById(this.feId);
        slider_fe.tag = this;
        slider_fe.oninput = function() 
        {
            this.tag.tick();
        }

        this.tick();
        var label = document.getElementById(this.labelId);
        label.innerHTML = (10 - slider_pe.value) + ' sec';
    }

    draw = (canvas, pe, fe) => {
        var ctx = canvas.getContext("2d");

        var newImage = new Image();
        newImage.onload = () => {ctx.drawImage(newImage, 0, 0, 301, 150);}
        newImage.src = 'https://mlsatellite.com/wp-content/uploads/2023/01/kspace_' + pe + '_' + fe + '.png'
        return true;
    }

    tick() {
        var slider_pe = document.getElementById(this.peId);
        var pe = slider_pe.value;
		var slider_fe = document.getElementById(this.feId);
        var fe = slider_fe.value;
        if (this.draw(this.canvas, pe, fe))
        {
            this.old_pe = pe;
            this.old_fe = fe;
        }
    }

}

class mri_phaseencoding{
    constructor(canvasId, protonID1, protonID2, protonID3, protonID4, protonID5, frequencyID, buttonId){
        this.canvas = document.getElementById(canvasId);
        if (this.canvas == null)
            return

        this.buttonId = buttonId;
        this.protonIDs = [protonID1, protonID2, protonID3, protonID4, protonID5];
        this.frequencyID = frequencyID;
        this.Mxy = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        this.x0 = [30, 90, 150, 210, 270]
        this.dephase = [-5, -2.5, 0, 2.5, 5]
        this.width = 25;
        this.height = 6;
        this.isCollect = false;

        this.old_time = 0;
        this.state = 0;
        this.old_pe = 1;
        this.old_fe = 1;

        var button = document.getElementById(this.buttonId);
        button.tag = this;
        button.onclick = function() {
            var slider = document.getElementById(this.tag.frequencyID);
            slider.value = -1;
            this.tag.isCollect = true;
        }

        window.requestAnimationFrame(this.tick);
    }

    draw = (canvas, frequency) => {
        var ctx = canvas.getContext("2d");
        ctx.resetTransform();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Create gradient
        var baseline = 64
        var grad = baseline * ((frequency * 1) + 1)
        var grd = ctx.createLinearGradient(300, 0, 0, 0);
        grd.addColorStop(0,  'rgb(' + (192 - grad) + ',' + (192 - grad) + ',' + (192 - grad) + ')');
        grd.addColorStop(1, 'rgb(' + (64 - (-grad)) + ',' + (64 - (-grad)) + ',' + (64 - (-grad)) + ')');
        ctx.fillStyle = grd;
        ctx.fillRect(0, 220, 300, 30);

        var y0 = 235;
        var Mx = 0;
        var My = 0;
        var numel = 0;
        for (let i=0; i < 5; i++)
        {
            if (document.getElementById(this.protonIDs[i]).checked)
            {
                var alpha = this.dephase[i] * frequency
                ctx.beginPath();
                ctx.lineWidth = 1;
                ctx.setLineDash([10, 5]);
                ctx.ellipse(this.x0[i], y0, this.width, this.height, 0, 0, 2 * Math.PI);
                ctx.stroke();
                
                ctx.beginPath();
                ctx.setLineDash([]);
                ctx.lineWidth = 1;
                canvas_arrow(ctx, this.x0[i], y0, this.x0[i] + this.width * Math.cos(alpha), y0 - this.height * Math.sin(alpha), 5)
                ctx.stroke();

                Mx += Math.sin(alpha)
                My += Math.cos(alpha)
                numel += 1
            }
        }
        Mx /= numel
        My /= numel

        this.Mxy.shift();
        this.Mxy = this.Mxy.concat(Math.sqrt(Mx*Mx + My*My))

        

        // Plotting Mxy
        var x0 = 10;
        var y0 = 210;
        ctx.setLineDash([]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.strokeStyle = "#000000";
        ctx.moveTo(x0, y0 - (200 * this.Mxy[0]));
        for (let i=0; i < this.Mxy.length; i++)
        {
            ctx.lineTo(x0 + (i * 7), y0 - (200 * this.Mxy[i]));
        }
        ctx.stroke();

        canvas_arrow(ctx, x0, y0, x0 + 280, y0, 5)
        canvas_arrow(ctx, x0, y0, x0, y0 - 205, 5)
        ctx.stroke();

    }

    tick = (time) => {
        var dt = (time - this.old_time)/1000;
        var frequency = document.getElementById(this.frequencyID);
        var freq = frequency.value;

        if (dt > 1/24)
        {
            this.draw(this.canvas, freq);
            if (this.isCollect)
            {
                if (Number(freq) < Number(frequency.max))
                {
                    frequency.value = Number(freq) + Number('0.1');
                }
                else
                {
                    this.isCollect = false;
                }
            }
            this.old_time = time;
        }
        window.requestAnimationFrame(this.tick);
    }

}

class mlbasics_complexity{
    constructor(canvasId, labelId, complexityId){
        this.canvas = document.getElementById(canvasId);
        if (this.canvas == null)
            return

        this.labelId = labelId;
        this.complexityId = complexityId;


        var label = document.getElementById(this.labelId);
        var slider = document.getElementById(this.complexityId);
        label.innerHTML = slider.value;
        slider.tag = this;
        slider.oninput = function() 
        {
            this.tag.tick();
            label.innerHTML = this.value;
        }
        
        this.state = 0;
        this.old_time = 0;
        window.requestAnimationFrame(this.tick);
        this.Points =   [[ 0.        ,  1.18766179],
                    [ 0.10526316,  0.96502991],
                    [ 0.21052632,  1.32404951],
                    [ 0.31578947,  3.7230705 ],
                    [ 0.42105263,  3.81949993],
                    [ 0.52631579,  3.59271584],
                    [ 0.63157895,  6.80882445],
                    [ 0.73684211,  6.43150809],
                    [ 0.84210526,  7.5535498 ],
                    [ 0.94736842,  8.41866679],
                    [ 1.05263158,  9.458921  ],
                    [ 1.15789474,  8.97376778],
                    [ 1.26315789,  9.0455999 ],
                    [ 1.36842105, 11.14471495],
                    [ 1.47368421, 10.74706869],
                    [ 1.57894737, 11.96146858],
                    [ 1.68421053,  8.60756059],
                    [ 1.78947368,  9.9136143 ],
                    [ 1.89473684,  7.96472936],
                    [ 2.        ,  5.03372468]];
    }
    
    
    func(x)
    {
        var slider = document.getElementById(this.complexityId);
        var comp = slider.value;
        
        switch(comp)
        {
            case "1": 
                return  x * 4.19346686 + 2.64032046
            case "2": 
                return  x**2 * -5.44220456 + x**1 * 15.07787597 - 0.79686136
            case "3": 
                return  x**3 * -5.26580025 + x**2 * 10.35519621 + x**1 * 2.76086285 + 0.98855125
            case "4": 
                return  x**4 * -2.08915776 + x**3 * 3.09083078 + x**2 * -0.21708294 + x**1 * 7.19215909 + 0.64769302
            case "5": 
                return  x**5 * -5.45867444 + x**4 * 25.20421442 + x**3 * -44.89308485 + x**2 * 34.56117524 + x**1 * -1.82750158 + 1.03831339
            case "6": 
                return  x**6 * 2.4515465 + x**5 * -20.16795343 + x**4 * 58.36892808 + x**3 * -79.49007952 + x**2 * 50.97690234 + x**1 * -4.68971043 + 1.10882179
            case "7": 
                return  x**7 * -1.95453988 + x**6 * 16.13332565 + x**5 * -57.84022068 + x**4 * 109.91247281 + x**3 * -115.7996195 + x**2 * 63.18557657 + x**1 * -6.2080719 + 1.12953225
            case "8": 
                return  x**8 * 3.79003888e-05 + x**7 * -1.95484308e+00 + x**6 * 1.61343122e+01 + x**5 * -5.78418951e+01 + x**4 * 1.09914051e+02 + x**3 * -1.15800430e+02 + x**2 * 6.31857807e+01 + x**1 * -6.20809093e+00 + 1.12953238e+00
            case "9": 
                return  x**9 * -1.04963221e+02 + x**8 * 9.44669026e+02 + x**7 * -3.54932517e+03 + x**6 * 7.21390552e+03 + x**5 * -8.59345830e+03 + x**4 * 6.07792622e+03 + x**3 * -2.46938883e+03 + x**2 * 5.28195105e+02 + x**1 * -4.02462568e+01 + 1.24435867e+00
            case "10": 
                return  x**10 * -7.98023163e+01 + x**9 * 6.93059942e+02 + x**8 * -2.45289219e+03 + x**7 * 4.47860867e+03 + x**6 * -4.28769205e+03 + x**5 * 1.65220981e+03 + x**4 * 4.87526883e+02 + x**3 * -7.04445238e+02 + x**2 * 2.44041884e+02 + x**1 * -2.31608184e+01 + 1.22017554e+00
            case "11": 
                return  x**11 *   3.29727454e+02 + x**10 * -3.70680431e+03 + x**9 * 1.79597137e+04 + x**8 * -4.90427160e+04 + x**7 * 8.28498370e+04 + x**6 * -8.94338021e+04 + x**5 * 6.14990968e+04 + x**4 * -2.59985912e+04 + x**3 * 6.22212945e+03 + x**2 * -6.96373801e+02 + x**1 * 2.52625089e+01 + 1.19538338e+00
            case "12": 
                return  x**12 * 5.08376148e+02 + x**11 * -5.77078633e+03 + x**10 * 2.84087552e+04 + x**9 * -7.95103764e+04 + x**8 * 1.39432567e+05 + x**7 * -1.59360690e+05 + x**6 * 1.19931336e+05 + x**5 * -5.89727129e+04 + x**4 * 1.86221237e+04 + x**3 * -3.73036633e+03 + x**2 * 4.77749900e+02 + x**1 * -2.83619218e+01 + 1.20378058e+00
            case "13": 
                return  x**14 * -1.12908653e+03 + x**13 * 2.07376247e+04 + x**12 * -1.62760341e+05 + x**11 * 7.32091861e+05 + x**10 * -2.11747449e+06 + x**9 * 4.16593624e+06 + x**8 * -5.73020725e+06 + x**7 * 5.56192964e+06 + x**6 * -3.78938519e+06 + x**5 * 1.77572868e+06 + x**4 * -5.50033758e+05 + x**3 * 1.04898806e+05 + x**2 * -1.07588435e+04 + x**1 * 4.33789265e+02 + 1.18759341e+00
            case "14": 
                return  x**15 * 1.01533568e+04 + x**14 * -1.53429438e+05 + x**13 * 1.05326874e+06 + x**12 * -4.34611015e+06 + x**11 * 1.20166013e+07 + x**10 * -2.34877682e+07 + x**9 * 3.33572265e+07 + x**8 * -3.48209491e+07 + x**7 * 2.67060030e+07 + x**6 * -1.48749714e+07 + x**5 * 5.87187236e+06 + x**4 * -1.57488982e+06 + x**3 * 2.67125897e+05 + x**2 * -2.50799527e+04 + x**1 * 9.54532073e+02 + 1.18687699e+00
        }
    }
    draw = (canvas, state) => {
        var ctx = canvas.getContext("2d");

        ctx.resetTransform();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.beginPath();
        ctx.lineWidth = 1;
        ctx.strokeStyle = "#000000";
        ctx.rect(0, 0, canvas.width, canvas.height);
        ctx.stroke();

        ctx.fillStyle = "#ff4e33";
        this.Points.forEach((point, index, arr) => 
        {
            ctx.beginPath();
            ctx.arc(150 * point[0], (canvas.height - (20 * point[1])), 4, 0, 2 * Math.PI);
            ctx.fill();
        });

        ctx.beginPath();
        ctx.moveTo(150 * 0, canvas.height - (20 * this.func(0)));
        ctx.lineWidth = 4;
        ctx.strokeStyle = "#88a9f2";
        for (let i = 0; i < 2; i+= 0.01) 
        {
            ctx.lineTo(150 * i, canvas.height - (20 * this.func(i)));
            ctx.stroke();
            ctx.moveTo(150 * i, canvas.height - (20 * this.func(i)));
        }
        ctx.stroke();
    }

    tick = (time) => {
        this.dt = (time - this.old_time)/1000;
        this.old_time = time;
        var freq = 1;
        this.state += freq * this.dt
        this.draw(this.canvas, this.state);
    }
}


class mlbasics_complexity_decomp{
    constructor(canvasId, labelId, complexityId){
        this.canvas = document.getElementById(canvasId);
        if (this.canvas == null)
            return

        this.labelId = labelId;
        this.complexityId = complexityId;

        var label = document.getElementById(this.labelId);
        var slider = document.getElementById(this.complexityId);
        label.innerHTML = slider.value;
        slider.tag = this;
        slider.oninput = function() 
        {
            this.tag.tick();
            label.innerHTML = this.value;
        }

        this.state = 0;
        this.old_time = 0;
        window.requestAnimationFrame(this.tick);

        

        this.Points =   [[ 0.        ,  1.18766179],
                    [ 0.10526316,  0.96502991],
                    [ 0.21052632,  1.32404951],
                    [ 0.31578947,  3.7230705 ],
                    [ 0.42105263,  3.81949993],
                    [ 0.52631579,  3.59271584],
                    [ 0.63157895,  6.80882445],
                    [ 0.73684211,  6.43150809],
                    [ 0.84210526,  7.5535498 ],
                    [ 0.94736842,  8.41866679],
                    [ 1.05263158,  9.458921  ],
                    [ 1.15789474,  8.97376778],
                    [ 1.26315789,  9.0455999 ],
                    [ 1.36842105, 11.14471495],
                    [ 1.47368421, 10.74706869],
                    [ 1.57894737, 11.96146858],
                    [ 1.68421053,  8.60756059],
                    [ 1.78947368,  9.9136143 ],
                    [ 1.89473684,  7.96472936],
                    [ 2.        ,  5.03372468]];
    }
    

    func(x)
    {
        var slider = document.getElementById(this.complexityId);
        var comp = slider.value;
        
        switch(comp)
        {
            case "1": 
                return  x * 4.19346686 + 2.64032046
            case "2": 
                return  x**2 * -5.44220456 + x**1 * 15.07787597 - 0.79686136
            case "3": 
                return  x**3 * -5.26580025 + x**2 * 10.35519621 + x**1 * 2.76086285 + 0.98855125
            case "4": 
                return  x**4 * -2.08915776 + x**3 * 3.09083078 + x**2 * -0.21708294 + x**1 * 7.19215909 + 0.64769302
            case "5": 
                return  x**5 * -5.45867444 + x**4 * 25.20421442 + x**3 * -44.89308485 + x**2 * 34.56117524 + x**1 * -1.82750158 + 1.03831339
            case "6": 
                return  x**6 * 2.4515465 + x**5 * -20.16795343 + x**4 * 58.36892808 + x**3 * -79.49007952 + x**2 * 50.97690234 + x**1 * -4.68971043 + 1.10882179
            case "7": 
                return  x**7 * -1.95453988 + x**6 * 16.13332565 + x**5 * -57.84022068 + x**4 * 109.91247281 + x**3 * -115.7996195 + x**2 * 63.18557657 + x**1 * -6.2080719 + 1.12953225
            case "8": 
                return  x**8 * 3.79003888e-05 + x**7 * -1.95484308e+00 + x**6 * 1.61343122e+01 + x**5 * -5.78418951e+01 + x**4 * 1.09914051e+02 + x**3 * -1.15800430e+02 + x**2 * 6.31857807e+01 + x**1 * -6.20809093e+00 + 1.12953238e+00
            case "9": 
                return  x**9 * -1.04963221e+02 + x**8 * 9.44669026e+02 + x**7 * -3.54932517e+03 + x**6 * 7.21390552e+03 + x**5 * -8.59345830e+03 + x**4 * 6.07792622e+03 + x**3 * -2.46938883e+03 + x**2 * 5.28195105e+02 + x**1 * -4.02462568e+01 + 1.24435867e+00
            case "10": 
                return  x**10 * -7.98023163e+01 + x**9 * 6.93059942e+02 + x**8 * -2.45289219e+03 + x**7 * 4.47860867e+03 + x**6 * -4.28769205e+03 + x**5 * 1.65220981e+03 + x**4 * 4.87526883e+02 + x**3 * -7.04445238e+02 + x**2 * 2.44041884e+02 + x**1 * -2.31608184e+01 + 1.22017554e+00
            case "11": 
                return  x**11 *   3.29727454e+02 + x**10 * -3.70680431e+03 + x**9 * 1.79597137e+04 + x**8 * -4.90427160e+04 + x**7 * 8.28498370e+04 + x**6 * -8.94338021e+04 + x**5 * 6.14990968e+04 + x**4 * -2.59985912e+04 + x**3 * 6.22212945e+03 + x**2 * -6.96373801e+02 + x**1 * 2.52625089e+01 + 1.19538338e+00
            case "12": 
                return  x**12 * 5.08376148e+02 + x**11 * -5.77078633e+03 + x**10 * 2.84087552e+04 + x**9 * -7.95103764e+04 + x**8 * 1.39432567e+05 + x**7 * -1.59360690e+05 + x**6 * 1.19931336e+05 + x**5 * -5.89727129e+04 + x**4 * 1.86221237e+04 + x**3 * -3.73036633e+03 + x**2 * 4.77749900e+02 + x**1 * -2.83619218e+01 + 1.20378058e+00
            case "13": 
                return  x**14 * -1.12908653e+03 + x**13 * 2.07376247e+04 + x**12 * -1.62760341e+05 + x**11 * 7.32091861e+05 + x**10 * -2.11747449e+06 + x**9 * 4.16593624e+06 + x**8 * -5.73020725e+06 + x**7 * 5.56192964e+06 + x**6 * -3.78938519e+06 + x**5 * 1.77572868e+06 + x**4 * -5.50033758e+05 + x**3 * 1.04898806e+05 + x**2 * -1.07588435e+04 + x**1 * 4.33789265e+02 + 1.18759341e+00
            case "14": 
                return  x**15 * 1.01533568e+04 + x**14 * -1.53429438e+05 + x**13 * 1.05326874e+06 + x**12 * -4.34611015e+06 + x**11 * 1.20166013e+07 + x**10 * -2.34877682e+07 + x**9 * 3.33572265e+07 + x**8 * -3.48209491e+07 + x**7 * 2.67060030e+07 + x**6 * -1.48749714e+07 + x**5 * 5.87187236e+06 + x**4 * -1.57488982e+06 + x**3 * 2.67125897e+05 + x**2 * -2.50799527e+04 + x**1 * 9.54532073e+02 + 1.18687699e+00
        }
    }

    true_func(x)
    {
        return  x**4 * -2 + x**3 * 2 + x**2 * 3 + x**1 * 4 + 1
    }

    draw = (canvas, state) => {
        var ctx = canvas.getContext("2d");

        ctx.resetTransform();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.beginPath();
        ctx.lineWidth = 1;
        ctx.strokeStyle = "#000000";
        ctx.rect(0, 0, canvas.width, canvas.height);
        ctx.stroke();

        ctx.fillStyle = "#ff4e33";
        this.Points.forEach((point, index, arr) => 
        {
            ctx.beginPath();
            ctx.arc(150 * point[0], (canvas.height - (20 * point[1])), 4, 0, 2 * Math.PI);
            ctx.fill();
        });

        ctx.beginPath();
        ctx.moveTo(150 * 0, canvas.height - (20 * this.true_func(0)));
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#ff0000";
        for (let i = 0; i < 2; i+= 0.01) 
        {
            ctx.lineTo(150 * i, canvas.height - (20 * this.true_func(i)));
            ctx.stroke();
            ctx.moveTo(150 * i, canvas.height - (20 * this.true_func(i)));
        }
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(150 * 0, canvas.height - (20 * this.func(0)));
        ctx.lineWidth = 4;
        ctx.strokeStyle = "#88a9f2";
        for (let i = 0; i < 2; i+= 0.01) 
        {
            ctx.lineTo(150 * i, canvas.height - (20 * this.func(i)));
            ctx.stroke();
            ctx.moveTo(150 * i, canvas.height - (20 * this.func(i)));
        }
        ctx.stroke();


        ctx.fillStyle = "#000000";
        ctx.font = "15px Arial";
        ctx.fillText("Irr. error:", 7, 15);
        ctx.fillText("Bias:", 7, 30);
        ctx.fillText("Variance:", 7, 45);


        var errors = this.get_errors();
        ctx.beginPath();
        ctx.lineWidth = 1;
        ctx.fillStyle = "#ff0000";
        ctx.fillRect(75, 4, errors[0], 12);
        ctx.stroke();

        ctx.beginPath();
        ctx.lineWidth = 1;
        ctx.fillStyle = "#ff0000";
        ctx.fillRect(75, 19, errors[1], 12);
        ctx.stroke();

        ctx.beginPath();
        ctx.lineWidth = 1;
        ctx.fillStyle = "#ff0000";
        ctx.fillRect(75, 34, errors[2], 12);
        ctx.stroke();

    }

    get_errors(x)
    {
        var slider = document.getElementById(this.complexityId);
        var comp = slider.value;
        var x = Number(comp)

        return [20, 80 - 21 * Math.sqrt(x), x*x]
    }

    tick = (time) => {
        this.dt = (time - this.old_time)/1000;
        this.old_time = time;
        var freq = 1;
        this.state += freq * this.dt
        this.draw(this.canvas, this.state);

        
        // window.requestAnimationFrame(this.tick);
    }
}

class mlbasics_gradients{
    constructor(canvasId, gradientId, labelId, posId){
        this.canvas = document.getElementById(canvasId);
        if (this.canvas == null)
            return

        this.gradientId = gradientId;
        this.labelId = labelId;
        this.posId = posId;
        this.state = 0;
        this.old_time = 0;
        
        this.alpha0 = 10
        this.alpha1 =  -1.86666667e-01
        this.alpha2 =  2.13333333e-03 
        this.alpha3 =  -8.83333333e-06 
        this.alpha4 = 1.16666667e-08  
        
        var slider = document.getElementById(this.gradientId);
        slider.tag = [this.alpha0, this.alpha1, this.alpha2, this.alpha3, this.alpha4, this.tick]
        var x0 = slider.value
        var labelgrad = document.getElementById(this.labelId);
        labelgrad.innerHTML =  Math.round(10000 * (4 * this.alpha4 * x0**3 + 3 * this.alpha3 * x0**2 + 2 * this.alpha2 * x0**1 + this.alpha1)) / 100
        var labelpos = document.getElementById(this.posId);
        labelpos.innerHTML = slider.value

        slider.oninput = function() 
        {
            x0 = slider.value  * (4/3)
            labelgrad.innerHTML = Math.round(10000 * (4 * this.tag[4] * x0**3 + 3 * this.tag[3] * x0**2 + 2 * this.tag[2] * x0**1 + this.tag[1])) / 100
            labelpos.innerHTML = (slider.value  * (4/3)).toFixed(0);
            slider.tag[5]();
        }

        window.requestAnimationFrame(this.tick);
    }

    f(x)
    {
        return  this.alpha4 * x**4 + this.alpha3 * x**3 + this.alpha2 * x**2 + this.alpha1 * x**1 + this.alpha0
    }
    g(x)
    {
        var slider = document.getElementById(this.gradientId);
        var x0 = slider.value * (4/3);
        return  (x - x0) * Math.round((4 * this.alpha4 * x0**3 + 3 * this.alpha3 * x0**2 + 2 * this.alpha2 * x0**1 + this.alpha1) * 500) / 500
    }
    draw = (canvas, state) => {
        var ctx = canvas.getContext("2d");
        var slider = document.getElementById(this.gradientId);
        var x0 = slider.value;

        ctx.resetTransform();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.beginPath();
        ctx.strokeStyle = "#000000";
        ctx.rect(0, 0, canvas.width, canvas.height);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(0, canvas.height - (30 * this.f(0)));
        ctx.lineWidth = 4;
        ctx.strokeStyle = "#88a9f2";
        for (let i = 0; i < 300; i++) 
        {
            var pos = i * (4 / 3)
            ctx.lineTo(i, canvas.height - (30 * this.f(pos)));
            ctx.stroke();
            ctx.moveTo(i, canvas.height - (30 * this.f(pos)));
        }
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(0, canvas.height - (30 * (this.f(x0 * (4 / 3)) + this.g(0))));
        ctx.lineWidth = 3;
        if (this.g(0) < 0)
            ctx.strokeStyle = "#820b00";
        else
            ctx.strokeStyle = "#00026e";
        ctx.lineTo(x0, canvas.height - (30 * (this.f(x0 * (4 / 3)) + this.g(x0 * (4 / 3)))));
        ctx.stroke();
        ctx.beginPath();
        ctx.lineWidth = 2;
        if (this.g(0) > 0)
            ctx.strokeStyle = "#820b00";
        else
            ctx.strokeStyle = "#00026e";
        ctx.moveTo(x0, canvas.height - (30 * (this.f(x0 * (4/3)) + this.g(x0 * (4/3)))));


        ctx.lineTo(canvas.width, canvas.height - (30 * (this.f(x0 * (4/3)) + this.g(400))));
        ctx.stroke();
        ctx.moveTo(canvas.width, canvas.height - (30 * (this.f(x0 * (4/3)) + this.g(400))));
        ctx.stroke();
        
        ctx.fillStyle = "#ff4e33";
        ctx.beginPath();
        ctx.arc(x0, (canvas.height - (30 * this.f(x0 * (4/3)))), 5, 0, 2 * Math.PI);
        ctx.fill();
    }

    
    tick = (time) => {
        this.dt = (time - this.old_time)/1000;
        this.old_time = time;
        var freq = 1;
        this.state += freq * this.dt
        this.draw(this.canvas, this.state);
        // window.requestAnimationFrame(this.tick);
    }
}

class mlbasics_momentum{
    constructor(canvasId, numIterId, learningRateId, momentumId, labelId)
    {
        this.canvas = document.getElementById(canvasId);
        if (this.canvas == null)
            return
        this.numIterId = numIterId;
        this.learningRateId = learningRateId;
        this.momentumId = momentumId;
        this.labelId = labelId;

        this.state = 0;
        this.mom = 0;
        
        this.alpha0 = 10
        this.alpha1 =  -1.86666667e-01
        this.alpha2 =  2.13333333e-03 
        this.alpha3 =  -8.83333333e-06 
        this.alpha4 = 1.16666667e-08  

        this.x0 = 10
        window.requestAnimationFrame(this.tick);

        var slider_numIter = document.getElementById(numIterId);
        slider_numIter.tag = this;
        this.numIter = slider_numIter.value;
        slider_numIter.oninput = function() 
        {
            this.tag.numIter = slider_numIter.value;
            this.tag.tick();
        }
        
        var slider_learningRate = document.getElementById(learningRateId);
        slider_learningRate.tag = this;
        this.lr = slider_learningRate.value;
        slider_learningRate.oninput = function() 
        {
            this.tag.lr = slider_learningRate.value;
            this.tag.tick();
        }
        
        if (momentumId == null)
        {
            this.momentum = 0;
        }
        else
        {
            var slider_momentum = document.getElementById(momentumId);
            slider_momentum.tag = this;
            this.momentum = slider_momentum.value;
            slider_momentum.oninput = function() 
            {
                this.tag.momentum = slider_momentum.value;
                this.tag.tick();
            }
        }
        
    }

    gradient_step(x, lr)
    {
        return  lr * (4 * this.alpha4 * x**3 + 3 * this.alpha3 * x**2 + 2 * this.alpha2 * x**1 + this.alpha1)
    }
    f(x)
    {
        return  this.alpha4 * x**4 + this.alpha3 * x**3 + this.alpha2 * x**2 + this.alpha1 * x**1 + this.alpha0
    }
    
    calculate_loss(numiter, lr)
    {
        this.x0 = 10
        var change_x = 0;
        for (let i = 0; i < numiter; i++) 
        {
            change_x = this.gradient_step(this.x0, lr) + this.momentum * change_x;
            this.x0 = this.x0 - change_x;
        }
        return this.f(this.x0).toFixed(2);
    }
    draw = (canvas, state) => {
        var ctx = canvas.getContext("2d");
        var change_x = 0;
        var x0 = 10;
        var x = x0;

        ctx.resetTransform();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.beginPath();
        ctx.strokeStyle = "#000000";
        ctx.rect(0, 0, canvas.width, canvas.height);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(0, canvas.height - (30 * this.f(0)));
        ctx.lineWidth = 4;
        ctx.strokeStyle = "#88a9f2";
        for (let i = 0; i < canvas.width; i++) 
        {
            ctx.lineTo(i, canvas.height - (30 * this.f(i * (4/3))));
            ctx.stroke();
            ctx.moveTo(i, canvas.height - (30 * this.f(i * (4/3))));
        }
        ctx.stroke();

        ctx.beginPath();
        ctx.lineWidth = 3;
        ctx.fillStyle = "#ff4e33";
        ctx.arc(x0, (canvas.height - (30 * this.f(x0 * (4/3)))), 5, 0, 2 * Math.PI);
        ctx.fill();
        for (let i = 0; i < this.numIter; i++) 
        {
            ctx.beginPath();
            ctx.strokeStyle = "#ff4e33";
            ctx.moveTo(x0, (canvas.height - (30 * this.f(x0 * (4/3)))));
            ctx.stroke();
            change_x = this.gradient_step(x0 * (4/3), this.lr) + this.momentum * change_x;
            x0 = x0 - change_x;
            ctx.lineTo(x0, (canvas.height - (30 * this.f(x0 * (4/3)))));
            ctx.stroke();
            ctx.beginPath();
            ctx.fillStyle = "#ff4e33";
            ctx.arc(x0, (canvas.height - (30 * this.f(x0 * (4/3)))), 5, 0, 2 * Math.PI);
            ctx.fill();
        }
    }

    old_time = 0;
    tick = (time) => {
        var losslabel = document.getElementById(this.labelId);
        losslabel.innerHTML = this.f(this.x0 * (4/3)).toFixed(2);
        losslabel.innerHTML = this.calculate_loss(this.numIter, this.lr);

        this.dt = (time - this.old_time)/1000;
        this.old_time = time;
        var freq = 1;
        this.state += freq * this.dt

        this.draw(this.canvas, this.state);
    }
}

class mlbasics_forwardpass{
    constructor(canvasId, input1Id, input2Id, w1Id, w2Id, w3Id, outputId, pauseId, total_time, T)
    {
        this.canvas = document.getElementById(canvasId);
        if (this.canvas == null)
            return
        this.input1Id = input1Id;
        this.input2Id = input2Id;
        this.pauseId = pauseId;
        
        this.timepaused = false;
        this.reset = false;

        this.w1Id = w1Id;
        this.w2Id = w2Id;
        this.w3Id = w3Id;
        this.outputId = outputId;
        this.total_time = total_time;
        this.length = T;
        this.precision = 2;
        this.cost = 0;

        this.time = 0;
        this.dt = 0;
        this.old_time = 0;

        this.updatew1w2 = false;
        this.updatew3 = false;
        this.updateCost = false;
        
        
        
        var input = document.getElementById(this.pauseId);
        input.tag = this;
        input.onclick = this.toggle;

        var input = document.getElementById(this.input1Id);
        input.tag = this;
        input.oninput = this.input_check;
        
        input = document.getElementById(this.input2Id);
        input.tag = this;
        input.oninput = this.input_check;
        
        if (this.w1Id != null)
        {
            input = document.getElementById(this.w1Id);
            input.tag = this;
            input.oninput = this.input_check;
            
            input = document.getElementById(this.w2Id);
            input.tag = this;
            input.oninput = this.input_check;
            
            input = document.getElementById(this.w3Id);
            input.tag = this;
            input.oninput = this.input_check;
        }
        
        this.lr = 0.0
        if (this.outputId != null)
        {
            var input = document.getElementById(this.outputId);
            input.tag = this;
            input.oninput = this.input_check;
            this.lr = 0.1;
        }

        this.update_values();
        window.requestAnimationFrame(this.tick);
    }

    input_check()
    {
        if (Number(this.value) >= Number(this.max))
            this.value = this.max;
        if (Number(this.value) < Number(this.min))
            this.value = this.min;
        this.tag.update_values();
    }

    toggle()
    {
        this.tag.toggle_time();
    }

    toggle_time()
    {
        this.timepaused = !this.timepaused;

        if (!this.timepaused)
        {
            this.reset_round();
        }
    }

    update_values()
    {

        if (this.w1Id != null)
        {
            this.w1 = document.getElementById(this.w1Id).value;
            this.w2 = document.getElementById(this.w2Id).value;
            this.w3 = document.getElementById(this.w3Id).value;
        }
        else
        {
            this.w1 = 0.0;
            this.w2 = 1.0;
            this.w3 = 0.5;
        }
        if (document.getElementById(this.outputId) != null)
            this.outp = document.getElementById(this.outputId).value;
        else
            this.outp = null;

        this.reset_round();
    }

    reset_round()
    {
        this.updatew1w2 = true;
        this.updatew3 = true;
        this.updateCost = true;
        this.reset = true;
    }
    
    draw = (canvas, phase, inp1, inp2, w1, w2, w3, outp, time) => {
        var ctx = canvas.getContext("2d");
        ctx.reset();
        if (this.updateCost)
        {
            this.y_pred = (w3 * (inp1 + w1)) + ((1 - w3) * (inp2 * w2));
            this.cost = ((outp - this.y_pred)**2);
            this.updateCost = false;
            this.a1 = (inp1 + w1);
            this.a2 = (inp2 * w2);
            this.a3 = this.y_pred
        }
        var cost_thr = 0.005;
        ctx.font = "8px Arial";
        ctx.textAlign = "center";

        ctx.beginPath();
        ctx.rect(0, 0, 300, 150);
        ctx.stroke();

        ctx.beginPath();
        ctx.rect(3, 55, 35, 15);
        ctx.stroke();
        ctx.textAlign = "center";
        ctx.fillText("x1=" + inp1.toFixed(this.precision), 20.5, 67);
        
        ctx.beginPath();
        ctx.rect(3, 80, 35, 15);
        ctx.stroke();
        ctx.textAlign = "center";
        ctx.fillText("x2=" + inp2.toFixed(this.precision), 20.5, 92);

        if (this.outputId != null)
        {
            ctx.beginPath();
            ctx.rect(262, 3, 35, 15);
            ctx.stroke();
            ctx.textAlign = "center";
            ctx.fillText("Y=" + outp.toFixed(this.precision), 279.5, 15);


            
            ctx.textAlign = "right";
            ctx.fillText("LR=" + this.lr, 297, 148);
            ctx.textAlign = "center";
        }

        if (this.state == 6 && this.outputId != null && this.cost > cost_thr)
        {
            ctx.fillStyle = "#FF0000";
        }
        ctx.textAlign = "center";
        //ctx.font = "12px Arial";
        ctx.fillText("w1 = " + w1.toFixed(this.precision), 110, 48);
        ctx.fillStyle = "#000000";
        ctx.font = "8px Arial";

        
        if (this.state == 6 && this.outputId != null && this.cost > cost_thr)
        {
            ctx.fillStyle = "#FF0000";
        }
        ctx.textAlign = "center";
        //ctx.font = "12px Arial";
        ctx.fillText("w2 = " + w2.toFixed(this.precision), 110, 83);
        ctx.fillStyle = "#000000";
        ctx.font = "8px Arial";

        


        if (this.state == 5 && this.outputId != null && this.cost > cost_thr)
        {
            ctx.fillStyle = "#FF0000";
        }
        ctx.textAlign = "center";
        //ctx.font = "12px Arial";
        ctx.fillText("w3 = " + w3.toFixed(this.precision), 200, 66);
        ctx.fillStyle = "#000000";
        ctx.font = "8px Arial";
        



        


        // Writing A1 & A2
        if (this.state < 1)
        {
            var a1text = "a1";
            var a2text = "a2";
            var color = "#000000";
        }
        else if (this.state == 1)
        {
            var a1text = "a1 = " + this.a1.toFixed(this.precision);
            var a2text = "a2 = " + this.a2.toFixed(this.precision);
            if (this.state == 1)
                var color = "#FF0000";
        }
        else
        {
            var a1text = "a1 = " + this.a1.toFixed(this.precision);
            var a2text = "a2 = " + this.a2.toFixed(this.precision);
        }
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.rect(90, 50, 40, 15);
        ctx.rect(90, 85, 40, 15);
        ctx.fillStyle = color;
        ctx.fillText(a1text, 110, 62);
        ctx.fillText(a2text, 110, 97);
        canvas_arrow(ctx, 38, 63, 90, 57, 5);
        canvas_arrow(ctx, 38, 88, 90, 92, 5);
        ctx.stroke();
        ctx.strokeStyle = "#000000";
        ctx.fillStyle = "#000000";


        // Writing A3
        if (this.state >= 2)
        {
            var a3text = "a3 = " + this.y_pred.toFixed(this.precision);
            if (this.state == 2)
                var color = "#FF0000";
        }
        else
        {
            var a3text = "a3";
            var color = "#000000";
        }
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.rect(180, 67.5, 40, 15);
        ctx.fillStyle = color;
        ctx.fillText(a3text, 200, 78.5);
        canvas_arrow(ctx, 130, 57, 180, 72, 5);
        canvas_arrow(ctx, 130, 92, 180, 77, 5);
        ctx.stroke();
        ctx.strokeStyle = "#000000";
        ctx.fillStyle = "#000000";

        // Writing y_pred
        var costText = "";
        if (this.state >= 3)
        {
            var outputText = "y=" + this.y_pred.toFixed(this.precision);
            if (this.state == 3)
                var color = "#FF0000";
            if (this.outputId != null) 
                costText = "L (MSE) = " + this.cost.toFixed(this.precision);
        }
        else
        {
            var outputText = "y";
            var color = "#000000";
        }
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.rect(262, 67.5, 35, 15);
        ctx.fillStyle = color;
        ctx.fillText(outputText, 279.5, 78.5);
        ctx.font = "bold 10px Arial";
        ctx.textAlign = "right";
        ctx.fillText(costText, 298, 45);
        ctx.textAlign = "center";
        ctx.font = "8px Arial";
        canvas_arrow(ctx, 220, 75, 262, 75, 5);
        ctx.stroke();
        ctx.strokeStyle = "#000000";
        ctx.fillStyle = "#000000";

        
        if (this.outputId != null && this.cost > cost_thr)
        {
            //ctx.font = "12px Arial";
            if (this.state < 4)
            {
                var color = "#000000";
                var dLdyText = "";
            }
            else 
            {
                if (this.state == 4)
                    var color = "#FF0000";
                else
                    var color = "#000000";
                var dLdyText = "dL / dy = 2 ( a3 - Y)";
            }
            ctx.beginPath();
            ctx.fillStyle = color;
            ctx.textAlign = "right";
            ctx.fillText(dLdyText, 295, 55);
            ctx.stroke();
            ctx.strokeStyle = "#000000";
            ctx.fillStyle = "#000000";

            var dw3 = (this.a1-this.a2)*2*(this.a3-outp);
            if (this.state >= 5)
            {
                if (this.updatew3)
                {
                    this.w3 = this.w3 - this.lr*dw3;
                    this.updatew3 = false;
                }
                
                if (this.state == 5)
                    var color = "#FF0000";
                else
                    var color = "#000000";
                
                ctx.beginPath();
                ctx.fillStyle = color;
                ctx.textAlign = "left";
                ctx.fillText("dw3 = " + (-this.lr*dw3).toFixed(5), 175, 90);
                ctx.stroke();
                ctx.strokeStyle = "#000000";
                ctx.fillStyle = "#000000";
            }

            
            var dw1 = 1 * w3 * 2*(this.a3-outp);
            var dw2 = inp1 * (1-w3) * 2*(this.a3-outp);
            if (this.state >= 6)
            {
                if (this.updatew1w2)
                {
                    this.w1 = this.w1 - this.lr*dw1;
                    this.w2 = this.w2 - this.lr*dw2;
                    this.updatew1w2 = false;
                }
                
                if (this.state == 6)
                    var color = "#FF0000";
                else
                    var color = "#000000";
                
                ctx.beginPath();
                ctx.fillStyle = color;
                ctx.textAlign = "left";
                ctx.fillText("dw1 = " + (-this.lr*dw1).toFixed(5), 85, 73);

                
                ctx.fillText("dw2 = " + (-this.lr*dw2).toFixed(5), 85, 107);
                ctx.stroke();
                ctx.strokeStyle = "#000000";
                ctx.fillStyle = "#000000";
            }
        }
    }

    tick = (time) => {
        if (this.timepaused)
        {
            setTimeout(function() {  }, 100);
        }
        else
        {
            this.time = time;
            if (this.reset)
            {
                this.old_time = this.time;
                this.reset = false;
            }
            var input1 = document.getElementById(this.input1Id).value;
            var input2 = document.getElementById(this.input2Id).value;
    
            var freq = 1 / Number(this.length);
            this.dt = freq * (this.time - this.old_time);
            if (this.dt >= this.total_time)
            {
                this.reset_round();
            }
            this.state = Math.floor(this.dt % this.total_time);
    
            this.draw(this.canvas, this.state, Number(input1), Number(input2), Number(this.w1), Number(this.w2), Number(this.w3), Number(this.outp), this.dt % this.total_time);
        }
        window.requestAnimationFrame(this.tick);
    }
}

class mlbasics_receptivefield{
    constructor(canvasId, kernelId, depthId, fieldLabelId)
    {
        this.canvas = document.getElementById(canvasId);
        if (this.canvas == null)
            return
        
        this.kernelId = kernelId;
        this.depthId = depthId;
        this.fieldLabelId = fieldLabelId;

        
        var slider = document.getElementById(this.kernelId);
        slider.tag = this;
        slider.oninput = function() 
        {
            this.tag.tick();
        }
        
        var slider = document.getElementById(this.depthId);
        slider.tag = this;
        slider.oninput = function() 
        {
            this.tag.tick();
        }
        this.tick();
    }

    tick()
    {
        this.kernel = document.getElementById(this.kernelId).value;
        var depth = document.getElementById(this.depthId).value;
        var fieldLabel = document.getElementById(this.fieldLabelId).innerHTML = ((Number(this.kernel - 1) * Number(depth)) + 1).toFixed(0) + " pixels";
        this.draw(this.canvas, Number(this.kernel), Number(depth))
    }

    draw(canvas, kernel, depth)
    {
        var ctx = canvas.getContext("2d");
        this.focusedNodeColor = "#FF0000";
        this.width = 12;
        this.num_blocks = canvas.width / this.width;
        this.center = Math.floor(this.num_blocks/2);
        ctx.strokeStyle="#000000";
        ctx.lineWidth = 1;
        ctx.reset();

        this.draw_nodes(ctx, canvas.height - 15, 1, true);

        if (depth > 0)
        {
            this.draw_nodes(ctx, canvas.height - 55, (1*(kernel-1)) + 1, depth > 1);
        }

        if (depth > 1)
        {
            this.draw_nodes(ctx, canvas.height - 95, (2*(kernel-1)) + 1, depth > 2);
        }
        
        if (depth > 2)
        {
            this.draw_nodes(ctx, canvas.height - 135, (3*(kernel-1)) + 1, depth > 3);
        }
        
        if (depth > 3)
        {
            this.draw_nodes(ctx, canvas.height - 175, (4*(kernel-1)) + 1, false);
        }
    }
    
    draw_nodes(ctx, y0, field_size, draw_paths)
    {
        ctx.beginPath();
        ctx.fillStyle = this.focusedNodeColor;
        for (let i = 0; i < this.num_blocks; i++)
        {  
            if (Math.abs(i - this.center) < Math.ceil(field_size / 2))
            {
                ctx.fillRect(i*this.width, y0, this.width, this.width);
            }
        }
        ctx.stroke();
        
        if (draw_paths)
        {
            for (let i = 0; i < this.num_blocks; i++)
            {  
                if (Math.abs(i - this.center) < Math.ceil(field_size / 2))
                {
                    ctx.beginPath();
                    ctx.fillStyle = "#000000";
                    ctx.moveTo(i*this.width, y0)
                    ctx.lineTo((i - Math.floor(this.kernel / 2))*this.width, y0 - (40 - this.width));
                    ctx.stroke();
                    
                    ctx.beginPath();
                    ctx.fillStyle = "#000000";
                    ctx.moveTo((i+1)*this.width, y0)
                    ctx.lineTo(((i+1) + Math.floor(this.kernel / 2))*this.width, y0 - (40 - this.width));
                    ctx.stroke();
                }
            }
        }

        ctx.beginPath();
        ctx.fillStyle = "#000000";
        for (let i = 0; i < this.num_blocks; i++)
        {
            ctx.rect(i*this.width, y0, this.width, this.width);
        }
        ctx.stroke();
    }
}

class mlbasics_convolution{
    constructor(canvasId, kernelsId, sliderId)
    {
        this.canvas = document.getElementById(canvasId);
        if (this.canvas == null)
            return
        
        this.kernelsId = kernelsId;
        this.sliderId = sliderId;
        
        var slider = document.getElementById(this.kernelsId);
        slider.tag = this;
        slider.onchange = function() 
        {
            this.tag.tick();
        }
        
        var slider = document.getElementById(this.sliderId);
        slider.tag = this;
        slider.oninput = function() 
        {
            this.tag.tick();
        }
        this.tick();
    }
    tick()
    {
        this.kernel = document.getElementById(this.kernelsId).value;
        var iteration = document.getElementById(this.sliderId).value;
        this.draw(this.canvas, this.kernel, Number(iteration))
    }

    draw(canvas, kernel, idx)
    {
        var ctx = canvas.getContext("2d");
        if (kernel == "eye")
        {
            var kernel = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
            var min = 0
            var max = 2
        }
        if (kernel == "gauss")
        {
            var kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
            var min = 9
            var max = 25
        }
        if (kernel == "laplace")
        {
            var kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
            var min = -2
            var max = 2
        }
        if (kernel == "sobel")
        {
            var kernel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
            var min = -5
            var max = 5
        }
        
        var img = [[0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 2, 2, 1, 1, 0],
                   [0, 1, 1, 2, 2, 1, 1, 0],
                   [0, 1, 1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0]];

        var res = []
        for (let i = 1; i < 7; i++)
        {
            var res_line = []
            for (let j = 1; j < Math.min(7, (idx + 2) - ((i - 1) * 6)); j++)
            {
                var conv = 0;
                for (let ii = 0; ii < 3; ii++)
                    for (let jj = 0; jj < 3; jj++)
                        conv += img[i+ii-1][j+jj-1] * kernel[ii][jj]
                res_line = res_line.concat(conv);
            }
            res = res.concat([res_line])
        }

        ctx.reset()
        var img_indeces = [
        Math.floor(idx / 6) * 8 + (idx % 6),
        Math.floor(idx / 6) * 8 + (idx % 6) + 1, 
        Math.floor(idx / 6) * 8 + (idx % 6) + 2, 
        (Math.floor(idx / 6) + 1) * 8 + (idx % 6),
        (Math.floor(idx / 6) + 1) * 8 + (idx % 6) + 1,
        (Math.floor(idx / 6) + 1) * 8 + (idx % 6) + 2,
        (Math.floor(idx / 6) + 2) * 8 + (idx % 6),
        (Math.floor(idx / 6) + 2) * 8 + (idx % 6) + 1,
        (Math.floor(idx / 6) + 2) * 8 + (idx % 6) + 2]
        this.draw_grid(ctx, img, 1, 30, 10, img_indeces,
        0, 2)
        this.draw_grid(ctx, kernel, (idx % 6) * 3 + 105, Math.floor(idx / 6) * 3 + 60, 10, [0, 1, 2, 3, 4, 5, 6, 7, 8], -4, 4)
        this.draw_grid(ctx, res, 180, 100, 10, [idx], min, max)

        ctx.textAlign = "left";
        var text_x0 = 5;
        var text_y0 = 130;
        var colors = ["#e3342f", "#f6993f", "#ffed4a", "#38c172", "#4dc0b5", "#3490dc", "#6574cd", "#9561e2", "#f66d9b"]
        for (let i = 0; i < 3; i++)
        {
            for (let j = 0; j < 3; j++)
            {
                ctx.fillStyle = colors[i * 3 + j];
                ctx.font = "10px Arial";
                var text = img[Math.floor(idx / 6) + i][(idx % 6) + j] + " x " + kernel[i][j]
                if ((i != 2) | (j != 2))
                    text += " + ";
                ctx.fillText(text, text_x0 + (j * 33), text_y0 + (i * 14));
                ctx.stroke();
            }
        }
        ctx.fillStyle = colors[0];
        ctx.font = "10px Arial";
        ctx.fillText("= " + conv, text_x0, text_y0 + (3 * 14));
        ctx.stroke();
    }
    
    draw_grid(ctx, kernel, x0, y0, grid_size, selected_idx, color_min, color_max)
    {
        var colors = ["#e3342f", "#f6993f", "#ffed4a", "#38c172", "#4dc0b5", "#3490dc", "#6574cd", "#9561e2", "#f66d9b"]
        var alpha = degrees_to_radians(10)
        var y_offset = grid_size * Math.sin(alpha)
        var grid_size_x = grid_size * Math.cos(alpha)
        var grid_size_y = grid_size
        var grid_count = 0
        var color_paint_min = 48
        var color_paint_max = 255
        for (let i = 0; i < kernel.length; i++)
        {
            for (let j = 0; j < kernel[i].length; j++)
            {
                ctx.beginPath();
                var pixelValue = ((kernel[i][j] - color_min) / (color_max - color_min)) * (color_paint_max - color_paint_min) + color_paint_min;
                ctx.fillStyle = 'rgb(' + pixelValue + ',' + pixelValue + ',' + pixelValue + ')';

                ctx.strokeStyle = "#000000"
                for (let color_idx = 0; color_idx < selected_idx.length; color_idx++)
                {
                    if (grid_count == selected_idx[color_idx])
                        ctx.fillStyle = colors[color_idx]
                }
                ctx.lineTo(x0 + (j * grid_size_y), y0 + (i * grid_size_x) - (j * y_offset));
                ctx.lineTo(x0 + (j * grid_size_y), y0 + (i * grid_size_x) - (j * y_offset) - grid_size_y);
                ctx.lineTo(x0 + (j * grid_size_y) + grid_size_x, y0 + (i * grid_size_x) - ((j + 1) * y_offset) - grid_size_y);
                ctx.lineTo(x0 + (j * grid_size_y) + grid_size_x, y0 + (i * grid_size_x) - ((j + 1) * y_offset));
                ctx.lineTo(x0 + (j * grid_size_y), y0 + (i * grid_size_x) - (j * y_offset));
                ctx.fill();
                ctx.stroke();
                ctx.closePath();
                
                ctx.textAlign = "center";
                ctx.fillStyle = "#000000";
                ctx.font = (grid_size - 3) + "px Arial";
                ctx.fillText(kernel[i][j], x0 + (j * grid_size_y) + grid_size_y / 2, y0 + (i * grid_size_x) - (j * y_offset) - 3);
                ctx.stroke();
                grid_count += 1
            }
        }
    }
}

window.onload = function()
{
    fig_mri_contrast = new mri_contrast("canvas1", "te1", "tr1");
    fig_mri_b0 = new mri_rf("canvas2", "b02", null);
    fig_mri_rf = new mri_rf("canvas3", "b03", "rfpulse3");
    fig_mri_dephasing = new mri_full_dephasing("canvas4", "rfpulse4", true, "t14", "t24", true, "1804", true, "gradient+4", "gradient-4");
    fig_mri_simple_dephasing = new mri_full_dephasing("canvas5", "rfpulse5", false, null, null, false, null, false, null, null);
    fig_mri_spin_echo = new mri_full_dephasing("canvas6", "rfpulse6", false, null, null, true, "1806", false, null, null);
    fig_mri_gradient_echo = new mri_full_dephasing("canvas7", "rfpulse7", false, null, null, false, null, true, "gradient+7", "gradient-7");
    fig_mri_t1t2 = new mri_full_dephasing("canvas8", "rfpulse8", true, "t18", "t28", false, null, false, null, null);
    fig_mri_gradients = new mri_gradients("canvas9", "gradient9", "rfpulse9")
    fig_mri_simple_downsample = new mri_downsample("canvas10", "acceleration10", "acceleration10", "")
    fig_mri_downsample = new mri_downsample("canvas11", "PE11", "FE11", "valuelabel11")
    fig_mri_phaseencoding = new mri_phaseencoding("canvas12", "protonswitch12_1", "protonswitch12_2", "protonswitch12_3", "protonswitch12_4", "protonswitch12_5", "frequency12", "collect12")
    fig_mlbasics_complexity = new mlbasics_complexity("mlbasics-canvas1", "mlbasics-valuelabel1", "mlbasics-complexity1")
    fig_mlbasics_complexity_decomp = new mlbasics_complexity_decomp("mlbasics-canvas1_decomp", "mlbasics-valuelabel1_decomp", "mlbasics-complexity1_decomp")
    fig_mlbasics_gradients = new mlbasics_gradients("mlbasics-canvas2", "mlbasics-gradient2", "mlbasics-label2", "mlbasics-poslabel2")
    fig_mlbasics_learningrate = new mlbasics_momentum("mlbasics-canvas3", "mlbasics_numofiter3", "mlbasics_learningrate3", null, "mlbasics_losslabel3")
    fig_mlbasics_momentum = new mlbasics_momentum("mlbasics-canvas4", "mlbasics_numofiter4", "mlbasics_learningrate4", "mlbasics_momentum4", "mlbasics_losslabel4")
    fig_mlbasics_forwardpass = new mlbasics_forwardpass("mlbasics-canvas5", "mlbasics_input15", "mlbasics_input25", "mlbasics_w15", "mlbasics_w25", "mlbasics_w35", null, "mlbasics_pause5", 15, 250)
    fig_mlbasics_backwardspass = new mlbasics_forwardpass("mlbasics-canvas6", "mlbasics_input16", "mlbasics_input26", null, null, null, "mlbasics_out6", "mlbasics_pause6", 12, 500)
    fig_mlbasics_receptivefield = new mlbasics_receptivefield("mlbasics-canvas7", "mlbasics-kernel7", "mlbasics-depth7", "mlbasics-fieldlabel7")
    fig_mlbasics_convolution = new mlbasics_convolution("mlbasics-canvas8", "mlbasics-kernels8", "mlbasics-slider8")
}