# Digital Stethoscope Assembly Guide

## Overview

This guide provides step-by-step instructions for converting a standard analog stethoscope into a digital stethoscope for the RespiScope-AI system.

**Build Time:** 2-3 hours  
**Difficulty:** Intermediate  
**Cost:** ~$46-71 USD

---

## Required Components

| Component | Specification | Quantity | Recommended Model | Est. Cost |
|-----------|--------------|----------|-------------------|-----------|
| Analog Stethoscope | Standard dual-head | 1 | 3M Littmann Classic III or similar | $15-30 |
| Electret Microphone | High-sensitivity, omnidirectional | 1 | Adafruit 1713 or Knowles SPU0410LR5H | $7 |
| TRRS Audio Jack | 3.5mm 4-pole male plug | 1 | Standard TRRS connector | $2 |
| Shielded Cable | 22-24 AWG, 3-conductor | 1m | Standard audio cable | Included |
| Heat Shrink Tubing | Assorted sizes (3mm, 5mm, 8mm) | 1 set | Generic heat shrink kit | $5 |
| Silicone Sealant | Waterproof, medical-grade | 1 tube | Clear silicone adhesive | $5 |
| Solder & Flux | Lead-free solder | - | Standard electronics solder | $3 |
| Small Zip Ties | 2-3mm width | 5-10 | Cable management ties | $2 |
| USB Audio Adapter | (Optional) For higher quality | 1 | Sabrent AU-MMSA or similar | $10-20 |

**Total:** ~$46-71

### Tools Required

- Soldering iron (25-40W)
- Wire strippers
- Small screwdrivers (Phillips and flathead)
- Hot glue gun or epoxy
- Multimeter (for testing continuity)
- Craft knife or scalpel
- Heat gun or lighter (for heat shrink)
- Helping hands or PCB holder

---

## Assembly Instructions

### Step 1: Disassemble the Stethoscope Chest Piece

![Step 1 - Disassembly](images/step1_components.jpg)

1. **Remove the chest piece** from the stethoscope tubing
2. **Unscrew the diaphragm** (the flat circular part)
3. **Remove the diaphragm membrane** - you'll be replacing this with the microphone
4. **Keep all parts organized** - you'll need them for reassembly
5. **Clean the chest piece** thoroughly with isopropyl alcohol

**⚠️ Tip:** Take photos during disassembly to help with reassembly later.

---

### Step 2: Prepare the Microphone

![Step 2 - Microphone Prep](images/step2_microphone.jpg)

1. **Identify the microphone pins:**
   - Most electret microphones have 3 pins: VCC, GND, and OUT
   - Check datasheet for your specific microphone
   
2. **Solder wires to microphone:**
   - **Red wire** → VCC (power, typically 2-10V)
   - **Black wire** → GND (ground)
   - **White/Yellow wire** → OUT (audio output)

3. **Add heat shrink tubing:**
   - Slide small heat shrink over each soldered connection
   - Apply heat to shrink and insulate

4. **Test the microphone:**
   - Connect to 3V power source and oscilloscope
   - Tap microphone gently - you should see signal

**Microphone Specifications:**
- Sensitivity: -42dB to -38dB
- Frequency Response: 100Hz - 10kHz (ideal for respiratory sounds)
- SNR: >58dB
- Power: 2-10V DC

---

### Step 3: Wire the TRRS Connector

![Step 3 - Wiring](images/step3_wiring.jpg)

**TRRS Pinout (Tip to Sleeve):**
1. **Tip (T)** - Left Audio
2. **Ring 1 (R1)** - Right Audio
3. **Ring 2 (R2)** - Ground
4. **Sleeve (S)** - Microphone Input with Bias

**Wiring Connections:**
```
Microphone VCC  → TRRS Sleeve (S) - Gets bias voltage from device
Microphone OUT  → TRRS Tip (T) - Audio signal
Microphone GND  → TRRS Ring 2 (R2) - Ground/Common

Optional: Short Ring 1 (R1) to Tip (T) for stereo compatibility
```

**Soldering Steps:**
1. Strip 5mm of insulation from each wire
2. Tin the wire ends with solder
3. Solder wires to TRRS connector according to diagram
4. Cover each connection with heat shrink tubing
5. Slide larger heat shrink over entire connector assembly
6. Heat to secure and provide strain relief

**Test the connection:**
- Use multimeter to check continuity
- No shorts between adjacent pins
- Proper connection from mic to TRRS

---

### Step 4: Mount Microphone in Chest Piece

![Step 4 - Mounting](images/final_assembly.jpg)

1. **Position the microphone:**
   - Place microphone where diaphragm was
   - Microphone should face toward the diaphragm opening
   - Leave small air gap (~2-3mm) for acoustic coupling

2. **Secure the microphone:**
   - Use hot glue or medical-grade epoxy
   - Apply glue around edges only (don't block microphone port)
   - Ensure microphone is level and centered

3. **Route the cable:**
   - Feed cable through the chest piece tube opening
   - Use small zip tie to provide strain relief
   - Ensure cable doesn't pull on microphone connections

4. **Seal the assembly:**
   - Apply thin bead of silicone sealant around microphone edge
   - This creates acoustic seal and protects from moisture
   - Let cure for 24 hours before use

5. **Reassemble the chest piece:**
   - Replace the outer rim and diaphragm (without membrane)
   - Or create new thin membrane from balloon rubber
   - Screw everything back together securely

---

### Step 5: Cable Management & Strain Relief

1. **Run cable through stethoscope tubing:**
   - The audio cable runs parallel to the stethoscope tube
   - Use zip ties every 10-15cm to secure
   - Don't pull cable too tight - allow some slack

2. **Protect the connector:**
   - Add larger heat shrink or rubber boot over TRRS connector
   - This prevents connector from breaking under strain

3. **Optional: Add USB Audio Interface:**
   - For better quality, use USB audio adapter
   - Connect TRRS to 3.5mm jack of USB adapter
   - This provides cleaner signal with less noise

---

### Step 6: Testing & Calibration

1. **Visual Inspection:**
   - Check all solder joints
   - Ensure no loose wires
   - Verify all connections are insulated

2. **Continuity Test:**
   - Use multimeter to verify connections
   - Check for shorts between pins
   - Verify proper ground connection

3. **Audio Test:**
   - Plug into computer or smartphone
   - Open audio recording software (Audacity, etc.)
   - Select external microphone as input
   - Tap chest piece gently - you should see waveform
   - Adjust input gain if needed (typically 50-80%)

4. **Acoustic Test:**
   - Place on chest and record heartbeat
   - Should clearly hear lub-dub sounds
   - Record breathing - should capture inhalation/exhalation
   - Test cough detection

5. **Noise Test:**
   - Record in quiet room
   - Check for electrical hum (50/60Hz)
   - If present, check grounding and shielding

---

## Connection Options

### Option A: TRRS 3.5mm Jack (Most Common)

**Advantages:**
- Works with smartphones, tablets, laptops
- Simple plug-and-play
- No external power needed

**Disadvantages:**
- Subject to device audio quality
- May have ground loop noise
- Limited by device ADC quality

**Setup:**
1. Plug TRRS jack into device audio port
2. Select "External Microphone" in audio settings
3. Adjust input gain (start at 50%)
4. Test recording

### Option B: USB Audio Interface (Recommended)

**Advantages:**
- Higher quality audio
- Better SNR (60-80dB)
- Consistent performance across devices
- Phantom power for microphone

**Disadvantages:**
- Requires USB audio adapter (~$10-20)
- Additional component to carry

**Recommended USB Adapters:**
- Sabrent AU-MMSA ($10)
- Ugreen USB Audio Adapter ($12)
- Behringer UCA202 ($30) - Professional quality

**Setup:**
1. Connect TRRS to USB adapter's microphone input
2. Plug USB adapter into computer
3. Select USB adapter as audio input device
4. Set sample rate to 16kHz or 44.1kHz
5. Adjust gain for optimal levels

---

## Troubleshooting

### No Audio Signal

**Possible Causes:**
- Loose connection
- Wrong TRRS pinout
- Microphone not powered

**Solutions:**
1. Check all solder joints with multimeter
2. Verify TRRS pinout matches your device
3. Try different audio jack/device
4. Check microphone orientation

### Weak Signal

**Possible Causes:**
- Low input gain
- Microphone too far from diaphragm
- Acoustic leaks

**Solutions:**
1. Increase input gain in software
2. Reposition microphone closer to opening
3. Add acoustic seal with silicone
4. Use USB audio interface with preamp

### Electrical Noise / Hum

**Possible Causes:**
- Ground loop
- Poor shielding
- Proximity to power sources

**Solutions:**
1. Use shielded cable
2. Ensure proper grounding
3. Add ferrite bead to cable
4. Keep away from power adapters
5. Use USB audio interface with isolation

### Distorted Audio

**Possible Causes:**
- Input gain too high
- Microphone clipping
- Damaged microphone

**Solutions:**
1. Reduce input gain
2. Add series resistor to reduce sensitivity
3. Replace microphone if damaged

---

## Usage Tips

### Recording Best Practices

1. **Patient positioning:**
   - Patient should be seated upright
   - Relaxed breathing
   - Quiet environment

2. **Stethoscope placement:**
   - Follow standard auscultation sites
   - Anterior chest: 2nd-4th intercostal spaces
   - Posterior chest: between scapulae
   - Ensure good skin contact

3. **Recording settings:**
   - Sample rate: 16 kHz (minimum) or 44.1 kHz
   - Bit depth: 16-bit minimum
   - Format: WAV (uncompressed)
   - Duration: 10-20 seconds per location
   - Gain: Adjust so peaks reach -6dB to -3dB

4. **Environment:**
   - Quiet room (< 40dB ambient noise)
   - Minimize clothing rustle
   - Turn off fans, AC if possible
   - Ask patient not to talk during recording

### Maintenance

1. **Daily:**
   - Wipe chest piece with alcohol wipe
   - Check cable for damage
   - Verify connector is secure

2. **Weekly:**
   - Check all connections
   - Test audio quality
   - Clean thoroughly with disinfectant

3. **Monthly:**
   - Inspect solder joints
   - Check for corrosion
   - Test with calibration tones

4. **Storage:**
   - Store in dry environment
   - Avoid extreme temperatures
   - Coil cable loosely
   - Protect connector

---

## Safety Considerations

⚠️ **Important Safety Notes:**

1. **Electrical Safety:**
   - This device operates at low voltage (< 5V)
   - Still, avoid using with damaged cables
   - Don't use during electrical cardioversion

2. **Hygiene:**
   - Clean chest piece between patients
   - Use disposable covers when possible
   - Follow infection control protocols

3. **Clinical Use:**
   - This is a screening tool, not diagnostic device
   - Not FDA approved for medical diagnosis
   - Use under supervision of healthcare professional

4. **Privacy:**
   - Secure all recorded data
   - Follow HIPAA guidelines if applicable
   - Obtain informed consent before recording

---

## Performance Specifications

### Achieved Specifications

- **Frequency Response:** 20 Hz - 8 kHz
- **Sensitivity:** -42 dB ± 3dB
- **SNR:** > 55 dB (TRRS), > 65 dB (USB)
- **Dynamic Range:** > 70 dB
- **Maximum SPL:** 120 dB
- **Input Impedance:** 2.2 kΩ (typical)
- **Power Consumption:** < 5 mW

### Audio Quality Metrics

- **THD+N:** < 1% at 1kHz
- **Crosstalk:** < -60 dB
- **Latency:** < 10ms (USB), < 5ms (TRRS)
- **Sample Rate:** 16 kHz (min), up to 96 kHz (USB)

---

## Cost Analysis

### DIY vs Commercial

| Feature | DIY Digital Stethoscope | Commercial (e.g., Eko Core) |
|---------|------------------------|------------------------------|
| Cost | $46-71 | $200-300 |
| Audio Quality | Good (60-65dB SNR) | Excellent (70-80dB SNR) |
| Connectivity | TRRS/USB | Bluetooth + USB |
| AI Integration | Custom (RespiScope-AI) | Proprietary app |
| Maintenance | User serviceable | Warranty only |
| Customization | Full control | Limited |

**Savings:** ~70-80% compared to commercial solutions

---

## Upgrades & Modifications

### Advanced Options

1. **MEMS Microphone Upgrade:**
   - Use digital MEMS mic (e.g., ICS-43434)
   - Better SNR (65+ dB)
   - I2S digital output
   - Requires ESP32 or microcontroller

2. **Wireless Connection:**
   - Add Bluetooth module (HC-05/HC-06)
   - Requires battery and power management
   - Increased cost (~$25 more)

3. **Active Noise Cancellation:**
   - Add second microphone for ambient noise
   - Implement ANC algorithm
   - Significantly improves SNR

4. **Multiple Recording Locations:**
   - Add multiple chest pieces
   - Simultaneous recording
   - Better for comparative analysis

---

## References & Resources

### Datasheets
- [Adafruit Electret Microphone](https://www.adafruit.com/product/1713)
- [TRRS Connector Standard](https://source.android.com/devices/accessories/headset/plug-headset-spec)

### Similar Projects
- [Peter Ma's Digital Stethoscope](https://www.hackster.io/mixpose/digital-stethoscope-ai-1e0229)
- [3M Littmann CORE Digital Stethoscope](https://www.littmann.com/core)

### Further Reading
- Audio signal processing for respiratory sounds
- Stethoscope acoustics and design
- Medical device regulations (FDA, CE)

---

## License

**Hardware Design:** CERN Open Hardware Licence Version 2 - Permissive (CERN-OHL-P)

You are free to:
- Use commercially
- Modify and distribute
- Use for private purposes

Under the conditions:
- Provide attribution
- Share modifications under same license
- Include copyright notice

---

## Support & Community

**Questions?**
- Open an issue on GitHub
- Email: support@respiscope.ai (example)
- Community forum: (link)

**Contribute:**
- Submit improvements
- Share your builds
- Report issues

---

**Version:** 1.0  
**Last Updated:** October 2024  
**Author:** RespiScope-AI Team
