$ErrorActionPreference = "Continue"

Write-Host "Stopping adb..."
Get-Process adb -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

Write-Host "`nDisable USB selective suspend for AC/DC..."
powercfg /SETACVALUEINDEX SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0
powercfg /SETDCVALUEINDEX SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0
powercfg /SETACTIVE SCHEME_CURRENT

Write-Host "`nStart portable device services..."
Set-Service WPDBusEnum -StartupType Manual -ErrorAction SilentlyContinue
Start-Service WPDBusEnum -ErrorAction SilentlyContinue
Start-Service ShellHWDetection -ErrorAction SilentlyContinue

Write-Host "`nDisable power saving on USB hubs/controllers where possible..."
$usbDevices = Get-CimInstance Win32_PnPEntity | Where-Object {
    $_.Name -match 'USB Root Hub|USB 根集线器|Generic USB Hub|通用 USB 集线器|USB Composite|USB xHCI|可扩展主机控制器'
}

foreach ($device in $usbDevices) {
    $escaped = $device.PNPDeviceID -replace '\\', '\\'
    $pm = Get-CimInstance -Namespace root\wmi -ClassName MSPower_DeviceEnable -ErrorAction SilentlyContinue |
        Where-Object { $_.InstanceName -like "*$escaped*" }
    foreach ($item in $pm) {
        try {
            $item.Enable = $false
            Set-CimInstance -InputObject $item | Out-Null
            Write-Host "Disabled power management: $($device.Name)"
        } catch {
            Write-Host "Skip power management: $($device.Name) - $($_.Exception.Message)"
        }
    }
}

Write-Host "`nRemove REDMI/ADB stale devices..."
$targets = Get-PnpDevice | Where-Object {
    $_.FriendlyName -match 'Android|ADB|MTP|Portable|Xiaomi|Redmi|REDMI|手机' -or
    $_.InstanceId -match 'VID_2717|VID_18D1'
}

$targets | Select-Object Status, Class, FriendlyName, InstanceId | Format-Table -AutoSize
foreach ($device in $targets) {
    Write-Host "Removing $($device.FriendlyName) [$($device.InstanceId)]"
    pnputil /remove-device "$($device.InstanceId)"
}

Write-Host "`nScan devices..."
pnputil /scan-devices

Write-Host "`nStart adb..."
& "D:\soft\platform-tools\adb.exe" start-server
Start-Sleep -Seconds 1
& "D:\soft\platform-tools\adb.exe" devices -l

Write-Host "`nDone. Unplug the phone, wait 5 seconds, plug it back directly into the laptop, then run:"
Write-Host 'D:\soft\platform-tools\adb.exe devices -l'
Pause
