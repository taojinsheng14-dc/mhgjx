$ErrorActionPreference = "Continue"

Write-Host "Stopping adb..."
Get-Process adb -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

$targets = Get-PnpDevice | Where-Object {
    $_.FriendlyName -match 'Android|ADB|MTP|Portable|Xiaomi|Redmi|REDMI|手机' -or
    $_.InstanceId -match 'VID_2717|VID_18D1'
}

Write-Host "`nMatched devices before enable:"
$targets | Select-Object Status, Class, FriendlyName, InstanceId | Format-Table -AutoSize

foreach ($device in $targets) {
    Write-Host "`nEnabling $($device.FriendlyName) [$($device.InstanceId)]"
    pnputil /enable-device "$($device.InstanceId)"
}

Write-Host "`nRestarting parent composite devices..."
$parents = $targets | Where-Object { $_.Class -eq 'USB' -or $_.InstanceId -match 'VID_2717&PID_FF48\\' }
foreach ($device in $parents) {
    Write-Host "`nRestarting $($device.FriendlyName) [$($device.InstanceId)]"
    pnputil /restart-device "$($device.InstanceId)"
}

Write-Host "`nScanning devices..."
pnputil /scan-devices

Write-Host "`nStarting adb..."
& "D:\soft\platform-tools\adb.exe" start-server
Start-Sleep -Seconds 1

Write-Host "`nMatched devices after enable:"
Get-PnpDevice | Where-Object {
    $_.FriendlyName -match 'Android|ADB|MTP|Portable|Xiaomi|Redmi|REDMI|手机' -or
    $_.InstanceId -match 'VID_2717|VID_18D1'
} | Select-Object Status, Class, FriendlyName, InstanceId | Format-Table -AutoSize

Write-Host "`nadb devices:"
& "D:\soft\platform-tools\adb.exe" devices -l
Pause
