$ErrorActionPreference = "Continue"

Write-Host "Stopping adb..."
Get-Process adb -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

$targets = Get-PnpDevice | Where-Object {
    $_.FriendlyName -match 'Android|ADB|MTP|Portable|Xiaomi|Redmi|REDMI|手机' -or
    $_.InstanceId -match 'VID_2717|VID_18D1'
}

Write-Host "`nMatched devices:"
$targets | Select-Object Status, Class, FriendlyName, InstanceId | Format-Table -AutoSize

foreach ($device in $targets) {
    Write-Host "`nRemoving $($device.FriendlyName) [$($device.InstanceId)]"
    pnputil /remove-device "$($device.InstanceId)"
}

Write-Host "`nScanning devices..."
pnputil /scan-devices

Write-Host "`nStarting adb..."
& "D:\soft\platform-tools\adb.exe" start-server
Start-Sleep -Seconds 1

Write-Host "`nadb devices:"
& "D:\soft\platform-tools\adb.exe" devices -l

Write-Host "`nDone. If adb devices is empty, unplug and plug the phone again, then run:"
Write-Host 'D:\soft\platform-tools\adb.exe devices -l'
Pause
