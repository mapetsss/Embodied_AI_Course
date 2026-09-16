$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $modulePrefix = "$([char]0x6A21)$([char]0x5757)"
    foreach ($assignment in @("${modulePrefix}1", "${modulePrefix}2")) {
        Write-Host "Running $assignment"
        python "$assignment/main.py"
        if ($LASTEXITCODE -ne 0) { throw "$assignment execution failed" }
        python "$assignment/test_homework.py"
        if ($LASTEXITCODE -ne 0) { throw "$assignment tests failed" }
    }
} finally {
    Pop-Location
}
