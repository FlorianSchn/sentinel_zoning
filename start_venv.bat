@echo off
powershell -NoExit -Command "& {Set-Location '%~dp0'; & '.\Scripts\activate'}"
