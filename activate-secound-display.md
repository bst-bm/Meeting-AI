### In case the second screen is Missing

#### Enter in Terminal: 

``` bash
grep -r "amdgpu" /etc/modprobe.d/
```

#### When amdgpu is blacklisted like: 

``` bash
/etc/modprobe.d/blacklist-amdgpu.conf:blacklist amdgpu
```

#### Restart Driver: 

``` bash
sudo rm /etc/modprobe.d/blacklist-amdgpu.conf
sudo update-initramfs -u
sudo modprobe amdgpu
xrandr --auto

```
