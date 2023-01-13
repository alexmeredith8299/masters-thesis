let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/masters-thesis/cloud-detection-code/scripts
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
set shortmess=aoO
badd +1 __init__.py
badd +122 basic_blocks.py
badd +112 block_builder.py
badd +368 c8_invariant_cnn.py
badd +234 cloud_dataset.py
badd +167 equivariant_basic_blocks.py
badd +61 escnn_extension.py
badd +197 evaluate_engaging.py
badd +278 evaluate_model.py
badd +1 focal_loss.py
badd +275 generate_sbatch.py
badd +127 get_from_slurm.py
badd +1 test.py
badd +1 test_bug.py
badd +1 test_cloud_dataset.py
badd +104 train_on_engaging.py
badd +178 train_pytorch_model.py
badd +144 ../tests/ci_ignore/test_train_on_engaging.py
badd +197 ../tests/test_generate_sbatch.py
badd +58 ../tests/test_c8_invariant_cnn.py
badd +57 ../tests/test_basic_block_conversions.py
badd +281 ../tests/test_escnn_extension.py
badd +1 ../tests/test_train_on_engaging.py
badd +40 ../../massachusetts-roads-dataset/create_dataset.py
badd +1 ../../scitech-dataset/create_dataset.py
badd +11 ../tests/test_cloud_dataset.py
badd +1 ../tests/test_road_dataset.py
badd +200 road_dataset.py
argglobal
%argdel
$argadd __init__.py
$argadd basic_blocks.py
$argadd block_builder.py
$argadd c8_invariant_cnn.py
$argadd cloud_dataset.py
$argadd equivariant_basic_blocks.py
$argadd escnn_extension.py
$argadd evaluate_engaging.py
$argadd evaluate_model.py
$argadd focal_loss.py
$argadd generate_sbatch.py
$argadd get_from_slurm.py
$argadd test.py
$argadd test_bug.py
$argadd test_cloud_dataset.py
$argadd train_on_engaging.py
$argadd train_pytorch_model.py
edit evaluate_engaging.py
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 39 + 40) / 80)
exe 'vert 2resize ' . ((&columns * 40 + 40) / 80)
argglobal
if bufexists(fnamemodify("evaluate_engaging.py", ":p")) | buffer evaluate_engaging.py | else | edit evaluate_engaging.py | endif
if &buftype ==# 'terminal'
  silent file evaluate_engaging.py
endif
balt evaluate_model.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 197 - ((10 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 197
normal! 011|
wincmd w
argglobal
if bufexists(fnamemodify("evaluate_model.py", ":p")) | buffer evaluate_model.py | else | edit evaluate_model.py | endif
if &buftype ==# 'terminal'
  silent file evaluate_model.py
endif
balt generate_sbatch.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 278 - ((10 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 278
normal! 053|
wincmd w
exe 'vert 1resize ' . ((&columns * 39 + 40) / 80)
exe 'vert 2resize ' . ((&columns * 40 + 40) / 80)
tabnext 1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
let &shortmess = s:shortmess_save
let &winminheight = s:save_winminheight
let &winminwidth = s:save_winminwidth
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
