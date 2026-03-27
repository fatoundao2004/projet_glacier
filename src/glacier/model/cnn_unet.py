from __future__ import annotations
 
import torch
import torch.nn as nn
 
 
class DoubleConv(nn.Module):
    """(Conv3×3 → BN → ReLU) × 2"""
 
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x):
        return self.block(x)
 
 
class Down(nn.Module):
    """MaxPool → DoubleConv"""
 
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
 
    def forward(self, x):
        return self.block(x)
 
 
class AttentionGate(nn.Module):
    """
    Attention Gate (Oktay et al., 2018).
 
    Filtre les skip connections pour focaliser le décodeur sur les régions
    pertinentes. Le gate signal g (niveau inférieur, sémantiquement riche)
    guide l'attention sur le skip signal x (haute résolution, détails fins).
    """
 
    def __init__(self, f_g: int, f_x: int, f_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_x, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = nn.functional.interpolate(
                g1, size=x1.shape[2:], mode="bilinear", align_corners=False
            )
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
 
 
class Up(nn.Module):
    """ConvTranspose (×2) → Attention Gate (optionnel) → concat skip → DoubleConv"""
 
    def __init__(self, in_ch: int, out_ch: int, use_attention: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.use_attention = use_attention
        if use_attention:
            self.attn = AttentionGate(
                f_g=in_ch // 2,
                f_x=in_ch // 2,
                f_int=in_ch // 4,
            )
        self.conv = DoubleConv(in_ch, out_ch)
 
    def forward(self, x, skip):
        x = self.up(x)
        # Si les tailles diffèrent (arrondi après pool), on crop le skip
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy != 0 or dx != 0:
            skip = skip[
                :,
                :,
                dy // 2 : dy // 2 + x.size(2),
                dx // 2 : dx // 2 + x.size(3),
            ]
        if self.use_attention:
            skip = self.attn(g=x, x=skip)
        return self.conv(torch.cat([skip, x], dim=1))
 
 
class UNet(nn.Module):
    """
    U-Net pour segmentation binaire, avec Attention Gates optionnels.
 
    Parameters
    ----------
    in_channels : int
        Nombre de bandes en entrée (5 pour B, G, R, NIR, SWIR16).
    out_channels : int
        Nombre de classes en sortie (1 pour segmentation binaire).
    features : list of int
        Nombre de filtres à chaque niveau de l'encoder.
    dropout : float
        Taux de dropout dans le bottleneck.
    use_attention : bool
        Si True, ajoute des Attention Gates dans le décodeur.
 
    La sortie est en **logits** (pas de sigmoid).
    """
 
    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 1,
        features: list[int] | None = None,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        f = features
 
        # Encoder
        self.enc0 = DoubleConv(in_channels, f[0])
        self.enc1 = Down(f[0], f[1])
        self.enc2 = Down(f[1], f[2])
        self.enc3 = Down(f[2], f[3])
 
        # Bottleneck
        self.bottleneck = nn.Sequential(
            Down(f[3], f[3] * 2),
            nn.Dropout2d(dropout),
        )
 
        # Decoder avec Attention Gates optionnels
        self.dec3 = Up(f[3] * 2, f[3], use_attention=use_attention)
        self.dec2 = Up(f[3],     f[2], use_attention=use_attention)
        self.dec1 = Up(f[2],     f[1], use_attention=use_attention)
        self.dec0 = Up(f[1],     f[0], use_attention=use_attention)
 
        # Tête de segmentation
        self.head = nn.Conv2d(f[0], out_channels, kernel_size=1)
 
    def forward(self, x):
        # Encoder
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
 
        b = self.bottleneck(s3)
 
        # Decoder + skip connections
        d3 = self.dec3(b, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)
 
        return self.head(d0)
 