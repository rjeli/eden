package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"math"
	"os"

	"github.com/harrydb/go/img/grayscale"
)

const HISTOGRAM_SUBDIVISION_SIDE_LEN = 4

func main() {
	fmt.Println("hello, seleção")

	r, err := os.Open("lfw-deepfunneled/George_W_Bush/George_W_Bush_0001.jpg")
	if err != nil {
		panic(err)
	}
	defer r.Close()

	src, err := jpeg.Decode(r)
	if err != nil {
		panic(err)
	}

	gray := grayscale.Convert(src, grayscale.ToGrayLuminance)
	lbp := elbp(gray)

	out, err := os.Create("output.png")
	if err != nil {
		panic(err)
	}
	defer out.Close()

	err = png.Encode(out, lbp)
	if err != nil {
		panic(err)
	}
}

func Float64GrayAt(img *image.Gray, x, y int) float64 {
	return float64(img.GrayAt(x, y).Y) / float64(math.MaxUint8)
}

func elbp(img *image.Gray) *image.Gray {
	neighbors := 8
	radius := 1
	b := img.Bounds()
	w := b.Size().X
	h := b.Size().Y

	dst := image.NewGray(img.Bounds())

	for n := 0; n < neighbors; n++ {
		// sample points
		x := math.Cos(2.0 * math.Pi * float64(n) / float64(neighbors))
		y := math.Cos(2.0 * math.Pi * float64(n) / float64(neighbors))
		// relative indices
		fx := int(math.Floor(x))
		fy := int(math.Floor(y))
		cx := int(math.Ceil(x))
		cy := int(math.Ceil(y))
		// fractional part
		ty := y - float64(fy)
		tx := x - float64(fx)
		// set interpolation weights
		w1 := (1 - tx) * (1 - ty)
		w2 := tx * (1 - ty)
		w3 := (1 - tx) * ty
		w4 := tx * ty
		// iterate through data
		for i := radius; i < w-radius; i++ {
			for j := radius; j < h-radius; j++ {
				pw1 := w1 * Float64GrayAt(img, i+fy, j+fx)
				pw2 := w2 * Float64GrayAt(img, i+fy, j+cx)
				pw3 := w3 * Float64GrayAt(img, i+cy, j+fx)
				pw4 := w4 * Float64GrayAt(img, i+cy, j+cx)
				t := pw1 + pw2 + pw3 + pw4
				center := Float64GrayAt(img, i, j)
				if t > center {
					p := dst.PixOffset(i-radius, j-radius)
					dst.Pix[p] += (1 << uint(n))
				}
			}
		}
	}

	return dst
}
