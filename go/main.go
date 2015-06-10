package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"

	"github.com/harrydb/go/img/grayscale"
)

// number of subdivisions per side, so
//  _____
// |_|_|_|
// |_|_|_|
// |_|_|_|
// would be 3
const HIST_SQRT_NUM_SUBDIV = 16

var stannis_hist []uint

type AnalyzedPicture struct {
	path string
	hist []uint
}

type ByStannis []AnalyzedPicture

func (a ByStannis) Len() int      { return len(a) }
func (a ByStannis) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a ByStannis) Less(i, j int) bool {
	return CosineDistance(a[i].hist, stannis_hist) > CosineDistance(a[j].hist, stannis_hist)
}

func main() {
	fmt.Println("hello, seleção")
	ncpu := runtime.NumCPU()
	fmt.Printf("%v CPUs detected\n", ncpu)
	runtime.GOMAXPROCS(ncpu)

	r, err := os.Open("lfw-deepfunneled/Al_Gore/Al_Gore_0006.jpg")
	if err != nil {
		panic(err)
	}
	defer r.Close()

	src, err := jpeg.Decode(r)
	if err != nil {
		panic(err)
	}

	stannis_gray := grayscale.Convert(src, grayscale.ToGrayLuminance)
	lbp := Elbp(stannis_gray)
	stannis_hist = Histogram(lbp, HIST_SQRT_NUM_SUBDIV)

	var analyzedPictures []AnalyzedPicture
	progress := make(chan bool)
	quit := make(chan bool)
	go func() {
		c := 0
		for {
			select {
			case <-progress:
				c++
				if c%100 == 0 {
					log.Printf("analyzed %v pictures\n", c)
				}
			case <-quit:
				return
			}
		}
	}()

	c := 0
	var wg sync.WaitGroup
	filepath.Walk("lfw-deepfunneled/", func(path string, f os.FileInfo, err error) error {
		if !f.IsDir() {
			r, err := os.Open(path)
			if err != nil {
				panic(err)
			}
			defer r.Close()

			src, err := jpeg.Decode(r)
			if err != nil {
				panic(err)
			}
			wg.Add(1)
			go func(wg *sync.WaitGroup) {
				gray := grayscale.Convert(src, grayscale.ToGrayLuminance)
				analysis := AnalyzedPicture{
					path: path,
					hist: Histogram(Elbp(gray), HIST_SQRT_NUM_SUBDIV),
				}
				analyzedPictures = append(analyzedPictures, analysis)
				progress <- true
				wg.Done()
			}(&wg)
			c++
		}
		return nil
	})

	wg.Wait()
	quit <- true

	sort.Sort(ByStannis(analyzedPictures))

	for i := 0; i < 10; i++ {
		fmt.Println(analyzedPictures[i].path)
	}

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

// gray is stored as a uint8, we want float64
func Float64GrayAt(img *image.Gray, x, y int) float64 {
	return float64(img.GrayAt(x, y).Y) / float64(math.MaxUint8)
}

// extended local binary pattern
func Elbp(img *image.Gray) *image.Gray {
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

func Histogram(img *image.Gray, sqrtNumSubdivisions int) (res []uint) {
	subdivWidth := img.Bounds().Size().X / sqrtNumSubdivisions
	subdivHeight := img.Bounds().Size().Y / sqrtNumSubdivisions

	var histograms [][]uint

	for i := 0; i < sqrtNumSubdivisions; i++ {
		for j := 0; j < sqrtNumSubdivisions; j++ {
			// idk why +1 but it goes out of bounds otherwise
			freq := make([]uint, math.MaxUint8+1)
			startX := subdivWidth * i
			startY := subdivHeight * j
			k := 0
			for x := startX; x < startX+subdivWidth; x++ {
				for y := startY; y < startY+subdivHeight; y++ {
					freq[img.GrayAt(x, y).Y] += 1
					k++
				}
			}
			histograms = append(histograms, freq)
		}
	}

	for _, v := range histograms {
		res = append(res, v...)
	}
	return
}

func CosineDistance(v1, v2 []uint) float64 {
	if len(v1) != len(v2) {
		panic("vectors must be same size")
	}

	dotProduct := 0.0
	sumSquares1 := 0.0
	sumSquares2 := 0.0
	for i := range v1 {
		dotProduct += float64(v1[i] * v2[i])
		sumSquares1 += math.Pow(float64(v1[i]), 2)
		sumSquares2 += math.Pow(float64(v2[i]), 2)
	}
	return dotProduct / (math.Sqrt(sumSquares1) * math.Sqrt(sumSquares2))
}
