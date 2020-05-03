using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork
{
    class RGB
    {
        public ImagePart Red;
        public ImagePart Green;
        public ImagePart Blue;

        public RGB(ImagePart red, ImagePart green, ImagePart blue)
        {
            Red = red;
            Green = green;
            Blue = blue;
        }
    }

    class ImagePart
    {
        int size;
        double[,] pixels;
        Point left_top_index;
        Point left_bottom_index;
        Point right_top_index;
        Point right_bottom_index;

        public ImagePart(int size)
        {
            this.size = size;
            pixels = new double[size, size];
        }

        public int Size
        {
            get
            {
                return size;
            }
        }

        public double[,] Pixels
        {
            get
            {
                return pixels;
            }
            set
            {
                pixels = value;
            }
        }

        public double Avg
        {
            get
            {
                double avg = 0;
                for (int i = 0; i < pixels.GetLength(0); i++)
                {
                    for (int j = 0; j < pixels.GetLength(1); j++)
                    {
                        avg += pixels[i, j];
                    }
                }

                return avg / (size * size);
            }
        }

        public double Max
        {
            get
            {
                double max = double.MinValue;
                for (int i = 0; i < pixels.GetLength(0); i++)
                {
                    for (int j = 0; j < pixels.GetLength(1); j++)
                    {
                        if (max < pixels[i, j])
                        {
                            max = pixels[i, j];
                        }
                    }
                }

                return max;
            }
        }

        public Point LeftTopIndex { get => left_top_index; set => left_top_index = value; }
        public Point LeftBottomIndex { get => left_bottom_index; set => left_bottom_index = value; }
        public Point RightTopIndex { get => right_top_index; set => right_top_index = value; }
        public Point RightBottomIndex { get => right_bottom_index; set => right_bottom_index = value; }

        public ImagePart Pool(int pooling_size)
        {
            if (size % pooling_size != 0)
            {
                double[,] old_pixels = pixels;
                pixels = new double[old_pixels.GetLength(0) + 1, old_pixels.GetLength(1) + 1];

                for (int i = 0; i < old_pixels.GetLength(0) + 1; i++)
                {
                    for (int j = 0; j < old_pixels.GetLength(1) + 1; j++)
                    {
                        if (i >= old_pixels.GetLength(0))
                        {
                            pixels[i, j] = 0;
                        }
                        else if (j >= old_pixels.GetLength(0))
                        {
                            pixels[i, j] = 0;
                        }
                        else
                        {
                            pixels[i, j] = old_pixels[i, j];
                        }
                    }
                }
            }

            int width = (size / pooling_size) + 1;
            ImagePart image_part = new ImagePart(width);

            for (int i = 0; i < pixels.GetLength(0); i += pooling_size)
            {
                for (int j = 0; j < pixels.GetLength(1); j += pooling_size)
                {
                    ImagePart part = new ImagePart(pooling_size);
                    for (int row = 0; row < pooling_size; row++)
                    {
                        for (int col = 0; col < pooling_size; col++)
                        {
                            part.Pixels[row, col] = pixels[i + row, j + col];
                        }
                    }

                    image_part.Pixels[i / pooling_size, j / pooling_size] = part.Max;
                }
            }

            return image_part;
        }
    }
}
