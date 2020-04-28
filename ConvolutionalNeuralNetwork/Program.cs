using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork
{
    class Program
    {
        static List<double> cone_outputs = new List<double>();

        static double sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        static double sigmoid_p(double x)
        {
            return sigmoid(x) * (1 - sigmoid(x));
        }

        static string layer1_filters_folder_path = "filters/layer1";
        static string layer2_filters_folder_path = "filters/layer2";

        static string train_images_cones_folder_path = "train_images/cone";
        static string train_images_not_cones_folder_path = "train_images/not_cone";

        static string test_images_cones_folder_path = "test_images/cones";
        static string test_images_not_cones_folder_path = "test_images/not_cones";

        static void Main(string[] args)
        {
            List<Tuple<double[], short>> data = new List<Tuple<double[], short>>();

            List<RGB> filters = new List<RGB>();
            List<RGB> filters2 = new List<RGB>();

            DirectoryInfo layer1_filters_directory = new DirectoryInfo(layer1_filters_folder_path);
            FileInfo[] Files = layer1_filters_directory.GetFiles();
            foreach (FileInfo file in Files)
            {
                addFilter(string.Format("{0}/{1}", layer1_filters_folder_path, file.Name), ref filters, 7);
            }

            DirectoryInfo layer2_filters_directory = new DirectoryInfo(layer2_filters_folder_path);
            Files = layer2_filters_directory.GetFiles();
            foreach (FileInfo file in Files)
            {
                addFilter(string.Format("{0}/{1}", layer2_filters_folder_path, file.Name), ref filters2, 5);
            }

            List<Tuple<string, short>> inputs = new List<Tuple<string, short>>();

            DirectoryInfo train_images_cones_directory = new DirectoryInfo(train_images_cones_folder_path);
            Files = train_images_cones_directory.GetFiles();
            foreach (FileInfo file in Files)
            {
                inputs.Add(new Tuple<string, short>(string.Format("{0}/{1}", train_images_cones_folder_path, file.Name), 1));
            }

            DirectoryInfo train_images_not_cones_directory = new DirectoryInfo(train_images_not_cones_folder_path);
            Files = train_images_not_cones_directory.GetFiles();
            foreach (FileInfo file in Files)
            {
                inputs.Add(new Tuple<string, short>(string.Format("{0}/{1}", train_images_not_cones_folder_path, file.Name), 0));
            }

            Random rand = new Random();
            double bias = rand.NextDouble();
            List<double> weights = new List<double>();

            for (int i = 0; i < 21; i++)
            {
                weights.Add(rand.NextDouble());
            }


            foreach (Tuple<string, short> input in inputs)
            {
                data.Add(makeLayers(input.Item1, ref filters, ref filters2, true, input.Item2));
            }

            int iterations = 1000000;
            float learning_rate = 0.1f;
            List<double> costs = new List<double>();

            for (int i = 0; i < iterations; i++)
            {
                Tuple<double[], short> point = data[rand.Next(0, data.Count - 1)];

                double z = 0;
                for (int j = 0; j < weights.Count; j++)
                {
                    z += point.Item1[j] * weights[j];
                }
                z += bias;

                double pred = sigmoid(z);

                double target = point.Item2;

                if (i % 100 == 0)
                {
                    double c = 0;
                    for (int j = 0; j < data.Count; j++)
                    {
                        Tuple<double[], short> p = data[j];
                        double into_sigmoid = 0;
                        for (int u = 0; u < weights.Count; u++)
                        {
                            into_sigmoid += p.Item1[u] * weights[u];
                        }
                        into_sigmoid += bias;

                        double p_pred = sigmoid(into_sigmoid);

                        c += Math.Pow(p_pred - p.Item2, 2);
                    }
                    costs.Add(c);
                    Console.WriteLine("iteration {0}\t{1}", i, c);
                }

                double dcost_dpred = 2 * (pred - target);
                double dpred_dz = sigmoid_p(z);

                List<double> dz_d_weights = new List<double>();
                for (int j = 0; j < point.Item1.Length; j++)
                {
                    dz_d_weights.Add(point.Item1[j]);
                }

                int dz_db = 1;

                double dcost_dz = dcost_dpred * dpred_dz;

                List<double> d_cost_d_weights = new List<double>();
                for (int j = 0; j < dz_d_weights.Count; j++)
                {
                    d_cost_d_weights.Add(dcost_dz * dz_d_weights[j]);
                }

                double dcost_db = dcost_dz * dz_db;

                for (int j = 0; j < weights.Count; j++)
                {
                    weights[j] -= learning_rate * d_cost_d_weights[j];
                }

                bias -= learning_rate * dcost_db;
            }

            List<string> test_cones_inputs = new List<string>();
            List<string> test_not_cones_inputs = new List<string>();

            DirectoryInfo test_images_cones_directory = new DirectoryInfo(test_images_cones_folder_path);
            Files = test_images_cones_directory.GetFiles();
            foreach (FileInfo file in Files)
            {
                test_cones_inputs.Add(string.Format("{0}/{1}", test_images_cones_folder_path, file.Name));
            }

            DirectoryInfo test_images_not_cones_directory = new DirectoryInfo(test_images_not_cones_folder_path);
            Files = test_images_not_cones_directory.GetFiles();
            foreach (FileInfo file in Files)
            {
                test_not_cones_inputs.Add(string.Format("{0}/{1}", test_images_not_cones_folder_path, file.Name));
            }

            Console.WriteLine("\n\nTest cones");

            foreach (string test_input in test_cones_inputs)
            {
                List<double> mystery_cone = new List<double>();




                Bitmap image = new Bitmap(test_input);
                double[,] red = new double[image.Width, image.Height];
                double[,] green = new double[image.Width, image.Height];
                double[,] blue = new double[image.Width, image.Height];

                for (int i = 0; i < image.Width; i++)
                {
                    for (int j = 0; j < image.Height; j++)
                    {
                        Color pixel = image.GetPixel(i, j);
                        red[i, j] = pixel.R;
                        green[i, j] = pixel.G;
                        blue[i, j] = pixel.B;
                    }
                }

                Console.WriteLine("\tMaking reds.");
                List<ImagePart> reds = new List<ImagePart>();
                for (int i = 0; i < 41; i += 10)
                {
                    for (int j = 0; j < 41; j += 10)
                    {
                        ImagePart image_part = new ImagePart(100);
                        for (int row = 0; row < 100; row++)
                        {
                            for (int col = 0; col < 100; col++)
                            {
                                image_part.Pixels[row, col] = red[i + row, j + col];
                            }
                        }
                        reds.Add(image_part);
                    }
                }

                Console.WriteLine("\tMaking greens.");
                List<ImagePart> greens = new List<ImagePart>();
                for (int i = 0; i < 41; i += 10)
                {
                    for (int j = 0; j < 41; j += 10)
                    {
                        ImagePart image_part = new ImagePart(100);
                        for (int row = 0; row < 100; row++)
                        {
                            for (int col = 0; col < 100; col++)
                            {
                                image_part.Pixels[row, col] = green[i + row, j + col];
                            }
                        }
                        greens.Add(image_part);
                    }
                }

                Console.WriteLine("\tMaking blues.");
                List<ImagePart> blues = new List<ImagePart>();
                for (int i = 0; i < 41; i += 10)
                {
                    for (int j = 0; j < 41; j += 10)
                    {
                        ImagePart image_part = new ImagePart(100);
                        for (int row = 0; row < 100; row++)
                        {
                            for (int col = 0; col < 100; col++)
                            {
                                image_part.Pixels[row, col] = blue[i + row, j + col];
                            }
                        }
                        blues.Add(image_part);
                    }
                }

                List<double> outputs = new List<double>();

                Console.WriteLine("\tCalculating reds.");
                int reds_index = 0;
                foreach (ImagePart image_part in reds)
                {
                    mystery_cone = makeLayers(test_input, ref filters, ref filters2, false).Item1.ToList();

                    double z1 = 0;
                    for (int i = 0; i < weights.Count; i++)
                    {
                        z1 += weights[i] * mystery_cone[i];
                    }
                    z1 += bias;

                    outputs.Add(sigmoid(z1));
                    Console.WriteLine("\t\t{0}/{1}", reds_index++, reds.Count);
                }

                Console.WriteLine("\tCalculating greens.");
                int greens_index = 0;
                foreach (ImagePart image_part in greens)
                {
                    mystery_cone = makeLayers(test_input, ref filters, ref filters2, false).Item1.ToList();

                    double z1 = 0;
                    for (int i = 0; i < weights.Count; i++)
                    {
                        z1 += weights[i] * mystery_cone[i];
                    }
                    z1 += bias;

                    outputs.Add(sigmoid(z1));
                    Console.WriteLine("\t\t{0}/{1}", greens_index++, greens.Count);
                }

                Console.WriteLine("\tCalculating blues.");
                int blues_index = 0;
                foreach (ImagePart image_part in blues)
                {
                    mystery_cone = makeLayers(test_input, ref filters, ref filters2, false).Item1.ToList();

                    double z1 = 0;
                    for (int i = 0; i < weights.Count; i++)
                    {
                        z1 += weights[i] * mystery_cone[i];
                    }
                    z1 += bias;

                    outputs.Add(sigmoid(z1));
                    Console.WriteLine("\t\t{0}/{1}", blues_index++, blues.Count);
                }

                double output = outputs.Average();
                Console.WriteLine("{0}:\t{1}\t{2}", test_input, output, output >= 0.5 ? "IGEN" : "NEM");

                /* mystery_cone = makeLayers(test_input, ref filters, ref filters2, false).Item1.ToList();

                 double z1 = 0;
                 for (int i = 0; i < weights.Count; i++)
                 {
                     z1 += weights[i] * mystery_cone[i];
                 }
                 z1 += bias;

                 double pred1 = sigmoid(z1);

                 Console.WriteLine("{0}:\t{1}\t{2}", test_input, pred1, pred1 >= 0.5 ? "IGEN" : "NEM");*/
            }

            Console.WriteLine("\n\nTest not cones");

            foreach (string test_input in test_not_cones_inputs)
            {
                List<double> mystery_cone = new List<double>();
                mystery_cone = (makeLayers(test_input, ref filters, ref filters2, false).Item1.ToList());

                double z1 = 0;
                for (int i = 0; i < weights.Count; i++)
                {
                    z1 += weights[i] * mystery_cone[i];
                }
                z1 += bias;

                double pred1 = sigmoid(z1);

                Console.WriteLine("{0}:\t{1}\t{2}", test_input, pred1, pred1 >= 0.5 ? "IGEN" : "NEM");
            }

            Console.WriteLine("finished");
            Console.ReadKey();
        }

        static void addFilter(string file_name, ref List<RGB> filters, int size)
        {
            Bitmap image = new Bitmap(file_name);
            byte[,] red = new byte[image.Width, image.Height];
            byte[,] green = new byte[image.Width, image.Height];
            byte[,] blue = new byte[image.Width, image.Height];

            for (int i = 0; i < image.Width; i++)
            {
                for (int j = 0; j < image.Height; j++)
                {
                    Color pixel = image.GetPixel(i, j);
                    red[i, j] = pixel.R;
                    green[i, j] = pixel.G;
                    blue[i, j] = pixel.B;
                }
            }
            ImagePart filterRed = new ImagePart(size);
            for (int i = 0; i < image.Width; i++)
            {
                for (int j = 0; j < image.Height; j++)
                {
                    filterRed.Pixels[i, j] = red[i, j];
                }
            }

            ImagePart filterGreen = new ImagePart(size);
            for (int i = 0; i < image.Width; i++)
            {
                for (int j = 0; j < image.Height; j++)
                {
                    filterGreen.Pixels[i, j] = green[i, j];
                }
            }

            ImagePart filterBlue = new ImagePart(size);
            for (int i = 0; i < image.Width; i++)
            {
                for (int j = 0; j < image.Height; j++)
                {
                    filterBlue.Pixels[i, j] = blue[i, j];
                }
            }

            filters.Add(new RGB(filterRed, filterGreen, filterBlue));
        }

        static ImagePart multiplication(ImagePart image_part, ImagePart filter)
        {
            ImagePart multiplicated = new ImagePart(image_part.Size);

            for (int i = 0; i < image_part.Size; i++)
            {
                for (int j = 0; j < image_part.Size; j++)
                {
                    multiplicated.Pixels[i, j] = image_part.Pixels[i, j] * filter.Pixels[i, j];
                }
            }

            return multiplicated;
        }

        static Tuple<double[], short> makeLayers(string file_name, ref List<RGB> filters, ref List<RGB> filters2, bool kiir, short output = 10)
        {
            if (kiir)
                Console.WriteLine("reading {0} ", file_name);

            Bitmap image = new Bitmap(file_name);
            double[,] red = new double[image.Width, image.Height];
            double[,] green = new double[image.Width, image.Height];
            double[,] blue = new double[image.Width, image.Height];

            for (int i = 0; i < image.Width; i++)
            {
                for (int j = 0; j < image.Height; j++)
                {
                    Color pixel = image.GetPixel(i, j);
                    red[i, j] = pixel.R;
                    green[i, j] = pixel.G;
                    blue[i, j] = pixel.B;
                }
            }

            int size = 7;
            int width = 94;

            double[,] first_red = new double[width, width];
            double[,] first_green = new double[width, width];
            double[,] first_blue = new double[width, width];

            int a_index = 0;
            foreach (RGB filter in filters)
            {
                //RED
                List<ImagePart> reds = new List<ImagePart>();
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        ImagePart image_part = new ImagePart(size);
                        for (int row = 0; row < size; row++)
                        {
                            for (int col = 0; col < size; col++)
                            {
                                image_part.Pixels[row, col] = red[i + row, j + col];
                            }
                        }

                        reds.Add(multiplication(image_part, filter.Red));
                    }
                }

                //GREEN
                List<ImagePart> greens = new List<ImagePart>();
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        ImagePart image_part = new ImagePart(size);
                        for (int row = 0; row < size; row++)
                        {
                            for (int col = 0; col < size; col++)
                            {
                                image_part.Pixels[row, col] = green[i + row, j + col];
                            }
                        }

                        greens.Add(multiplication(image_part, filter.Green));
                    }
                }

                //BLUE
                List<ImagePart> blues = new List<ImagePart>();
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        ImagePart image_part = new ImagePart(size);
                        for (int row = 0; row < size; row++)
                        {
                            for (int col = 0; col < size; col++)
                            {
                                image_part.Pixels[row, col] = blue[i + row, j + col];
                            }
                        }

                        blues.Add(multiplication(image_part, filter.Blue));
                    }
                }

                List<RGB> pooled = new List<RGB>();
                for (int i = 0; i < blues.Count; i++)
                {
                    pooled.Add(new RGB(reds[i].Pool(2), greens[i].Pool(2), blues[i].Pool(2)));
                }

                int index = 0;
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < width; y++)
                    {
                        first_red[x, y] = (int)pooled[index].Red.Max;
                        first_green[x, y] = (int)pooled[index].Green.Max;
                        first_blue[x, y] = (int)pooled[index].Blue.Max;
                        index++;
                    }
                }

                Bitmap bitmap = new Bitmap(width, width, PixelFormat.Format32bppArgb);
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < width; y++)
                    {
                        Color color = Color.FromArgb(255, (int)first_red[x, y] / 255, (int)first_green[x, y] / 255, (int)first_blue[x, y] / 255);
                        bitmap.SetPixel(x, y, color);
                    }
                }

                bitmap.Save(string.Format("probak/conv1_{0}.bmp", a_index++), ImageFormat.Bmp);
            }

            double[,] pooled_first_red = new double[47, 47];
            for (int i = 0; i < width; i += 2)
            {
                for (int j = 0; j < width; j += 2)
                {
                    try
                    {
                        ImagePart part = new ImagePart(2);
                        for (int row = 0; row < 2; row++)
                        {
                            for (int col = 0; col < 2; col++)
                            {
                                part.Pixels[row, col] = first_red[i + row, j + col];
                            }
                        }
                        pooled_first_red[i / 2, j / 2] = part.Max;
                    }
                    catch (Exception)
                    {
                    }
                }
            }

            double[,] pooled_first_green = new double[47, 47];
            for (int i = 0; i < width; i += 2)
            {
                for (int j = 0; j < width; j += 2)
                {
                    try
                    {
                        ImagePart part = new ImagePart(2);
                        for (int row = 0; row < 2; row++)
                        {
                            for (int col = 0; col < 2; col++)
                            {
                                part.Pixels[row, col] = first_green[i + row, j + col];
                            }
                        }
                        pooled_first_green[i / 2, j / 2] = part.Max;
                    }
                    catch (Exception)
                    {
                    }
                }
            }

            double[,] pooled_first_blue = new double[47, 47];
            for (int i = 0; i < width; i += 2)
            {
                for (int j = 0; j < width; j += 2)
                {
                    try
                    {
                        ImagePart part = new ImagePart(2);
                        for (int row = 0; row < 2; row++)
                        {
                            for (int col = 0; col < 2; col++)
                            {
                                part.Pixels[row, col] = first_blue[i + row, j + col];
                            }
                        }
                        pooled_first_blue[i / 2, j / 2] = part.Max;
                    }
                    catch (Exception)
                    {
                    }
                }
            }

            width = 47;
            Bitmap bitmap1 = new Bitmap(width, width, PixelFormat.Format32bppArgb);
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < width; y++)
                {
                    double r = pooled_first_red[x, y];
                    while (r > 255)
                    {
                        r /= 255;
                    }

                    double g = pooled_first_green[x, y];
                    while (g > 255)
                    {
                        g /= 255;
                    }

                    double b = pooled_first_blue[x, y];
                    while (b > 255)
                    {
                        b /= 255;
                    }
                    Color color = Color.FromArgb(255, (int)r, (int)g, (int)b);
                    bitmap1.SetPixel(x, y, color);
                }
            }

            bitmap1.Save(string.Format("probak/conv1_pool_{0}.bmp", a_index++), ImageFormat.Bmp);





            size = 5;
            width = 43;

            double[,] second_red = new double[width, width];
            double[,] second_green = new double[width, width];
            double[,] second_blue = new double[width, width];

            foreach (RGB filter in filters2)
            {
                //RED
                List<ImagePart> reds = new List<ImagePart>();
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        try
                        {
                            ImagePart image_part = new ImagePart(size);
                            for (int row = 0; row < size; row++)
                            {
                                for (int col = 0; col < size; col++)
                                {
                                    image_part.Pixels[row, col] = pooled_first_red[i + row, j + col];
                                }
                            }

                            reds.Add(multiplication(image_part, filter.Red));
                        }
                        catch (Exception)
                        {
                        }
                    }
                }

                //GREEN
                List<ImagePart> greens = new List<ImagePart>();
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        try
                        {
                            ImagePart image_part = new ImagePart(size);
                            for (int row = 0; row < size; row++)
                            {
                                for (int col = 0; col < size; col++)
                                {
                                    image_part.Pixels[row, col] = pooled_first_green[i + row, j + col];
                                }
                            }

                            greens.Add(multiplication(image_part, filter.Green));
                        }
                        catch (Exception)
                        {
                        }
                    }
                }

                //BLUE
                List<ImagePart> blues = new List<ImagePart>();
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        try
                        {
                            ImagePart image_part = new ImagePart(size);
                            for (int row = 0; row < size; row++)
                            {
                                for (int col = 0; col < size; col++)
                                {
                                    image_part.Pixels[row, col] = pooled_first_blue[i + row, j + col];
                                }
                            }

                            blues.Add(multiplication(image_part, filter.Blue));
                        }
                        catch (Exception)
                        {
                        }
                    }
                }

                List<RGB> pooled = new List<RGB>();
                for (int i = 0; i < blues.Count; i++)
                {
                    pooled.Add(new RGB(reds[i].Pool(2), greens[i].Pool(2), blues[i].Pool(2)));
                }

                int index = 0;
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < width; y++)
                    {
                        second_red[x, y] = (int)pooled[index].Red.Max;
                        second_green[x, y] = (int)pooled[index].Green.Max;
                        second_blue[x, y] = (int)pooled[index].Blue.Max;
                        index++;
                    }
                }

                Bitmap bitmap = new Bitmap(width, width, PixelFormat.Format32bppArgb);
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < width; y++)
                    {
                        double r = second_red[x, y];
                        while (r > 255)
                        {
                            r /= 255;
                        }

                        double g = second_green[x, y];
                        while (g > 255)
                        {
                            g /= 255;
                        }

                        double b = second_blue[x, y];
                        while (b > 255)
                        {
                            b /= 255;
                        }
                        Color color = Color.FromArgb(255, (int)r, (int)g, (int)b);
                        bitmap.SetPixel(x, y, color);
                    }
                }

                bitmap.Save(string.Format("probak/conv2_{0}.bmp", a_index++), ImageFormat.Bmp);
            }



            double[,] pooled_second_red = new double[21, 21];
            for (int i = 0; i < width; i += 2)
            {
                for (int j = 0; j < width; j += 2)
                {
                    try
                    {
                        ImagePart part = new ImagePart(2);
                        for (int row = 0; row < 2; row++)
                        {
                            for (int col = 0; col < 2; col++)
                            {
                                part.Pixels[row, col] = second_red[i + row, j + col];
                            }
                        }
                        pooled_second_red[i / 2, j / 2] = part.Max;
                    }
                    catch (Exception)
                    {
                    }
                }
            }

            double[,] pooled_second_green = new double[21, 21];
            for (int i = 0; i < width; i += 2)
            {
                for (int j = 0; j < width; j += 2)
                {
                    try
                    {
                        ImagePart part = new ImagePart(2);
                        for (int row = 0; row < 2; row++)
                        {
                            for (int col = 0; col < 2; col++)
                            {
                                part.Pixels[row, col] = second_green[i + row, j + col];
                            }
                        }
                        pooled_second_green[i / 2, j / 2] = part.Max;
                    }
                    catch (Exception)
                    {
                    }
                }
            }

            double[,] pooled_second_blue = new double[21, 21];
            for (int i = 0; i < width; i += 2)
            {
                for (int j = 0; j < width; j += 2)
                {
                    try
                    {
                        ImagePart part = new ImagePart(2);
                        for (int row = 0; row < 2; row++)
                        {
                            for (int col = 0; col < 2; col++)
                            {
                                part.Pixels[row, col] = second_blue[i + row, j + col];
                            }
                        }
                        pooled_second_blue[i / 2, j / 2] = part.Max;
                    }
                    catch (Exception)
                    {
                    }
                }
            }



            double[] layer = new double[21 * 21];
            int layer_index = 0;
            width = 21;
            Bitmap bitmap2 = new Bitmap(width, width, PixelFormat.Format32bppArgb);
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < width; y++)
                {
                    double r = pooled_second_red[x, y];
                    while (r > 255)
                    {
                        r /= 255;
                    }

                    double g = pooled_second_green[x, y];
                    while (g > 255)
                    {
                        g /= 255;
                    }

                    double b = pooled_second_blue[x, y];
                    while (b > 255)
                    {
                        b /= 255;
                    }
                    layer[layer_index++] = (r + g + b) / 1000;
                    Color color = Color.FromArgb(255, (int)r, (int)g, (int)b);
                    bitmap2.SetPixel(x, y, color);
                }
            }

            bitmap2.Save(string.Format("probak/conv2_pool_{0}.bmp", a_index++), ImageFormat.Bmp);


            Tuple<double[], short> last_layer;

            last_layer = new Tuple<double[], short>(layer, output);
            return last_layer;
        }

        static Tuple<double[], short> makeLayers(RGB image, ref List<RGB> filters, ref List<RGB> filters2, short output = 10)
        {
            // Bitmap image = new Bitmap(file_name);
            double[,] red = image.Red.Pixels;
            double[,] green = image.Red.Pixels;
            double[,] blue = image.Red.Pixels;

            int size = 7;
            int width = 94;

            double[,] first_red = new double[width, width];
            double[,] first_green = new double[width, width];
            double[,] first_blue = new double[width, width];

            int a_index = 0;
            foreach (RGB filter in filters)
            {
                //RED
                List<ImagePart> reds = new List<ImagePart>();
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        ImagePart image_part = new ImagePart(size);
                        for (int row = 0; row < size; row++)
                        {
                            for (int col = 0; col < size; col++)
                            {
                                image_part.Pixels[row, col] = red[i + row, j + col];
                            }
                        }

                        reds.Add(multiplication(image_part, filter.Red));
                    }
                }

                //GREEN
                List<ImagePart> greens = new List<ImagePart>();
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        ImagePart image_part = new ImagePart(size);
                        for (int row = 0; row < size; row++)
                        {
                            for (int col = 0; col < size; col++)
                            {
                                image_part.Pixels[row, col] = green[i + row, j + col];
                            }
                        }

                        greens.Add(multiplication(image_part, filter.Green));
                    }
                }

                //BLUE
                List<ImagePart> blues = new List<ImagePart>();
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        ImagePart image_part = new ImagePart(size);
                        for (int row = 0; row < size; row++)
                        {
                            for (int col = 0; col < size; col++)
                            {
                                image_part.Pixels[row, col] = blue[i + row, j + col];
                            }
                        }

                        blues.Add(multiplication(image_part, filter.Blue));
                    }
                }

                List<RGB> pooled = new List<RGB>();
                for (int i = 0; i < blues.Count; i++)
                {
                    pooled.Add(new RGB(reds[i].Pool(2), greens[i].Pool(2), blues[i].Pool(2)));
                }

                int index = 0;
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < width; y++)
                    {
                        first_red[x, y] = (int)pooled[index].Red.Max;
                        first_green[x, y] = (int)pooled[index].Green.Max;
                        first_blue[x, y] = (int)pooled[index].Blue.Max;
                        index++;
                    }
                }

                Bitmap bitmap = new Bitmap(width, width, PixelFormat.Format32bppArgb);
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < width; y++)
                    {
                        Color color = Color.FromArgb(255, (int)first_red[x, y] / 255, (int)first_green[x, y] / 255, (int)first_blue[x, y] / 255);
                        bitmap.SetPixel(x, y, color);
                    }
                }

                bitmap.Save(string.Format("probak/conv1_{0}.bmp", a_index++), ImageFormat.Bmp);
            }

            double[,] pooled_first_red = new double[47, 47];
            for (int i = 0; i < width; i += 2)
            {
                for (int j = 0; j < width; j += 2)
                {
                    try
                    {
                        ImagePart part = new ImagePart(2);
                        for (int row = 0; row < 2; row++)
                        {
                            for (int col = 0; col < 2; col++)
                            {
                                part.Pixels[row, col] = first_red[i + row, j + col];
                            }
                        }
                        pooled_first_red[i / 2, j / 2] = part.Max;
                    }
                    catch (Exception)
                    {
                    }
                }
            }

            double[,] pooled_first_green = new double[47, 47];
            for (int i = 0; i < width; i += 2)
            {
                for (int j = 0; j < width; j += 2)
                {
                    try
                    {
                        ImagePart part = new ImagePart(2);
                        for (int row = 0; row < 2; row++)
                        {
                            for (int col = 0; col < 2; col++)
                            {
                                part.Pixels[row, col] = first_green[i + row, j + col];
                            }
                        }
                        pooled_first_green[i / 2, j / 2] = part.Max;
                    }
                    catch (Exception)
                    {
                    }
                }
            }

            double[,] pooled_first_blue = new double[47, 47];
            for (int i = 0; i < width; i += 2)
            {
                for (int j = 0; j < width; j += 2)
                {
                    try
                    {
                        ImagePart part = new ImagePart(2);
                        for (int row = 0; row < 2; row++)
                        {
                            for (int col = 0; col < 2; col++)
                            {
                                part.Pixels[row, col] = first_blue[i + row, j + col];
                            }
                        }
                        pooled_first_blue[i / 2, j / 2] = part.Max;
                    }
                    catch (Exception)
                    {
                    }
                }
            }

            width = 47;
            Bitmap bitmap1 = new Bitmap(width, width, PixelFormat.Format32bppArgb);
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < width; y++)
                {
                    double r = pooled_first_red[x, y];
                    while (r > 255)
                    {
                        r /= 255;
                    }

                    double g = pooled_first_green[x, y];
                    while (g > 255)
                    {
                        g /= 255;
                    }

                    double b = pooled_first_blue[x, y];
                    while (b > 255)
                    {
                        b /= 255;
                    }
                    Color color = Color.FromArgb(255, (int)r, (int)g, (int)b);
                    bitmap1.SetPixel(x, y, color);
                }
            }

            bitmap1.Save(string.Format("probak/conv1_pool_{0}.bmp", a_index++), ImageFormat.Bmp);





            size = 5;
            width = 43;

            double[,] second_red = new double[width, width];
            double[,] second_green = new double[width, width];
            double[,] second_blue = new double[width, width];

            foreach (RGB filter in filters2)
            {
                //RED
                List<ImagePart> reds = new List<ImagePart>();
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        try
                        {
                            ImagePart image_part = new ImagePart(size);
                            for (int row = 0; row < size; row++)
                            {
                                for (int col = 0; col < size; col++)
                                {
                                    image_part.Pixels[row, col] = pooled_first_red[i + row, j + col];
                                }
                            }

                            reds.Add(multiplication(image_part, filter.Red));
                        }
                        catch (Exception)
                        {
                        }
                    }
                }

                //GREEN
                List<ImagePart> greens = new List<ImagePart>();
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        try
                        {
                            ImagePart image_part = new ImagePart(size);
                            for (int row = 0; row < size; row++)
                            {
                                for (int col = 0; col < size; col++)
                                {
                                    image_part.Pixels[row, col] = pooled_first_green[i + row, j + col];
                                }
                            }

                            greens.Add(multiplication(image_part, filter.Green));
                        }
                        catch (Exception)
                        {
                        }
                    }
                }

                //BLUE
                List<ImagePart> blues = new List<ImagePart>();
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        try
                        {
                            ImagePart image_part = new ImagePart(size);
                            for (int row = 0; row < size; row++)
                            {
                                for (int col = 0; col < size; col++)
                                {
                                    image_part.Pixels[row, col] = pooled_first_blue[i + row, j + col];
                                }
                            }

                            blues.Add(multiplication(image_part, filter.Blue));
                        }
                        catch (Exception)
                        {
                        }
                    }
                }

                List<RGB> pooled = new List<RGB>();
                for (int i = 0; i < blues.Count; i++)
                {
                    pooled.Add(new RGB(reds[i].Pool(2), greens[i].Pool(2), blues[i].Pool(2)));
                }

                int index = 0;
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < width; y++)
                    {
                        second_red[x, y] = (int)pooled[index].Red.Max;
                        second_green[x, y] = (int)pooled[index].Green.Max;
                        second_blue[x, y] = (int)pooled[index].Blue.Max;
                        index++;
                    }
                }

                Bitmap bitmap = new Bitmap(width, width, PixelFormat.Format32bppArgb);
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < width; y++)
                    {
                        double r = second_red[x, y];
                        while (r > 255)
                        {
                            r /= 255;
                        }

                        double g = second_green[x, y];
                        while (g > 255)
                        {
                            g /= 255;
                        }

                        double b = second_blue[x, y];
                        while (b > 255)
                        {
                            b /= 255;
                        }
                        Color color = Color.FromArgb(255, (int)r, (int)g, (int)b);
                        bitmap.SetPixel(x, y, color);
                    }
                }

                bitmap.Save(string.Format("probak/conv2_{0}.bmp", a_index++), ImageFormat.Bmp);
            }



            double[,] pooled_second_red = new double[21, 21];
            for (int i = 0; i < width; i += 2)
            {
                for (int j = 0; j < width; j += 2)
                {
                    try
                    {
                        ImagePart part = new ImagePart(2);
                        for (int row = 0; row < 2; row++)
                        {
                            for (int col = 0; col < 2; col++)
                            {
                                part.Pixels[row, col] = second_red[i + row, j + col];
                            }
                        }
                        pooled_second_red[i / 2, j / 2] = part.Max;
                    }
                    catch (Exception)
                    {
                    }
                }
            }

            double[,] pooled_second_green = new double[21, 21];
            for (int i = 0; i < width; i += 2)
            {
                for (int j = 0; j < width; j += 2)
                {
                    try
                    {
                        ImagePart part = new ImagePart(2);
                        for (int row = 0; row < 2; row++)
                        {
                            for (int col = 0; col < 2; col++)
                            {
                                part.Pixels[row, col] = second_green[i + row, j + col];
                            }
                        }
                        pooled_second_green[i / 2, j / 2] = part.Max;
                    }
                    catch (Exception)
                    {
                    }
                }
            }

            double[,] pooled_second_blue = new double[21, 21];
            for (int i = 0; i < width; i += 2)
            {
                for (int j = 0; j < width; j += 2)
                {
                    try
                    {
                        ImagePart part = new ImagePart(2);
                        for (int row = 0; row < 2; row++)
                        {
                            for (int col = 0; col < 2; col++)
                            {
                                part.Pixels[row, col] = second_blue[i + row, j + col];
                            }
                        }
                        pooled_second_blue[i / 2, j / 2] = part.Max;
                    }
                    catch (Exception)
                    {
                    }
                }
            }



            double[] layer = new double[21 * 21];
            int layer_index = 0;
            width = 21;
            Bitmap bitmap2 = new Bitmap(width, width, PixelFormat.Format32bppArgb);
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < width; y++)
                {
                    double r = pooled_second_red[x, y];
                    while (r > 255)
                    {
                        r /= 255;
                    }

                    double g = pooled_second_green[x, y];
                    while (g > 255)
                    {
                        g /= 255;
                    }

                    double b = pooled_second_blue[x, y];
                    while (b > 255)
                    {
                        b /= 255;
                    }
                    layer[layer_index++] = (r + g + b) / 1000;
                    Color color = Color.FromArgb(255, (int)r, (int)g, (int)b);
                    bitmap2.SetPixel(x, y, color);
                }
            }

            bitmap2.Save(string.Format("probak/conv2_pool_{0}.bmp", a_index++), ImageFormat.Bmp);


            Tuple<double[], short> last_layer;

            last_layer = new Tuple<double[], short>(layer, output);
            return last_layer;
        }
    }
}
