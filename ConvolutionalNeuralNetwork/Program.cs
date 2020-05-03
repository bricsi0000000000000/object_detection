using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;

namespace ConvolutionalNeuralNetwork
{
    class Program
    {
        static string layer1_filters_folder_path = "filters/layer1";
        static string layer2_filters_folder_path = "filters/layer2";

        static string train_images_cones_folder_path = "train_images/cone";
        static string train_images_not_cones_folder_path = "train_images/not_cone";

        static string test_images_cones_folder_path = "test_images/cones";
        static string test_images_not_cones_folder_path = "test_images/not_cones";

        static void Main(string[] args)
        {
            Stopwatch all_stopwatch = new Stopwatch();
            all_stopwatch.Start();

            List<RGB> filters1 = new List<RGB>();
            List<RGB> filters2 = new List<RGB>();
            initFilters(ref filters1, ref filters2);

            bool learn = false;
            Console.Write("Would you like to train this model? [Y/n] ");
            string learn_input = Console.ReadLine();
            if (learn_input.Equals("") || learn_input.Equals("y") || learn_input.Equals("Y"))
            {
                learn = true;
            }

            List<double> weights = new List<double>();
            List<double> costs = new List<double>();
            double bias = 0;
            initWeights(ref weights, ref bias, ref learn);

            if (learn)
            {
                Console.WriteLine();

                List<Tuple<string, short>> inputs = new List<Tuple<string, short>>();
                initTrainImages(ref inputs);

                List<Tuple<double[], short>> data = new List<Tuple<double[], short>>();
                for (int i = 0; i < inputs.Count; i++)
                {
                    data.Add(makeLayers(inputs[i].Item1, ref filters1, ref filters2, true, inputs.Count, i, inputs[i].Item2));
                }
                double learning_rate = 0.01;
                Console.Write("Would you like to change the learning rate? Currently it's {0}. [Y/n] ", learning_rate);
                string overwrite_input = Console.ReadLine();
                if (overwrite_input.Equals("") || overwrite_input.Equals("y") || overwrite_input.Equals("Y"))
                {
                    Console.Write("New learning rate: ");
                    learning_rate = double.Parse(Console.ReadLine());
                }

                Learn(ref bias, ref costs, ref data, ref weights, ref learning_rate);
            }

            List<string> test_cones_inputs = new List<string>();
            List<string> test_not_cones_inputs = new List<string>();

            initTestConeImages(ref test_cones_inputs);
            initTestNotConeImages(ref test_not_cones_inputs);

            Console.Write("What should be the interval of the convolving step? ");
            int i_volume = int.Parse(Console.ReadLine());

            Console.WriteLine("\nTest cones");
            makeTestConesOutputs(ref test_cones_inputs, ref weights, ref bias, ref filters1, ref filters2, true, ref i_volume);

            Console.WriteLine("\nTest not cones");
            makeTestConesOutputs(ref test_cones_inputs, ref weights, ref bias, ref filters1, ref filters2, false, ref i_volume);

            all_stopwatch.Stop();

            Console.WriteLine("Finished in {0:hh\\:mm\\:ss}", all_stopwatch.Elapsed);
            Console.ReadKey();
        }

        static void Learn(ref double bias, ref List<double> costs, ref List<Tuple<double[], short>> data, ref List<double> weights, ref double learning_rate)
        {
            Stopwatch learn_stopwatch = new Stopwatch();
            learn_stopwatch.Start();

            Console.WriteLine("Learning\n");

            int iterations = 0;
            double last_cost = double.MaxValue;
            Random rand = new Random();

            while (last_cost >= 0.03)
            {
                Tuple<double[], short> random_data = data[rand.Next(0, data.Count - 1)];

                double weighted_average = 0;
                for (int index = 0; index < weights.Count; index++)
                {
                    weighted_average += random_data.Item1[index] * weights[index];
                }
                weighted_average += bias;

                double prediction = sigmoid(weighted_average);

                double target = random_data.Item2;

                if (iterations % 100 == 0)
                {
                    double act_cost = 0;
                    for (int i = 0; i < data.Count; i++)
                    {
                        Tuple<double[], short> act_data = data[i];
                        double act_weighted_average = 0;
                        for (int j = 0; j < weights.Count; j++)
                        {
                            act_weighted_average += act_data.Item1[j] * weights[j];
                        }
                        act_weighted_average += bias;

                        double act_prediction = sigmoid(act_weighted_average);

                        act_cost += Math.Pow(act_prediction - act_data.Item2, 2);
                    }
                    costs.Add(act_cost);
                    last_cost = act_cost;
                    if (iterations % 10000 == 0)
                    {
                        Console.SetCursorPosition(0, Console.CursorTop - 1);
                        ClearCurrentConsoleLine();
                        Console.WriteLine("iteration {0}\t{1}", iterations, act_cost);
                    }
                }

                double derivative_cost_of_prediction = 2 * (prediction - target);
                double derivative_prediction_of_weighted_average = sigmoid_prime(weighted_average);

                List<double> derivative_weighted_average_weights = new List<double>();
                for (int index = 0; index < random_data.Item1.Length; index++)
                {
                    derivative_weighted_average_weights.Add(random_data.Item1[index]);
                }

                int derivative_weighted_average_bias = 1;

                double derivative_cost_derivative_weighted_average = derivative_cost_of_prediction * derivative_prediction_of_weighted_average;

                List<double> derivative_cost_weights = new List<double>();
                for (int index = 0; index < derivative_weighted_average_weights.Count; index++)
                {
                    derivative_cost_weights.Add(derivative_cost_derivative_weighted_average * derivative_weighted_average_weights[index]);
                }

                double derivative_cost_derivative_bias = derivative_cost_derivative_weighted_average * derivative_weighted_average_bias;

                for (int index = 0; index < weights.Count; index++)
                {
                    weights[index] -= learning_rate * derivative_cost_weights[index];
                }

                bias -= learning_rate * derivative_cost_derivative_bias;

                iterations++;
            }

            learn_stopwatch.Stop();

            Console.WriteLine("Finished learning in {0:hh\\:mm\\:ss}", learn_stopwatch.Elapsed);
            Console.WriteLine("Saving weights.");
            string file_name = string.Format("weights_{0}.csv", weights.Count);
            if (File.Exists(file_name))
            {
                Console.WriteLine("There is already a weights file called '{0}'. Would you like to overwrite it? [Y/n] ", file_name);
                string overwrite_input = Console.ReadLine();
                if (!(overwrite_input.Equals("") || overwrite_input.Equals("y") || overwrite_input.Equals("Y")))
                {
                    Console.Write("Save weights file name: ");
                    file_name = Console.ReadLine();
                }
            }

            StreamWriter sw = new StreamWriter(file_name);
            sw.WriteLine(bias);
            foreach (var item in weights)
            {
                sw.WriteLine(item);
            }
            sw.Close();
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

        static Tuple<double[], short> makeLayers(string file_name, ref List<RGB> filters1, ref List<RGB> filters2, bool write_to_console, int allfilesconunt, int actfileindex, short output = 10)
        {
            if (write_to_console)
            {
                Console.SetCursorPosition(0, Console.CursorTop - 1);
                ClearCurrentConsoleLine();
                Console.WriteLine("Processing image: {0}/{1}\t{2} ", actfileindex + 1, allfilesconunt, file_name);
            }

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

            foreach (RGB filter in filters1)
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
                                    image_part.Pixels[row, col] = red[i + row, j + col];
                                }
                            }

                            reds.Add(multiplication(image_part, filter.Red));
                        }
                        catch (Exception) { }

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
                                    image_part.Pixels[row, col] = green[i + row, j + col];
                                }
                            }

                            greens.Add(multiplication(image_part, filter.Green));
                        }
                        catch (Exception) { }
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
                                    image_part.Pixels[row, col] = blue[i + row, j + col];
                                }
                            }

                            blues.Add(multiplication(image_part, filter.Blue));
                        }
                        catch (Exception) { }

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
                        first_red[x, y] = relu(pooled[index].Red.Max);
                        first_green[x, y] = relu(pooled[index].Green.Max);
                        first_blue[x, y] = relu(pooled[index].Blue.Max);
                        index++;
                    }
                }
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
                    catch (Exception) { }
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
                    catch (Exception) { }
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
                    catch (Exception) { }
                }
            }

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
                        second_red[x, y] = relu(pooled[index].Red.Max);
                        second_green[x, y] = relu(pooled[index].Green.Max);
                        second_blue[x, y] = relu(pooled[index].Blue.Max);
                        index++;
                    }
                }
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
                    catch (Exception) { }
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
                    catch (Exception) { }
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
                    catch (Exception) { }
                }
            }

            width = 21;
            double[] layer = new double[width * width];
            int layer_index = 0;
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
                }
            }

            Tuple<double[], short> last_layer;

            last_layer = new Tuple<double[], short>(layer, output);
            return last_layer;
        }

        static Tuple<double[], short, Point, Point, Point, Point> makeLayers(RGB image, ref List<RGB> filters, ref List<RGB> filters2, short output = 10)
        {
            double[,] red = image.Red.Pixels;
            double[,] green = image.Red.Pixels;
            double[,] blue = image.Red.Pixels;

            int size = 7;
            int width = 94;

            double[,] first_red = new double[width, width];
            double[,] first_green = new double[width, width];
            double[,] first_blue = new double[width, width];

            foreach (RGB filter in filters)
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
                                    image_part.Pixels[row, col] = red[i + row, j + col];
                                }
                            }

                            reds.Add(multiplication(image_part, filter.Red));
                        }
                        catch (Exception) { }
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
                                    image_part.Pixels[row, col] = green[i + row, j + col];
                                }
                            }

                            greens.Add(multiplication(image_part, filter.Green));
                        }
                        catch (Exception) { }

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
                                    image_part.Pixels[row, col] = blue[i + row, j + col];
                                }
                            }

                            blues.Add(multiplication(image_part, filter.Blue));
                        }
                        catch (Exception) { }
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
                        first_red[x, y] = relu(pooled[index].Red.Max);
                        first_green[x, y] = relu(pooled[index].Green.Max);
                        first_blue[x, y] = relu(pooled[index].Blue.Max);
                        index++;
                    }
                }
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
                    catch (Exception) { }
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
                    catch (Exception) { }
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
                    catch (Exception) { }
                }
            }

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
                        catch (Exception) { }
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
                        catch (Exception) { }
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
                        catch (Exception) { }
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
                        second_red[x, y] = relu(pooled[index].Red.Max);
                        second_green[x, y] = relu(pooled[index].Green.Max);
                        second_blue[x, y] = relu(pooled[index].Blue.Max);
                        index++;
                    }
                }
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
                    catch (Exception) { }
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
                    catch (Exception) { }
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
                    catch (Exception) { }
                }
            }


            width = 21;
            double[] layer = new double[width * width];
            int layer_index = 0;
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
                }
            }

            Tuple<double[], short, Point, Point, Point, Point> last_layer;

            last_layer = new Tuple<double[], short, Point, Point, Point, Point>(layer, output, image.Red.LeftTopIndex, image.Red.LeftBottomIndex, image.Red.RightBottomIndex, image.Red.RightTopIndex);
            return last_layer;
        }

        static double sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        static double sigmoid_prime(double x)
        {
            return sigmoid(x) * (1 - sigmoid(x));
        }

        static double relu(double x)
        {
            return Math.Max(0, x);
        }

        public static void ClearCurrentConsoleLine()
        {
            int currentLineCursor = Console.CursorTop;
            Console.SetCursorPosition(0, Console.CursorTop);
            Console.Write(new string(' ', Console.WindowWidth));
            Console.SetCursorPosition(0, currentLineCursor);
        }

        static void initFilters(ref List<RGB> filters1, ref List<RGB> filters2)
        {
            DirectoryInfo layer1_filters_directory = new DirectoryInfo(layer1_filters_folder_path);
            FileInfo[] Files = layer1_filters_directory.GetFiles();
            foreach (FileInfo file in Files)
            {
                addFilter(string.Format("{0}/{1}", layer1_filters_folder_path, file.Name), ref filters1, 7);
            }

            Console.WriteLine("{0} layer1 filters are initialized.", filters1.Count);

            DirectoryInfo layer2_filters_directory = new DirectoryInfo(layer2_filters_folder_path);
            Files = layer2_filters_directory.GetFiles();
            foreach (FileInfo file in Files)
            {
                addFilter(string.Format("{0}/{1}", layer2_filters_folder_path, file.Name), ref filters2, 5);
            }

            Console.WriteLine("{0} layer2 filters are initialized.", filters2.Count);
            Console.WriteLine();
        }

        static void initTrainImages(ref List<Tuple<string, short>> inputs)
        {
            DirectoryInfo train_images_cones_directory = new DirectoryInfo(train_images_cones_folder_path);
            FileInfo[] Files = train_images_cones_directory.GetFiles();
            foreach (FileInfo file in Files)
            {
                inputs.Add(new Tuple<string, short>(string.Format("{0}/{1}", train_images_cones_folder_path, file.Name), 1));
            }

            Console.WriteLine("{0} cone train images are initialized.", Files.Length);

            DirectoryInfo train_images_not_cones_directory = new DirectoryInfo(train_images_not_cones_folder_path);
            Files = train_images_not_cones_directory.GetFiles();
            foreach (FileInfo file in Files)
            {
                inputs.Add(new Tuple<string, short>(string.Format("{0}/{1}", train_images_not_cones_folder_path, file.Name), 0));
            }

            Console.WriteLine("{0} not cone train images are initialized.", Files.Length);
            Console.WriteLine();
            Console.WriteLine();
        }

        static void initWeights(ref List<double> weights, ref double bias, ref bool learn)
        {
            if (learn)
            {
                Random rand = new Random();
                for (int i = 0; i < 300; i++)
                {
                    weights.Add(rand.NextDouble());
                }
                bias = rand.NextDouble();
            }
            else
            {
                bool file_exists = false;
                do
                {
                    Console.Write("Weights file name: ");
                    string file_name = Console.ReadLine();
                    if (File.Exists(file_name))
                    {
                        StreamReader sr = new StreamReader(file_name);
                        bias = double.Parse(sr.ReadLine());
                        while (!sr.EndOfStream)
                        {
                            weights.Add(double.Parse(sr.ReadLine()));
                        }

                        file_exists = true;
                    }
                    else
                    {
                        Console.WriteLine("File name '{0}' doesn't exists", file_name);
                    }
                }
                while (!file_exists);
            }
        }

        static void initTestConeImages(ref List<string> test_cones_inputs)
        {
            DirectoryInfo test_images_cones_directory = new DirectoryInfo(test_images_cones_folder_path);
            FileInfo[] Files = test_images_cones_directory.GetFiles();
            foreach (FileInfo file in Files)
            {
                test_cones_inputs.Add(string.Format("{0}/{1}", test_images_cones_folder_path, file.Name));
            }
        }

        static void initTestNotConeImages(ref List<string> test_not_cones_inputs)
        {
            DirectoryInfo test_images_not_cones_directory = new DirectoryInfo(test_images_not_cones_folder_path);
            FileInfo[] Files = test_images_not_cones_directory.GetFiles();
            foreach (FileInfo file in Files)
            {
                test_not_cones_inputs.Add(string.Format("{0}/{1}", test_images_not_cones_folder_path, file.Name));
            }
        }

        static void makeTestConesOutputs(ref List<string> test_cones_inputs, ref List<double> weights, ref double bias, ref List<RGB> filters1, ref List<RGB> filters2, bool cones, ref int i_volume)
        {
            int save_image_index = 0;

            foreach (string test_input in test_cones_inputs)
            {
                Console.WriteLine();
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

                int j_to = image.Width - 100 - i_volume + 1;
                int i_to = image.Height - 100 - i_volume + 1;
                List<ImagePart> reds = new List<ImagePart>();
                for (int i = 0; i < i_to; i += i_volume)
                {
                    for (int j = 0; j < j_to; j += i_volume)
                    {
                        try
                        {
                            ImagePart image_part = new ImagePart(100);
                            image_part.LeftTopIndex = new Point(i, j);
                            image_part.LeftBottomIndex = new Point(i + 99, j);
                            image_part.RightBottomIndex = new Point(i + 99, j + 99);
                            image_part.RightTopIndex = new Point(i, j + 99);

                            for (int row = 0; row < 100; row++)
                            {
                                for (int col = 0; col < 100; col++)
                                {
                                    image_part.Pixels[row, col] = red[i + row, j + col];
                                }
                            }
                            reds.Add(image_part);
                        }
                        catch (Exception) { }
                    }
                }

                List<ImagePart> greens = new List<ImagePart>();
                for (int i = 0; i < i_to; i += i_volume)
                {
                    for (int j = 0; j < i_to; j += i_volume)
                    {
                        try
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
                        catch (Exception) { }
                    }
                }

                List<ImagePart> blues = new List<ImagePart>();
                for (int i = 0; i < i_to; i += i_volume)
                {
                    for (int j = 0; j < i_to; j += i_volume)
                    {
                        try
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
                        catch (Exception) { }
                    }
                }

                List<Tuple<double, Point, Point, Point, Point>> outputs = new List<Tuple<double, Point, Point, Point, Point>>();
                for (int i = 0; i < reds.Count; i++)
                {
                    Console.SetCursorPosition(0, Console.CursorTop - 1);
                    ClearCurrentConsoleLine();
                    Console.WriteLine("Making {0}/{1}", i + 1, reds.Count);

                    try
                    {
                        Tuple<double[], short, Point, Point, Point, Point> layers = makeLayers(new RGB(reds[i], greens[i], blues[i]), ref filters1, ref filters2);
                        mystery_cone = layers.Item1.ToList();

                        double z1 = 0;
                        for (int j = 0; j < weights.Count; j++)
                        {
                            z1 += weights[j] * mystery_cone[j];
                        }
                        z1 += bias;

                        outputs.Add(new Tuple<double, Point, Point, Point, Point>(sigmoid(z1), layers.Item3, layers.Item4, layers.Item5, layers.Item6));
                    }
                    catch (Exception) { }
                }

                double output = outputs.Average(a => a.Item1);

                List<Tuple<Point, Color>> edges = new List<Tuple<Point, Color>>();

                foreach (var item in outputs)
                {
                    if (item.Item1 >= .5)
                    {
                        for (int x = 0; x < image.Width; x++)
                        {
                            for (int y = 0; y < image.Width; y++)
                            {
                                try
                                {


                                    double r = red[x, y];
                                    double g = green[x, y];
                                    double b = blue[x, y];

                                    Point left_top = item.Item2;
                                    Point left_bot = item.Item3;
                                    Point right_bot = item.Item4;
                                    Point right_top = item.Item5;

                                    if (x == left_top.X && y >= left_top.Y && y <= right_top.Y ||
                                        x == left_bot.X && y >= left_bot.Y && y <= right_bot.Y ||
                                        y == left_top.Y && x >= left_top.X && x <= left_bot.X ||
                                        y == right_top.Y && x >= right_top.X && x <= right_bot.X
                                        )
                                    {
                                        if (item.Item1 >= .5 && item.Item1 < .6)
                                        {
                                            r = 255;
                                            g = 255;
                                            b = 0;
                                        }
                                        else if (item.Item1 >= .6 && item.Item1 < .7)
                                        {
                                            r = 0;
                                            g = 255;
                                            b = 0;
                                        }
                                        else if (item.Item1 >= .7 && item.Item1 < .8)
                                        {
                                            r = 0;
                                            g = 0;
                                            b = 255;
                                        }
                                        else if (item.Item1 >= .8 && item.Item1 < .9)
                                        {
                                            r = 255;
                                            g = 0;
                                            b = 255;
                                        }
                                        else if (item.Item1 >= .9)
                                        {
                                            r = 255;
                                            g = 0;
                                            b = 0;
                                        }

                                        edges.Add(new Tuple<Point, Color>(new Point(x, y), Color.FromArgb(255, (int)r, (int)g, (int)b)));
                                    }
                                }
                                catch (Exception) { }
                            }
                        }
                    }
                }

                Bitmap bitmap = new Bitmap(image.Width, image.Height, PixelFormat.Format32bppArgb);
                for (int x = 0; x < image.Width; x++)
                {
                    for (int y = 0; y < image.Width; y++)
                    {
                        try
                        {
                            double r = red[x, y];
                            double g = green[x, y];
                            double b = blue[x, y];

                            Color color = Color.FromArgb(255, (int)r, (int)g, (int)b);

                            foreach (var edge in edges)
                            {
                                if (x == edge.Item1.X && y == edge.Item1.Y)
                                {
                                    color = edge.Item2;
                                    break;
                                }
                            }

                            bitmap.SetPixel(x, y, color);
                        }
                        catch (Exception) { }
                    }
                }
                if (cones)
                {
                    bitmap.Save(string.Format("output_images/CONE_{0}.bmp", save_image_index++), ImageFormat.Bmp);
                }
                else
                {
                    bitmap.Save(string.Format("output_images/NOT_CONE_{0}.bmp", save_image_index++), ImageFormat.Bmp);
                }
            }
        }
    }
}
