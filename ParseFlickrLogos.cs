using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;

/// <summary>
/// A tool for parsing the FlickrLogos32 (also including Logos32plus) dataset to KITTI or DARKNET format.
/// 
/// Usage example: "Path\to\FlickrLogos-32plus_dataset_v2\FlickrLogos-v2\classes" "0.1" "false" "500" "darknet"
/// </summary>
namespace Parsing
{
    class ParseFlickrLogos
    {
        enum Format
        {
            KITTI,
            DARKNET
        }

        /// <summary>
        /// A tool for parsing FlickrLogos-32 including bounding box in class subdirectories into label files in KITTI format.
        /// </summary>
        /// <param name="args">Directory path including subdirectories named by classes</param>
        static void Main(string[] args)
        {
            if (args.Length != 5)
            {
                Console.WriteLine("Missing parameter! 1: FlickrLogos-32 classes directory (\"...\\FlickrLogos-v2\\classes\"), 2: percentage for test set (e.g. 0.1), 3: use no-logo images? (e.g. false); 4: Max Side Length (e.g. 512 px); 5: label format (e.g. KITTI or darknet)");
                Console.ReadKey();
                return;
            }

            // // check if classes directory exists
            DirectoryInfo dir = new DirectoryInfo(args[0]);

            if (!dir.Exists)
            {
                Console.WriteLine("Directory does not exist or could not be found...");
                return;
            }

            // check if validation set percentage is valid
            float percentage;
            if (!float.TryParse(args[1], out percentage))
            {
                Console.WriteLine("No valid percentage float (e.g. 0.1) given for validation set splitting...");
                return;
            }

            // check if value for no-logo image usage is valid
            bool useNoLogo;
            if (!bool.TryParse(args[2], out useNoLogo))
            {
                Console.WriteLine("No valid bool value (e.g. false) given for using no-logo images...");
                return;
            }

            // check if value for max side length is valid
            int sideLength;
            if (!Int32.TryParse(args[3], out sideLength))
            {
                Console.WriteLine("No valid int value (e.g. 512) given for max side length of rescaled images...");
                return;
            }

            // check if value for label format is valid
            Format format;
            if (args[4].Equals("kitti"))
            {
                format = Format.KITTI;
            }
            else if (args[4].Equals("darknet"))
            {
                format = Format.DARKNET;
            }
            else
            {
                Console.WriteLine("No valid format (kitti or darknet) given for label parsing...");
                return;
            }

            // get jpg and masks directories
            DirectoryInfo imagesDir = new DirectoryInfo(dir.FullName + @"\jpg");
            DirectoryInfo labelsDir = new DirectoryInfo(dir.FullName + @"\masks");

            // check if directory exists
            if (!imagesDir.Exists || !labelsDir.Exists)
            {
                Console.WriteLine("jpg or masks directory could not be found...");
                return;
            }

            // clean previously created directory
            if (Directory.Exists(dir.FullName + @"\parsed32_" + sideLength))
            {
                Directory.Delete(dir.FullName + @"\parsed32_" + sideLength, true);
            }

            // create directories for parsed files
            DirectoryInfo imagesTrainDir = Directory.CreateDirectory(dir.FullName + @"\parsed32_" + sideLength + "_" + args[4] + @"\train\images");
            DirectoryInfo labelsTrainDir = Directory.CreateDirectory(dir.FullName + @"\parsed32_" + sideLength + "_" + args[4] + @"\train\labels");
            DirectoryInfo imagesValDir = Directory.CreateDirectory(dir.FullName + @"\parsed32_" + sideLength + "_" + args[4] + @"\val\images");
            DirectoryInfo labelsValDir = Directory.CreateDirectory(dir.FullName + @"\parsed32_" + sideLength + "_" + args[4] + @"\val\labels");

            DirectoryInfo[] subDirs = imagesDir.GetDirectories();

            FileInfo[] images;
            string className, fileName;

            string[] lines;
            Image img;
            float ratio;
            int classNumber = 0;

            using (StreamWriter classesWriter = File.CreateText(labelsTrainDir.FullName + @"\..\..\classes.txt"))
            {
                using (StreamWriter truncWriter = File.CreateText(labelsTrainDir.FullName + @"\..\..\mayBeTruncated.txt"))
                {
                    truncWriter.WriteLine("Please check the following files for truncation. The objects are marked as \"not truncated\" right now, but have their bounding box on the edge of the image.");

                    for (int i = 0; i < subDirs.Length; i++)
                    {
                        className = subDirs[i].Name;

                        // get all jpg images and calculate splitting amounts
                        images = subDirs[i].GetFiles("*.jpg");
                        int amountForTesting = (int)(images.Length * percentage);

                        if (!className.Equals("no-logo"))
                        {
                            Console.Write(classNumber + " class \"" + className + "\" - split into (train, " + (images.Length - amountForTesting) + "), (test, " + amountForTesting + ")");

                            for (int j = 0; j < images.Length; j++)
                            {
                                fileName = Path.GetFileNameWithoutExtension(images[j].Name);

                                img = new Bitmap(images[j].FullName);
                                img = ScaleImage(img, sideLength, out ratio);

                                // check if image for training or testing
                                string labelPath;
                                if (amountForTesting <= 0)
                                {
                                    img.Save(imagesTrainDir.FullName + "\\" + fileName + ".png");
                                    labelPath = labelsTrainDir.FullName;
                                }
                                else
                                {
                                    img.Save(imagesValDir.FullName + "\\" + fileName + ".png");
                                    labelPath = labelsValDir.FullName;
                                }

                                // write bounding box information into KITTI format
                                using (StreamWriter writer = File.CreateText(labelPath + @"\" + fileName + @".txt"))
                                {
                                    lines = System.IO.File.ReadAllLines(images[j].FullName + "\\..\\..\\..\\masks\\" + className + "\\" + fileName + ".jpg.bboxes.txt");
                                    for (int k = 1; k < lines.Length; k++)
                                    {
                                        string[] values = lines[k].Split(' ');

                                        // write label file in correct format
                                        if (format == Format.KITTI)
                                        {
                                            int x1 = (int)(float.Parse(values[0]) * ratio);
                                            int y1 = (int)(float.Parse(values[1]) * ratio);
                                            int x2 = (int)((float.Parse(values[0]) + float.Parse(values[2])) * ratio);
                                            int y2 = (int)((float.Parse(values[1]) + float.Parse(values[3])) * ratio);

                                            writer.WriteLine(UppercaseFirst(className) + " 0.00 0 0.0 " + x1 + " " + y1 + " " + x2 + " " + y2 + " 0.0 0.0 0.0 0.0 0.0 0.0 0.0");
                                        }
                                        else if (format == Format.DARKNET)
                                        {
                                            int x1 = (int)((float.Parse(values[0]) + 0.5f * float.Parse(values[2])) * ratio);
                                            int y1 = (int)((float.Parse(values[1]) + 0.5f * float.Parse(values[3])) * ratio);
                                            int x2 = (int)(float.Parse(values[2]) * ratio);
                                            int y2 = (int)(float.Parse(values[3]) * ratio);


                                            if (x1 == 0)
                                            {
                                                x1 = 1;
                                            }
                                            if (y1 == 0)
                                            {
                                                y1 = 1;
                                            }

                                            if (x1 + x2 > (float)img.Width)
                                            {
                                                x2 -= 1;
                                            }
                                            if (y1 + y2 > (float)img.Height)
                                            {
                                                y2 -= 1;
                                            }

                                            writer.WriteLine(classNumber + " " + x1 / (float)img.Width + " " + y1 / (float)img.Height + " " + x2 / (float)img.Width + " " + y2 / (float)img.Height);
                                        }

                                        // TODO check other value too! 
                                        if (float.Parse(values[0]) == 0 || float.Parse(values[1]) == 0)
                                        {
                                            truncWriter.WriteLine(fileName);
                                        }
                                    }
                                }

                                img.Dispose();
                                amountForTesting--;
                            }
                        }

                        if (className.Equals("no-logo") && useNoLogo || !className.Equals("no-logo"))
                        {
                            classesWriter.Write(UppercaseFirst(className) + ",");
                            classNumber++;
                            Console.WriteLine(" ... finished");
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Capitalizes the first letter of the given string.
        /// </summary>
        /// <param name="s">String to capitalize</param>
        /// <returns>Given string with capitalized first letter</returns>
        static string UppercaseFirst(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return string.Empty;
            }
            char[] a = s.ToCharArray();
            a[0] = char.ToUpper(a[0]);
            return new string(a);
        }

        public static Image ScaleImage(Image image, int maxSideLength, out float ratio)
        {
            float ratioX = (float)maxSideLength / image.Width;
            float ratioY = (float)maxSideLength / image.Height;
            ratio = Math.Min(ratioX, ratioY);

            int newWidth = (int)(image.Width * ratio);
            int newHeight = (int)(image.Height * ratio);

            Bitmap newImage = new Bitmap(newWidth, newHeight);

            using (Graphics graphics = Graphics.FromImage(newImage))
                graphics.DrawImage(image, 0, 0, newWidth, newHeight);

            return newImage;
        }
    }
}
