
import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.util.zip.GZIPInputStream;
/** Downloads missing Fashion-MNIST files into a reusable project-local cache. */
public final class FashionMnistDownloader {
 public static final String DEFAULT_DIRECTORY="data/fashion";
 private static final String BASE="https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/";
 private static final String[] FILES={"train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz","t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz"};
 private FashionMnistDownloader(){}
 public static void main(String[] args) throws IOException {
  ensureAvailable(args.length == 0 ? DEFAULT_DIRECTORY : args[0], false);
 }
 public static void ensureAvailable(String directory, boolean testOnly) throws IOException {
  Path dir=Path.of(directory).toAbsolutePath().normalize(); Files.createDirectories(dir);
  System.out.println("Dataset directory: "+dir);
  for(int i=testOnly?2:0;i<FILES.length;i++){
   Path dest=dir.resolve(FILES[i]);
   if(Files.isRegularFile(dest)&&Files.size(dest)>0)continue;
   Path temp=Files.createTempFile(dir,FILES[i],".part");
   try{
    System.out.println("Downloading "+FILES[i]);
    URLConnection conn=URI.create(BASE+FILES[i]).toURL().openConnection();
    conn.setConnectTimeout(30000); conn.setReadTimeout(60000);
    try(InputStream in=conn.getInputStream()){Files.copy(in,temp,StandardCopyOption.REPLACE_EXISTING);}
    try(DataInputStream in=new DataInputStream(new GZIPInputStream(Files.newInputStream(temp)))){
     if(in.readInt()!=(i%2==0?2051:2049))throw new IOException("Invalid IDX file: "+FILES[i]);
     in.transferTo(OutputStream.nullOutputStream());
    }
    Files.move(temp,dest,StandardCopyOption.REPLACE_EXISTING);
   }catch(IOException e){throw new IOException("Cannot download "+FILES[i]+" into "+dir+": "+e.getMessage(),e);}
   finally{Files.deleteIfExists(temp);}
  }
 }
}
