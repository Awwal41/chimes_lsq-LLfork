#!/usr/bin/perl
#
# This program translates a LAMMPS xyz dump file into DFTB gen format.
# Please be careful not to use this program on a LAMMPS fractional
# coordinate dump file.  The atom_types variable below must be set to be
# consistent with the LAMMPS atom types.  The order of the atom types is
# significant.
#
$atom_types = "C N O H";
print "WARNING!! check atom types: $atom_types\n";
open (INF, "<$ARGV[0]") || die "Can't open input dump file\n";
$file = $ARGV[0];
$file =~ s/dump.//;
$file .= ".gen";
print "Outfile file: $file\n";
open (OUT, ">$file") || die "Can't open lmp_traj.gen\n";
while (<INF>) {
  $time_step = <INF>;
  chomp($time_step);
  $line = <INF>;
  $natom = <INF>;
  chomp($natom);
  print (OUT "  $natom S # time step $time_step\n");
  print (OUT "$atom_types\n");
  $line = <INF>;
  # assumes orthorhombic box
  @line = split " ", <INF>;
  $lx = $line[1] - $line[0];
  @line = split " ", <INF>;
  $ly = $line[1] - $line[0];
  @line = split " ", <INF>;
  $lz = $line[1] - $line[0];
  @line = split " ", <INF>;
  for ($i = 0; $i < @line; $i++) {
    if ($line[$i] =~ /id/) {
      $id_ind = $i-2;
    } elsif ($line[$i] =~ /type/) {
      $type_ind = $i - 2;
    } elsif ($line[$i] =~ /x/) {
      $x_ind = $i - 2;
    } elsif ($line[$i] =~ /y/) {
      $y_ind = $i - 2;
    } elsif ($line[$i] =~ /z/) {
      $z_ind = $i - 2;
    }
  }
  for ($i = 0; $i < $natom; $i++) {
    @line = split " ", <INF>;
    print (OUT "    $line[$id_ind] $line[$type_ind] $line[$x_ind] $line[$y_ind] $line[$z_ind]\n");
  }
  print (OUT "    0.0000000000E+00    0.0000000000E+00    0.0000000000E+00\n");
  print (OUT "    $lx    0.0000000000E+00    0.0000000000E+00\n");
  print (OUT "    0.0000000000E+00    $ly    0.0000000000E+00\n");
  print (OUT "    0.0000000000E+00    0.0000000000E+00    $lz\n");
}
close (INF);
close (OUT);
