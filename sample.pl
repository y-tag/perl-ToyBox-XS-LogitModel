#!/usr/bin/perl

use strict;
use warnings;

use Data::Dumper;

use lib qw(lib blib/lib blib/arch);
use ToyBox::XS::LogisticModel;

my $lm = ToyBox::XS::LogisticModel->new();

$lm->add_instance(attributes => {a => 2, b => 3}, label => 1);

my $attributes = {c => 1, d => 4};
my $label = 0;
$lm->add_instance(attributes => $attributes, label => $label);

$lm->train();

my $result = $lm->predict(attributes => {a => 2, b => 3});
print Dumper($result);

$attributes = {a => 1, b => 1, c => 1, d => 1};
$result = $lm->predict(attributes => $attributes);
print Dumper($result);
