package ToyBox::XS::LogitModel;

use 5.0080;
use strict;
use warnings;

require Exporter;

our $VERSION = '0.01';

require XSLoader;
XSLoader::load('ToyBox::XS::LogitModel', $VERSION);

sub add_instance {
    my ($self, %params) = @_;

    die "No params: attributes" unless defined($params{attributes});
    die "No params: label" unless defined($params{label});
    my $attributes = $params{attributes};
    my $label      = $params{label};
    die "attributes is not hash ref"   unless ref($attributes) eq 'HASH';
    die "attributes is empty hash ref" unless keys %$attributes;
    $label = [$label] unless ref($label) eq 'ARRAY';

    my %copy_attr = %$attributes;

    foreach my $l (@$label) {
        $self->xs_add_instance(\%copy_attr, $l);
    }
    1;
}

sub train{
    my ($self, %params) = @_;

    die "max_num should be greater than 0" if defined($params{max_iterations}) && $params{max_iterations} <= 0;
    die "sigma should be greater than 0" if defined($params{sigma}) && $params{sigma} <= 0;

    my $max_iterations   = $params{max_iterations} || 100;
    my $sigma            = $params{sigma}          || 1e2;
    my $algorithm        = $params{algorithm}      || "bfgs";

    $self->xs_train($max_iterations, $algorithm, $sigma);
    1;
}

sub predict {
    my ($self, %params) = @_;

    die "No params: attributes" unless defined($params{attributes});
    my $attributes = $params{attributes};
    die "attributes is not hash ref"   unless ref($attributes) eq 'HASH';
    die "attributes is empty hash ref" unless keys %$attributes;

    my $result = $self->xs_predict($attributes);

    $result;
}


1;
__END__
=head1 NAME

ToyBox::XS::LogitModel - Discriminant Analysis with Logit Model

=head1 SYNOPSIS

  use ToyBox::XS::LogitModel;

  my $lm = ToyBox::XS::LogitModel->new();
  
  $lm->add_instance(
      attributes => {a => 2, b => 3},
      label => 'positive'
  );
  
  $lm->add_instance(
      attributes => {c => 3, d => 1},
      label => 'negative'
  );
  
  $lm->train(max_iterations => 100, algorithm => "bfgs", sigma => 1e2);
  
  my $probs = $lm->predict(
                  attributes => {a => 1, b => 1, d => 1, e =>1}
              );

=head1 DESCRIPTION

This module implements a logistic model.

=head1 AUTHOR

TAGAMI Yukihiro <tagami.yukihiro@gmail.com>

=head1 LICENSE

This software is distributed under the term of the GNU General Public License.

L<http://opensource.org/licenses/gpl-license.php>
