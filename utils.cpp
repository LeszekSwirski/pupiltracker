#include "stdafx.h"
#include "utils.h"

static boost::mt19937 static_gen;
int random(int min, int max)
{
	boost::uniform_int<> dist(min, max);
	boost::variate_generator<boost::mt19937&, boost::uniform_int<> > die(static_gen, dist);

	return die();
}
int random(int min, int max, unsigned int seed)
{
	boost::mt19937 gen;
	gen.seed(seed);
	boost::uniform_int<> dist(min, max);
	boost::variate_generator<boost::mt19937&, boost::uniform_int<> > die(gen, dist);

	return die();
}