(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	thermograph0 - mode
	image1 - mode
	infrared3 - mode
	image2 - mode
	star4 - direction
	groundstation0 - direction
	groundstation1 - direction
	star2 - direction
	groundstation3 - direction
	star5 - direction
	phenomenon6 - direction
	planet7 - direction
	star8 - direction
	planet9 - direction
	planet10 - direction
	phenomenon11 - direction
	phenomenon12 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 image2)
	(supports instrument0 image1)
	(calibration_target instrument0 star2)
	(supports instrument1 image2)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 image1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument4 image2)
	(supports instrument4 thermograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 star2)
	(supports instrument5 image2)
	(supports instrument5 image1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation3)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(have_image star5 image2)
	(have_image phenomenon6 infrared3)
	(have_image planet7 thermograph0)
	(have_image star8 image2)
	(have_image planet9 infrared3)
	(have_image planet10 infrared3)
	(have_image phenomenon11 image1)
	(have_image phenomenon12 image1)
))

)
