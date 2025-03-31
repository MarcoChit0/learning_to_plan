(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	image3 - mode
	image1 - mode
	image2 - mode
	image0 - mode
	star0 - direction
	star1 - direction
	planet2 - direction
	star3 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 image2)
	(calibration_target instrument0 star1)
	(supports instrument1 image3)
	(supports instrument1 image0)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(supports instrument3 image0)
	(supports instrument3 image1)
	(supports instrument3 image2)
	(calibration_target instrument3 star0)
	(supports instrument4 image2)
	(supports instrument4 image0)
	(supports instrument4 image3)
	(calibration_target instrument4 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument5 image2)
	(supports instrument5 image0)
	(calibration_target instrument5 star1)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
)
(:goal (and
	(pointing satellite2 star3)
	(have_image planet2 image0)
	(have_image star3 image3)
))

)
