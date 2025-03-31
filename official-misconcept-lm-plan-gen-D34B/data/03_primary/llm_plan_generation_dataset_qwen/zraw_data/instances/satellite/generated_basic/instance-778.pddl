(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	image0 - mode
	image1 - mode
	star0 - direction
	groundstation2 - direction
	star1 - direction
	groundstation3 - direction
	star4 - direction
	planet5 - direction
	planet6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(supports instrument1 image0)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument2 image0)
	(supports instrument2 image1)
	(calibration_target instrument2 star4)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite1 planet5)
	(have_image planet5 image0)
	(have_image planet6 image1)
	(have_image phenomenon7 image0)
	(have_image star8 image1)
))

)
