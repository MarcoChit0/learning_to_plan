(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	image1 - mode
	image0 - mode
	star1 - direction
	groundstation2 - direction
	groundstation3 - direction
	star4 - direction
	groundstation0 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image0)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star8)
	(supports instrument1 image0)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation0)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
)
(:goal (and
	(pointing satellite0 groundstation0)
	(have_image phenomenon5 image0)
	(have_image phenomenon6 image1)
	(have_image planet7 image0)
	(have_image star8 image1)
))

)
