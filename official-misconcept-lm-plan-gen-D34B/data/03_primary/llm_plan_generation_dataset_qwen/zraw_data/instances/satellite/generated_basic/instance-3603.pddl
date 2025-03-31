(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	image1 - mode
	image0 - mode
	groundstation0 - direction
	groundstation2 - direction
	groundstation4 - direction
	star1 - direction
	star3 - direction
	planet5 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument1 image1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation4)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument2 image1)
	(calibration_target instrument2 star1)
	(supports instrument3 image1)
	(supports instrument3 image0)
	(calibration_target instrument3 star3)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet5)
)
(:goal (and
	(pointing satellite2 star1)
	(have_image planet5 image0)
))

)
