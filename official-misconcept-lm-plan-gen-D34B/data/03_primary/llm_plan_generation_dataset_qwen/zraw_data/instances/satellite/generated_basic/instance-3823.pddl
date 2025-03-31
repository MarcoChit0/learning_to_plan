(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	image0 - mode
	star1 - direction
	star3 - direction
	star5 - direction
	groundstation0 - direction
	star2 - direction
	groundstation4 - direction
	phenomenon6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument1 image0)
	(calibration_target instrument1 star2)
	(supports instrument2 image0)
	(calibration_target instrument2 star2)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(pointing satellite0 planet8)
	(have_image phenomenon6 image0)
	(have_image star7 image0)
	(have_image planet8 image0)
))

)
