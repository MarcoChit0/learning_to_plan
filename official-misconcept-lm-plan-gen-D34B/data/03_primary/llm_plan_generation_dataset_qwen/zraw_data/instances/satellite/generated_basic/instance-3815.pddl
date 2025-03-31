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
	groundstation0 - direction
	star3 - direction
	groundstation5 - direction
	groundstation2 - direction
	star4 - direction
	groundstation1 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation2)
	(calibration_target instrument1 star4)
	(supports instrument2 image0)
	(calibration_target instrument2 star4)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation1)
	(calibration_target instrument3 star4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
)
(:goal (and
	(pointing satellite1 star6)
	(have_image star6 image0)
	(have_image star7 image0)
	(have_image planet8 image0)
))

)
