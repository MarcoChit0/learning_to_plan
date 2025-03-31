(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	image0 - mode
	groundstation0 - direction
	star1 - direction
	groundstation2 - direction
	groundstation3 - direction
	phenomenon4 - direction
	planet5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation2)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation3)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
)
(:goal (and
	(have_image phenomenon4 image0)
	(have_image planet5 image0)
	(have_image phenomenon6 image0)
))

)
