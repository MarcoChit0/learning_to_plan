(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	image0 - mode
	infrared1 - mode
	thermograph2 - mode
	star0 - direction
	star1 - direction
	groundstation4 - direction
	star3 - direction
	star2 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 thermograph2)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument1 image0)
	(calibration_target instrument1 star3)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
)
(:goal (and
	(pointing satellite1 phenomenon5)
	(have_image phenomenon5 infrared1)
))

)
