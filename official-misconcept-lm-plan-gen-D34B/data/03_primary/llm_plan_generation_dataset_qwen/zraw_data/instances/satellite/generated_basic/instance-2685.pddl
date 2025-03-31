(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	infrared0 - mode
	thermograph3 - mode
	infrared1 - mode
	image2 - mode
	groundstation0 - direction
	groundstation2 - direction
	groundstation3 - direction
	star1 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 image2)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument2 thermograph3)
	(supports instrument2 infrared0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star1)
	(supports instrument3 image2)
	(calibration_target instrument3 star1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
)
(:goal (and
	(have_image phenomenon4 thermograph3)
))

)
