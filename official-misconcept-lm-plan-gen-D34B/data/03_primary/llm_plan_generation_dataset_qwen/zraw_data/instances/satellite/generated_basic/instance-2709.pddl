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
	thermograph3 - mode
	thermograph1 - mode
	image0 - mode
	infrared2 - mode
	groundstation3 - direction
	star0 - direction
	star1 - direction
	groundstation2 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument4 thermograph3)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star1)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
)
(:goal (and
	(have_image phenomenon4 image0)
))

)
