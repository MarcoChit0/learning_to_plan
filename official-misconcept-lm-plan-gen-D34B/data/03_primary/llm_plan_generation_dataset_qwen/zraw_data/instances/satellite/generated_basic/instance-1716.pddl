(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	thermograph1 - mode
	infrared0 - mode
	star1 - direction
	star4 - direction
	groundstation2 - direction
	star0 - direction
	star3 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star3)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument3 infrared0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star3)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument4 thermograph1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star3)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite2 groundstation2)
	(have_image phenomenon5 thermograph1)
))

)
