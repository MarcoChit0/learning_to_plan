(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	infrared0 - mode
	infrared1 - mode
	thermograph2 - mode
	star0 - direction
	groundstation1 - direction
	phenomenon2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon2)
	(supports instrument1 infrared1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(supports instrument3 thermograph2)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
	(supports instrument4 thermograph2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared0)
	(supports instrument5 thermograph2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared0)
	(supports instrument7 infrared1)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph2)
	(supports instrument8 infrared1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon2)
	(supports instrument9 infrared0)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 star0)
	(supports instrument10 infrared0)
	(supports instrument10 infrared1)
	(supports instrument10 thermograph2)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 thermograph2)
	(calibration_target instrument11 star0)
	(supports instrument12 infrared0)
	(supports instrument12 thermograph2)
	(supports instrument12 infrared1)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 infrared1)
	(calibration_target instrument13 groundstation1)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
)
(:goal (and
	(pointing satellite2 groundstation1)
	(have_image phenomenon2 thermograph2)
	(have_image planet3 infrared0)
))

)
