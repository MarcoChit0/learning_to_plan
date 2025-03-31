(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	thermograph2 - mode
	infrared1 - mode
	infrared0 - mode
	groundstation1 - direction
	groundstation3 - direction
	star5 - direction
	star4 - direction
	groundstation2 - direction
	groundstation0 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation1)
	(calibration_target instrument0 star4)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 star4)
	(supports instrument2 infrared1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 infrared0)
	(supports instrument3 infrared1)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star4)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet8)
	(supports instrument5 infrared1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 infrared1)
	(supports instrument6 thermograph2)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation2)
	(calibration_target instrument6 star5)
	(supports instrument7 infrared1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star5)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 groundstation3)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation3)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
	(supports instrument10 thermograph2)
	(supports instrument10 infrared1)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 infrared1)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 infrared1)
	(calibration_target instrument12 groundstation3)
	(calibration_target instrument12 groundstation2)
	(supports instrument13 thermograph2)
	(calibration_target instrument13 star4)
	(calibration_target instrument13 star5)
	(supports instrument14 infrared1)
	(supports instrument14 thermograph2)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation2)
	(calibration_target instrument14 star4)
	(supports instrument15 infrared1)
	(supports instrument15 thermograph2)
	(supports instrument15 infrared0)
	(calibration_target instrument15 groundstation0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite0 star7)
	(pointing satellite1 star4)
	(have_image planet6 infrared0)
	(have_image star7 infrared0)
	(have_image planet8 thermograph2)
	(have_image phenomenon9 infrared0)
))

)
