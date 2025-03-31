(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite3 - satellite
	instrument13 - instrument
	spectrograph1 - mode
	thermograph2 - mode
	infrared3 - mode
	infrared0 - mode
	star4 - direction
	star1 - direction
	star3 - direction
	groundstation0 - direction
	groundstation2 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	star7 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 thermograph2)
	(supports instrument1 infrared3)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph2)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument3 infrared3)
	(supports instrument3 infrared0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star1)
	(supports instrument4 thermograph2)
	(supports instrument4 infrared3)
	(calibration_target instrument4 star1)
	(supports instrument5 infrared3)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 infrared0)
	(supports instrument6 infrared3)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star4)
	(supports instrument7 infrared3)
	(supports instrument7 infrared0)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 star4)
	(supports instrument8 infrared3)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 groundstation2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument9 infrared3)
	(calibration_target instrument9 star1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star1)
	(supports instrument11 infrared0)
	(supports instrument11 spectrograph1)
	(calibration_target instrument11 star3)
	(supports instrument12 spectrograph1)
	(calibration_target instrument12 groundstation0)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument13 infrared3)
	(calibration_target instrument13 groundstation2)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
)
(:goal (and
	(pointing satellite0 star3)
	(pointing satellite3 star3)
	(have_image phenomenon5 thermograph2)
	(have_image phenomenon6 infrared3)
	(have_image star7 infrared3)
))

)
