(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	spectrograph2 - mode
	thermograph3 - mode
	infrared0 - mode
	image1 - mode
	star0 - direction
	star1 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 spectrograph2)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star0)
	(supports instrument5 image1)
	(calibration_target instrument5 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument6 thermograph3)
	(supports instrument6 spectrograph2)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star1)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon2)
	(supports instrument7 infrared0)
	(supports instrument7 thermograph3)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 star0)
	(supports instrument8 image1)
	(supports instrument8 infrared0)
	(supports instrument8 thermograph3)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared0)
	(supports instrument9 image1)
	(supports instrument9 thermograph3)
	(calibration_target instrument9 star0)
	(supports instrument10 image1)
	(supports instrument10 thermograph3)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star1)
	(supports instrument11 image1)
	(supports instrument11 infrared0)
	(supports instrument11 thermograph3)
	(calibration_target instrument11 star0)
	(supports instrument12 spectrograph2)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star1)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(have_image phenomenon2 spectrograph2)
))

)
