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
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	spectrograph2 - mode
	thermograph3 - mode
	infrared1 - mode
	infrared0 - mode
	star0 - direction
	star1 - direction
	planet2 - direction
)
(:init
	(supports instrument0 thermograph3)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph2)
	(supports instrument1 infrared1)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument4 thermograph3)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star1)
	(supports instrument6 infrared0)
	(supports instrument6 spectrograph2)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 star0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star1)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet2)
	(supports instrument8 spectrograph2)
	(supports instrument8 infrared0)
	(supports instrument8 thermograph3)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 star1)
	(supports instrument10 infrared1)
	(supports instrument10 thermograph3)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 star1)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
)
(:goal (and
	(pointing satellite1 planet2)
	(have_image planet2 infrared1)
))

)
