(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite3 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite4 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	spectrograph2 - mode
	thermograph1 - mode
	infrared0 - mode
	spectrograph3 - mode
	star1 - direction
	star2 - direction
	star0 - direction
	planet3 - direction
	planet4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	star7 - direction
	star8 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 spectrograph2)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet3)
	(supports instrument1 spectrograph3)
	(supports instrument1 infrared0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star2)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star2)
	(supports instrument3 spectrograph2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star2)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star2)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star0)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star7)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star2)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared0)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 star0)
	(on_board instrument7 satellite4)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet3)
)
(:goal (and
	(pointing satellite0 phenomenon6)
	(pointing satellite1 planet3)
	(pointing satellite4 phenomenon6)
	(have_image planet3 spectrograph2)
	(have_image planet4 thermograph1)
	(have_image phenomenon5 infrared0)
	(have_image phenomenon6 infrared0)
	(have_image star7 spectrograph3)
	(have_image star8 thermograph1)
))

)
