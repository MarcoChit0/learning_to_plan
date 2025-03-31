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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	satellite5 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	image0 - mode
	spectrograph1 - mode
	groundstation3 - direction
	star2 - direction
	star1 - direction
	star0 - direction
	groundstation5 - direction
	star4 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star2)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 image0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star4)
	(calibration_target instrument1 star2)
	(supports instrument2 image0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star7)
	(supports instrument3 image0)
	(calibration_target instrument3 star4)
	(calibration_target instrument3 groundstation5)
	(supports instrument4 spectrograph1)
	(supports instrument4 image0)
	(calibration_target instrument4 star1)
	(calibration_target instrument4 groundstation5)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument5 spectrograph1)
	(supports instrument5 image0)
	(calibration_target instrument5 star0)
	(calibration_target instrument5 groundstation5)
	(supports instrument6 spectrograph1)
	(supports instrument6 image0)
	(calibration_target instrument6 star1)
	(calibration_target instrument6 star0)
	(supports instrument7 image0)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star2)
	(calibration_target instrument7 groundstation3)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 star2)
	(supports instrument9 image0)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 star2)
	(calibration_target instrument9 groundstation3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument10 image0)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 groundstation5)
	(calibration_target instrument10 star1)
	(on_board instrument10 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star6)
	(supports instrument11 spectrograph1)
	(supports instrument11 image0)
	(calibration_target instrument11 star4)
	(supports instrument12 spectrograph1)
	(calibration_target instrument12 star4)
	(supports instrument13 image0)
	(supports instrument13 spectrograph1)
	(calibration_target instrument13 star0)
	(supports instrument14 spectrograph1)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation5)
	(supports instrument15 image0)
	(calibration_target instrument15 star4)
	(on_board instrument11 satellite5)
	(on_board instrument12 satellite5)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star7)
)
(:goal (and
	(pointing satellite0 star4)
	(pointing satellite3 phenomenon8)
	(pointing satellite4 star4)
	(have_image star6 spectrograph1)
	(have_image star7 image0)
	(have_image phenomenon8 image0)
))

)
