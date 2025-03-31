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
	instrument9 - instrument
	satellite2 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	spectrograph1 - mode
	image0 - mode
	spectrograph3 - mode
	image2 - mode
	star1 - direction
	star2 - direction
	star0 - direction
	star3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 spectrograph3)
	(supports instrument0 image2)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 star0)
	(supports instrument2 image2)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star0)
	(supports instrument3 image0)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 star1)
	(supports instrument4 image2)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument5 image0)
	(calibration_target instrument5 star1)
	(supports instrument6 image0)
	(supports instrument6 spectrograph3)
	(supports instrument6 image2)
	(calibration_target instrument6 star2)
	(supports instrument7 spectrograph3)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star2)
	(supports instrument8 image2)
	(supports instrument8 spectrograph3)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 star1)
	(supports instrument9 image2)
	(supports instrument9 image0)
	(calibration_target instrument9 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument10 image0)
	(calibration_target instrument10 star1)
	(supports instrument11 spectrograph3)
	(supports instrument11 image0)
	(supports instrument11 spectrograph1)
	(calibration_target instrument11 star2)
	(supports instrument12 image2)
	(calibration_target instrument12 star1)
	(supports instrument13 spectrograph1)
	(supports instrument13 spectrograph3)
	(calibration_target instrument13 star2)
	(supports instrument14 spectrograph1)
	(supports instrument14 image2)
	(supports instrument14 spectrograph3)
	(calibration_target instrument14 star0)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(pointing satellite0 planet4)
	(have_image star3 spectrograph1)
	(have_image planet4 spectrograph3)
))

)
