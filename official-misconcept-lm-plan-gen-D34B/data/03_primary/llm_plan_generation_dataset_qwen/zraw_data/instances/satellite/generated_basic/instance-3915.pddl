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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	image0 - mode
	spectrograph2 - mode
	image3 - mode
	spectrograph1 - mode
	star0 - direction
	star2 - direction
	star4 - direction
	groundstation3 - direction
	groundstation1 - direction
	groundstation5 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 image0)
	(supports instrument0 image3)
	(calibration_target instrument0 star2)
	(supports instrument1 spectrograph1)
	(supports instrument1 spectrograph2)
	(supports instrument1 image3)
	(calibration_target instrument1 star0)
	(calibration_target instrument1 star4)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image3)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation1)
	(calibration_target instrument6 star2)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet8)
	(supports instrument7 image3)
	(calibration_target instrument7 groundstation1)
	(calibration_target instrument7 star2)
	(supports instrument8 image0)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 groundstation1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star7)
	(supports instrument9 spectrograph2)
	(supports instrument9 image0)
	(calibration_target instrument9 star2)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument10 spectrograph1)
	(supports instrument10 image3)
	(calibration_target instrument10 star4)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 image3)
	(supports instrument11 spectrograph1)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation1)
	(calibration_target instrument11 groundstation3)
	(supports instrument12 image3)
	(supports instrument12 spectrograph1)
	(calibration_target instrument12 groundstation5)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star7)
)
(:goal (and
	(have_image planet6 spectrograph1)
	(have_image star7 spectrograph2)
	(have_image planet8 image0)
))

)
