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
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	spectrograph2 - mode
	spectrograph1 - mode
	infrared3 - mode
	spectrograph0 - mode
	star2 - direction
	star1 - direction
	groundstation0 - direction
	groundstation3 - direction
	star4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared3)
	(calibration_target instrument0 star2)
	(supports instrument1 spectrograph0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star2)
	(supports instrument2 spectrograph2)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument4 spectrograph1)
	(supports instrument4 infrared3)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star1)
	(supports instrument6 infrared3)
	(calibration_target instrument6 star1)
	(supports instrument7 infrared3)
	(supports instrument7 spectrograph1)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 star1)
	(supports instrument10 spectrograph0)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 groundstation3)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
	(supports instrument11 spectrograph1)
	(supports instrument11 spectrograph0)
	(supports instrument11 infrared3)
	(calibration_target instrument11 star2)
	(supports instrument12 infrared3)
	(supports instrument12 spectrograph2)
	(calibration_target instrument12 star1)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon6)
	(supports instrument13 spectrograph2)
	(supports instrument13 spectrograph1)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 groundstation3)
	(supports instrument14 spectrograph1)
	(supports instrument14 spectrograph2)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 infrared3)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 groundstation3)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
)
(:goal (and
	(pointing satellite2 star2)
	(pointing satellite3 star4)
	(have_image star4 spectrograph2)
	(have_image phenomenon5 spectrograph1)
	(have_image phenomenon6 spectrograph0)
))

)
