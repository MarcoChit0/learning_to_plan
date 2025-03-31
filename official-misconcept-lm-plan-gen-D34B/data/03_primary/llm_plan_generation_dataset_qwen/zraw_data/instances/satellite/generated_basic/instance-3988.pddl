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
	instrument8 - instrument
	instrument9 - instrument
	satellite2 - satellite
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	spectrograph0 - mode
	image1 - mode
	star2 - direction
	groundstation1 - direction
	star0 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star0)
	(supports instrument2 image1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon3)
	(supports instrument4 image1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star2)
	(supports instrument5 image1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star2)
	(supports instrument6 spectrograph0)
	(supports instrument6 image1)
	(calibration_target instrument6 star2)
	(supports instrument7 image1)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star2)
	(supports instrument8 image1)
	(calibration_target instrument8 star2)
	(supports instrument9 image1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument10 image1)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star0)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument11 image1)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 image1)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 spectrograph0)
	(supports instrument13 image1)
	(calibration_target instrument13 star0)
	(supports instrument14 image1)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 image1)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 star0)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(have_image phenomenon3 image1)
))

)
