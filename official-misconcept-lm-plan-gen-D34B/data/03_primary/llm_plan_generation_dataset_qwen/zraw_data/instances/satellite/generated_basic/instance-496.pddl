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
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	spectrograph2 - mode
	spectrograph0 - mode
	spectrograph1 - mode
	image3 - mode
	star4 - direction
	groundstation2 - direction
	groundstation0 - direction
	star1 - direction
	groundstation3 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star4)
	(supports instrument1 spectrograph2)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star4)
	(supports instrument2 image3)
	(supports instrument2 spectrograph2)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument3 image3)
	(calibration_target instrument3 star1)
	(supports instrument4 image3)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star4)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star4)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon7)
	(supports instrument8 image3)
	(supports instrument8 spectrograph1)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation3)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument10 spectrograph1)
	(supports instrument10 image3)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 image3)
	(supports instrument12 spectrograph2)
	(calibration_target instrument12 star1)
	(supports instrument13 spectrograph1)
	(calibration_target instrument13 groundstation3)
	(supports instrument14 image3)
	(supports instrument14 spectrograph2)
	(calibration_target instrument14 groundstation3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite0 phenomenon6)
	(pointing satellite1 star1)
	(pointing satellite2 groundstation0)
	(have_image phenomenon5 spectrograph0)
	(have_image phenomenon6 spectrograph1)
	(have_image phenomenon7 image3)
))

)
