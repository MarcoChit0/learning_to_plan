(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
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
	instrument11 - instrument
	satellite5 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	image1 - mode
	spectrograph2 - mode
	spectrograph0 - mode
	groundstation0 - direction
	groundstation3 - direction
	groundstation1 - direction
	groundstation2 - direction
	star4 - direction
	star5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	star9 - direction
	phenomenon10 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 image1)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 spectrograph2)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument5 image1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 spectrograph2)
	(supports instrument6 image1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 spectrograph0)
	(supports instrument7 image1)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 groundstation3)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument8 spectrograph0)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 spectrograph0)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument10 spectrograph0)
	(supports instrument10 spectrograph2)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 spectrograph0)
	(supports instrument11 spectrograph2)
	(calibration_target instrument11 groundstation0)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon8)
	(supports instrument12 image1)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 spectrograph2)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 spectrograph2)
	(supports instrument14 spectrograph0)
	(supports instrument14 image1)
	(calibration_target instrument14 groundstation3)
	(supports instrument15 spectrograph2)
	(supports instrument15 image1)
	(calibration_target instrument15 groundstation1)
	(supports instrument16 spectrograph2)
	(calibration_target instrument16 groundstation2)
	(on_board instrument12 satellite5)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star9)
)
(:goal (and
	(pointing satellite5 phenomenon10)
	(have_image star4 spectrograph2)
	(have_image star5 image1)
	(have_image phenomenon6 image1)
	(have_image phenomenon7 spectrograph0)
	(have_image phenomenon8 spectrograph0)
	(have_image star9 image1)
	(have_image phenomenon10 image1)
))

)
