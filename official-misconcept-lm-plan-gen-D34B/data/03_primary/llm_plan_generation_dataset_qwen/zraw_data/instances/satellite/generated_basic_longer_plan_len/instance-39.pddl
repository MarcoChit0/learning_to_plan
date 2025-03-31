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
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	spectrograph2 - mode
	spectrograph0 - mode
	image3 - mode
	image1 - mode
	star2 - direction
	star0 - direction
	groundstation3 - direction
	groundstation1 - direction
	star4 - direction
	planet5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image3)
	(calibration_target instrument2 star2)
	(supports instrument3 spectrograph2)
	(supports instrument3 image3)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon7)
	(supports instrument4 spectrograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image1)
	(calibration_target instrument6 star2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument7 image3)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 image3)
	(supports instrument8 spectrograph2)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star0)
	(supports instrument9 image1)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 star0)
	(supports instrument10 spectrograph0)
	(supports instrument10 image3)
	(supports instrument10 image1)
	(calibration_target instrument10 star0)
	(supports instrument11 image1)
	(supports instrument11 spectrograph2)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 star0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument12 image3)
	(calibration_target instrument12 groundstation3)
	(supports instrument13 spectrograph0)
	(supports instrument13 image3)
	(supports instrument13 image1)
	(calibration_target instrument13 groundstation3)
	(supports instrument14 image1)
	(supports instrument14 spectrograph2)
	(calibration_target instrument14 groundstation1)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
)
(:goal (and
	(have_image star4 image1)
	(have_image planet5 image3)
	(have_image phenomenon6 spectrograph2)
	(have_image phenomenon7 spectrograph2)
	(have_image phenomenon8 spectrograph0)
	(have_image planet9 image1)
))

)
