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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	satellite5 - satellite
	instrument18 - instrument
	spectrograph3 - mode
	infrared1 - mode
	spectrograph2 - mode
	image0 - mode
	groundstation1 - direction
	star0 - direction
	star3 - direction
	star2 - direction
	phenomenon4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(calibration_target instrument1 star2)
	(supports instrument2 image0)
	(supports instrument2 spectrograph2)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 star2)
	(supports instrument3 infrared1)
	(supports instrument3 image0)
	(calibration_target instrument3 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument4 spectrograph3)
	(supports instrument4 infrared1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image0)
	(supports instrument5 infrared1)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 star0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon5)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star3)
	(supports instrument7 spectrograph2)
	(supports instrument7 image0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 spectrograph3)
	(calibration_target instrument8 star0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument9 image0)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 star3)
	(supports instrument10 spectrograph2)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star0)
	(supports instrument11 infrared1)
	(calibration_target instrument11 star2)
	(supports instrument12 spectrograph3)
	(supports instrument12 image0)
	(calibration_target instrument12 star0)
	(supports instrument13 spectrograph2)
	(calibration_target instrument13 star0)
	(supports instrument14 spectrograph2)
	(supports instrument14 spectrograph3)
	(calibration_target instrument14 star3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument15 infrared1)
	(supports instrument15 spectrograph3)
	(supports instrument15 image0)
	(calibration_target instrument15 star2)
	(supports instrument16 spectrograph3)
	(calibration_target instrument16 star2)
	(supports instrument17 spectrograph2)
	(calibration_target instrument17 star3)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon5)
	(supports instrument18 spectrograph2)
	(supports instrument18 image0)
	(supports instrument18 infrared1)
	(calibration_target instrument18 star2)
	(on_board instrument18 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation1)
)
(:goal (and
	(pointing satellite1 groundstation1)
	(pointing satellite4 groundstation1)
	(have_image phenomenon4 infrared1)
	(have_image phenomenon5 image0)
))

)
