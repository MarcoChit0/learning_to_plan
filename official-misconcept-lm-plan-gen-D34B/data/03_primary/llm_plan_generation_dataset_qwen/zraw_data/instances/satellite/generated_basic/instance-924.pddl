(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	satellite5 - satellite
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	spectrograph1 - mode
	image0 - mode
	groundstation3 - direction
	groundstation1 - direction
	groundstation0 - direction
	star2 - direction
	planet4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star2)
	(supports instrument1 image0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star2)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon5)
	(supports instrument3 spectrograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image0)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 image0)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 image0)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 image0)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 groundstation0)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 image0)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 groundstation0)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument11 image0)
	(supports instrument11 spectrograph1)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 spectrograph1)
	(supports instrument12 image0)
	(calibration_target instrument12 star2)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 image0)
	(supports instrument15 spectrograph1)
	(calibration_target instrument15 groundstation1)
	(supports instrument16 spectrograph1)
	(supports instrument16 image0)
	(calibration_target instrument16 groundstation1)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
	(supports instrument17 spectrograph1)
	(supports instrument17 image0)
	(calibration_target instrument17 star2)
	(supports instrument18 spectrograph1)
	(calibration_target instrument18 star2)
	(supports instrument19 spectrograph1)
	(supports instrument19 image0)
	(calibration_target instrument19 groundstation0)
	(supports instrument20 image0)
	(calibration_target instrument20 groundstation0)
	(supports instrument21 image0)
	(supports instrument21 spectrograph1)
	(calibration_target instrument21 star2)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation0)
)
(:goal (and
	(pointing satellite1 star2)
	(pointing satellite2 phenomenon5)
	(pointing satellite4 groundstation0)
	(have_image planet4 image0)
	(have_image phenomenon5 image0)
))

)
