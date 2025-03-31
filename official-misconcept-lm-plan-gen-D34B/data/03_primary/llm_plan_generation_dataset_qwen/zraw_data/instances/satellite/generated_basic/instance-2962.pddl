(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	spectrograph3 - mode
	thermograph0 - mode
	image2 - mode
	image1 - mode
	star1 - direction
	groundstation3 - direction
	groundstation0 - direction
	groundstation2 - direction
	phenomenon4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star1)
	(supports instrument1 image2)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 groundstation0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph3)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star1)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 groundstation2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
	(supports instrument7 image2)
	(supports instrument7 spectrograph3)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image2)
	(supports instrument8 thermograph0)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 image2)
	(supports instrument9 thermograph0)
	(supports instrument9 spectrograph3)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 spectrograph3)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 image2)
	(calibration_target instrument11 groundstation3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument12 image2)
	(calibration_target instrument12 groundstation2)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 groundstation2)
	(supports instrument14 image2)
	(supports instrument14 spectrograph3)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 spectrograph3)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 spectrograph3)
	(supports instrument16 image1)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 groundstation2)
	(supports instrument17 spectrograph3)
	(supports instrument17 image1)
	(supports instrument17 thermograph0)
	(calibration_target instrument17 groundstation2)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation0)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite4 phenomenon5)
	(have_image phenomenon4 image2)
	(have_image phenomenon5 spectrograph3)
))

)
