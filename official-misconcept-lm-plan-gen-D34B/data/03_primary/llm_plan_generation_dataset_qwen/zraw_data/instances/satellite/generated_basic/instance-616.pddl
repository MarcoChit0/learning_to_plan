(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	spectrograph0 - mode
	thermograph1 - mode
	image2 - mode
	groundstation1 - direction
	groundstation3 - direction
	star2 - direction
	star0 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument1 thermograph1)
	(supports instrument1 spectrograph0)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 thermograph1)
	(supports instrument3 spectrograph0)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument4 image2)
	(supports instrument4 spectrograph0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star0)
	(supports instrument6 image2)
	(supports instrument6 spectrograph0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star2)
	(supports instrument7 thermograph1)
	(supports instrument7 spectrograph0)
	(supports instrument7 image2)
	(calibration_target instrument7 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 image2)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 thermograph1)
	(supports instrument10 spectrograph0)
	(supports instrument10 image2)
	(calibration_target instrument10 star0)
	(supports instrument11 thermograph1)
	(supports instrument11 image2)
	(calibration_target instrument11 groundstation3)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 groundstation3)
	(supports instrument13 image2)
	(calibration_target instrument13 star2)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon5)
	(supports instrument14 image2)
	(supports instrument14 thermograph1)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 star0)
	(supports instrument15 image2)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 star2)
	(supports instrument16 image2)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 star0)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon6)
)
(:goal (and
	(pointing satellite2 groundstation1)
	(pointing satellite4 phenomenon4)
	(have_image phenomenon4 spectrograph0)
	(have_image phenomenon5 image2)
	(have_image phenomenon6 image2)
))

)
