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
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite4 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite5 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	spectrograph0 - mode
	thermograph1 - mode
	image2 - mode
	groundstation3 - direction
	groundstation2 - direction
	groundstation1 - direction
	star0 - direction
	star4 - direction
	planet5 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
	phenomenon10 - direction
)
(:init
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 image2)
	(supports instrument1 spectrograph0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon9)
	(supports instrument2 spectrograph0)
	(supports instrument2 image2)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star0)
	(supports instrument3 image2)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 image2)
	(calibration_target instrument4 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument5 spectrograph0)
	(supports instrument5 thermograph1)
	(supports instrument5 image2)
	(calibration_target instrument5 groundstation2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument6 image2)
	(supports instrument6 spectrograph0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star0)
	(supports instrument7 spectrograph0)
	(supports instrument7 thermograph1)
	(supports instrument7 image2)
	(calibration_target instrument7 groundstation3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument8 thermograph1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star0)
	(supports instrument9 thermograph1)
	(supports instrument9 image2)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 image2)
	(supports instrument11 spectrograph0)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 groundstation3)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 image2)
	(calibration_target instrument13 groundstation3)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon9)
	(supports instrument14 image2)
	(supports instrument14 spectrograph0)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 groundstation3)
	(supports instrument15 spectrograph0)
	(supports instrument15 thermograph1)
	(supports instrument15 image2)
	(calibration_target instrument15 groundstation2)
	(supports instrument16 thermograph1)
	(supports instrument16 image2)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 groundstation1)
	(supports instrument17 spectrograph0)
	(supports instrument17 image2)
	(supports instrument17 thermograph1)
	(calibration_target instrument17 star0)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet5)
)
(:goal (and
	(pointing satellite1 phenomenon10)
	(pointing satellite2 phenomenon10)
	(pointing satellite3 phenomenon8)
	(pointing satellite5 phenomenon10)
	(have_image star4 spectrograph0)
	(have_image planet5 thermograph1)
	(have_image star6 image2)
	(have_image phenomenon7 thermograph1)
	(have_image phenomenon8 image2)
	(have_image phenomenon9 image2)
	(have_image phenomenon10 spectrograph0)
))

)
