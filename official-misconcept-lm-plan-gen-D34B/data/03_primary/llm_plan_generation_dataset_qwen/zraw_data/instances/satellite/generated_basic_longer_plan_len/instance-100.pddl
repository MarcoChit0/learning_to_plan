(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	instrument10 - instrument
	satellite5 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	spectrograph1 - mode
	thermograph0 - mode
	image2 - mode
	groundstation3 - direction
	groundstation1 - direction
	star0 - direction
	star2 - direction
	planet4 - direction
	star5 - direction
	phenomenon6 - direction
	planet7 - direction
	phenomenon8 - direction
	planet9 - direction
	planet10 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 image2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 thermograph0)
	(supports instrument2 spectrograph1)
	(supports instrument2 image2)
	(calibration_target instrument2 star0)
	(supports instrument3 image2)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star2)
	(supports instrument4 spectrograph1)
	(supports instrument4 image2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet7)
	(supports instrument6 thermograph0)
	(supports instrument6 image2)
	(calibration_target instrument6 star2)
	(supports instrument7 image2)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 image2)
	(calibration_target instrument8 groundstation1)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 image2)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 groundstation3)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet4)
	(supports instrument11 spectrograph1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 spectrograph1)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star0)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 star2)
	(supports instrument14 spectrograph1)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 star2)
	(supports instrument15 spectrograph1)
	(calibration_target instrument15 star2)
	(on_board instrument11 satellite5)
	(on_board instrument12 satellite5)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation3)
)
(:goal (and
	(pointing satellite0 planet4)
	(pointing satellite2 star2)
	(pointing satellite5 planet7)
	(have_image planet4 thermograph0)
	(have_image star5 thermograph0)
	(have_image phenomenon6 spectrograph1)
	(have_image planet7 spectrograph1)
	(have_image phenomenon8 image2)
	(have_image planet9 thermograph0)
	(have_image planet10 thermograph0)
))

)
