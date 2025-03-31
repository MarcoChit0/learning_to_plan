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
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	spectrograph3 - mode
	image2 - mode
	thermograph1 - mode
	thermograph0 - mode
	groundstation3 - direction
	groundstation1 - direction
	groundstation0 - direction
	star2 - direction
	star4 - direction
	planet5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star2)
	(supports instrument2 spectrograph3)
	(supports instrument2 thermograph1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument4 thermograph0)
	(supports instrument4 image2)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 star2)
	(supports instrument5 thermograph1)
	(supports instrument5 image2)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 image2)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 star4)
	(supports instrument7 thermograph0)
	(supports instrument7 spectrograph3)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument8 image2)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation1)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
	(supports instrument10 spectrograph3)
	(supports instrument10 image2)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 thermograph1)
	(supports instrument11 spectrograph3)
	(calibration_target instrument11 star4)
	(supports instrument12 spectrograph3)
	(supports instrument12 image2)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star2)
	(supports instrument13 image2)
	(supports instrument13 thermograph0)
	(supports instrument13 spectrograph3)
	(calibration_target instrument13 star4)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
)
(:goal (and
	(have_image planet5 image2)
	(have_image phenomenon6 thermograph0)
))

)
