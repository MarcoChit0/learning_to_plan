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
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	image2 - mode
	thermograph1 - mode
	thermograph3 - mode
	spectrograph0 - mode
	groundstation0 - direction
	star1 - direction
	star2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 spectrograph0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 thermograph3)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 thermograph1)
	(supports instrument5 image2)
	(calibration_target instrument5 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument6 thermograph3)
	(supports instrument6 spectrograph0)
	(supports instrument6 image2)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 spectrograph0)
	(supports instrument7 image2)
	(supports instrument7 thermograph3)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 thermograph3)
	(supports instrument9 image2)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 groundstation0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 spectrograph0)
	(supports instrument11 image2)
	(supports instrument11 thermograph3)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 groundstation0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
)
(:goal (and
	(have_image star1 thermograph1)
	(have_image star2 thermograph1)
	(have_image phenomenon3 thermograph1)
))

)
