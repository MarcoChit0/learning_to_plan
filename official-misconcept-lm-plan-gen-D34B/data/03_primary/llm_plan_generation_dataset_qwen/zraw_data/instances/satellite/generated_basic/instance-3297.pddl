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
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	thermograph0 - mode
	spectrograph1 - mode
	image2 - mode
	star1 - direction
	groundstation2 - direction
	groundstation0 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 image2)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 image2)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star1)
	(supports instrument4 image2)
	(supports instrument4 spectrograph1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 spectrograph1)
	(supports instrument5 image2)
	(calibration_target instrument5 star1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument8 spectrograph1)
	(supports instrument8 image2)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 spectrograph1)
	(supports instrument9 image2)
	(calibration_target instrument9 groundstation0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon3)
	(supports instrument10 spectrograph1)
	(supports instrument10 thermograph0)
	(supports instrument10 image2)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 image2)
	(supports instrument11 spectrograph1)
	(calibration_target instrument11 groundstation0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(have_image phenomenon3 image2)
))

)
