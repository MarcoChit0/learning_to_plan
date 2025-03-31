(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	image2 - mode
	thermograph3 - mode
	image0 - mode
	spectrograph1 - mode
	star3 - direction
	groundstation1 - direction
	groundstation0 - direction
	star4 - direction
	star2 - direction
	planet5 - direction
	phenomenon6 - direction
	planet7 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 image2)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star2)
	(supports instrument1 spectrograph1)
	(supports instrument1 image2)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 star3)
	(supports instrument2 image2)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph3)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 spectrograph1)
	(supports instrument4 image0)
	(supports instrument4 thermograph3)
	(calibration_target instrument4 star3)
	(supports instrument5 image2)
	(supports instrument5 image0)
	(supports instrument5 thermograph3)
	(calibration_target instrument5 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument6 spectrograph1)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 image2)
	(supports instrument7 thermograph3)
	(calibration_target instrument7 star4)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument8 thermograph3)
	(supports instrument8 image2)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 star4)
	(supports instrument9 image2)
	(supports instrument9 image0)
	(calibration_target instrument9 star4)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet7)
	(supports instrument10 thermograph3)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 star2)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
)
(:goal (and
	(pointing satellite2 planet5)
	(have_image planet5 spectrograph1)
	(have_image phenomenon6 image0)
	(have_image planet7 image2)
))

)
