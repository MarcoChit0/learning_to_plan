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
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	infrared1 - mode
	thermograph0 - mode
	image2 - mode
	spectrograph3 - mode
	groundstation1 - direction
	groundstation2 - direction
	star3 - direction
	star0 - direction
	star4 - direction
	planet5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image2)
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 star4)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 infrared1)
	(supports instrument2 thermograph0)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 star4)
	(supports instrument3 image2)
	(supports instrument3 thermograph0)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 star3)
	(supports instrument4 thermograph0)
	(supports instrument4 spectrograph3)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 infrared1)
	(supports instrument5 spectrograph3)
	(supports instrument5 image2)
	(calibration_target instrument5 star4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument6 thermograph0)
	(supports instrument6 spectrograph3)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star4)
	(supports instrument7 infrared1)
	(supports instrument7 spectrograph3)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 thermograph0)
	(supports instrument8 spectrograph3)
	(calibration_target instrument8 star3)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument9 thermograph0)
	(supports instrument9 spectrograph3)
	(supports instrument9 image2)
	(calibration_target instrument9 star0)
	(supports instrument10 image2)
	(calibration_target instrument10 star4)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
)
(:goal (and
	(pointing satellite1 planet6)
	(pointing satellite2 star3)
	(pointing satellite3 star4)
	(have_image planet5 infrared1)
	(have_image planet6 infrared1)
))

)
