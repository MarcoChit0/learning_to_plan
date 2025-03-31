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
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	thermograph3 - mode
	infrared1 - mode
	spectrograph2 - mode
	spectrograph0 - mode
	groundstation4 - direction
	star2 - direction
	star3 - direction
	groundstation1 - direction
	star0 - direction
	planet5 - direction
	star6 - direction
	planet7 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star3)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument4 infrared1)
	(supports instrument4 thermograph3)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star3)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star3)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star2)
	(supports instrument7 thermograph3)
	(supports instrument7 spectrograph0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star3)
	(supports instrument8 spectrograph0)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 star3)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument9 thermograph3)
	(supports instrument9 infrared1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star3)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet7)
	(supports instrument10 thermograph3)
	(supports instrument10 infrared1)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 spectrograph0)
	(supports instrument11 thermograph3)
	(calibration_target instrument11 star0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
)
(:goal (and
	(pointing satellite1 star6)
	(pointing satellite2 groundstation4)
	(pointing satellite3 star0)
	(have_image planet5 infrared1)
	(have_image star6 spectrograph2)
	(have_image planet7 thermograph3)
))

)
