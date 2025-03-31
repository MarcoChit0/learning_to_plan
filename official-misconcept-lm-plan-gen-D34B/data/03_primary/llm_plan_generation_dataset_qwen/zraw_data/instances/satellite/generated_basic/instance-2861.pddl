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
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	thermograph0 - mode
	spectrograph1 - mode
	groundstation0 - direction
	star4 - direction
	star3 - direction
	groundstation2 - direction
	star1 - direction
	planet5 - direction
	star6 - direction
	star7 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star3)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star4)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star3)
	(supports instrument3 thermograph0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star1)
	(supports instrument5 thermograph0)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 thermograph0)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 star1)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument8 spectrograph1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star3)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star3)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 star3)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument11 spectrograph1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 spectrograph1)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star1)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
)
(:goal (and
	(pointing satellite1 star6)
	(have_image planet5 thermograph0)
	(have_image star6 thermograph0)
	(have_image star7 thermograph0)
))

)
