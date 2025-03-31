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
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	spectrograph0 - mode
	thermograph2 - mode
	infrared1 - mode
	groundstation5 - direction
	star0 - direction
	star3 - direction
	groundstation2 - direction
	star1 - direction
	star4 - direction
	star6 - direction
	phenomenon7 - direction
	planet8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star1)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star0)
	(calibration_target instrument1 star3)
	(supports instrument2 spectrograph0)
	(supports instrument2 infrared1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 thermograph2)
	(supports instrument3 infrared1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star3)
	(calibration_target instrument3 star4)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star3)
	(supports instrument5 thermograph2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star0)
	(calibration_target instrument5 star3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star1)
	(supports instrument7 infrared1)
	(supports instrument7 thermograph2)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star4)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation5)
	(supports instrument8 spectrograph0)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 star3)
	(supports instrument10 infrared1)
	(supports instrument10 thermograph2)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star1)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 infrared1)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 star4)
	(calibration_target instrument11 star1)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite0 star3)
	(pointing satellite1 planet9)
	(pointing satellite3 star1)
	(have_image star6 infrared1)
	(have_image phenomenon7 thermograph2)
	(have_image planet8 spectrograph0)
	(have_image planet9 infrared1)
))

)
