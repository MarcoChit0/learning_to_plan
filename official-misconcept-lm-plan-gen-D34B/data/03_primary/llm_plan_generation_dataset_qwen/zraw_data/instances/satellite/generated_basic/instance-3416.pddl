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
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	spectrograph2 - mode
	infrared1 - mode
	thermograph0 - mode
	star2 - direction
	star1 - direction
	groundstation3 - direction
	star0 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 spectrograph2)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph0)
	(supports instrument1 infrared1)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star2)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star2)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star1)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 star1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 infrared1)
	(supports instrument9 thermograph0)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 star0)
	(supports instrument10 thermograph0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star0)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
)
(:goal (and
	(pointing satellite1 groundstation3)
	(pointing satellite3 groundstation3)
	(have_image phenomenon4 thermograph0)
))

)
