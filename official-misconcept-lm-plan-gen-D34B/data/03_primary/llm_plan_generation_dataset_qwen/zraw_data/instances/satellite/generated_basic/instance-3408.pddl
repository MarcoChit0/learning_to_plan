(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	thermograph2 - mode
	spectrograph0 - mode
	infrared1 - mode
	groundstation0 - direction
	star1 - direction
	star2 - direction
	groundstation3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph2)
	(supports instrument4 spectrograph0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star2)
	(supports instrument6 infrared1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument7 spectrograph0)
	(supports instrument7 thermograph2)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 infrared1)
	(supports instrument8 spectrograph0)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 groundstation3)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 thermograph2)
	(calibration_target instrument10 star2)
	(supports instrument11 infrared1)
	(calibration_target instrument11 groundstation3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet4)
)
(:goal (and
	(pointing satellite1 planet4)
	(pointing satellite3 star1)
	(have_image planet4 thermograph2)
))

)
