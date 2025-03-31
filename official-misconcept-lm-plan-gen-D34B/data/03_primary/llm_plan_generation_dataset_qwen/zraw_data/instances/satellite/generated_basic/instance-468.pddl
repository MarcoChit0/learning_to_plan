(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	infrared3 - mode
	thermograph2 - mode
	image1 - mode
	spectrograph0 - mode
	star2 - direction
	star1 - direction
	groundstation0 - direction
	star4 - direction
	groundstation3 - direction
	star5 - direction
	planet6 - direction
	star7 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared3)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 infrared3)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph0)
	(supports instrument2 infrared3)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument3 image1)
	(supports instrument3 infrared3)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 image1)
	(supports instrument4 thermograph2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument6 infrared3)
	(supports instrument6 thermograph2)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star1)
	(supports instrument7 image1)
	(supports instrument7 infrared3)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 star2)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 image1)
	(supports instrument9 infrared3)
	(calibration_target instrument9 star1)
	(supports instrument10 image1)
	(calibration_target instrument10 star1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument11 thermograph2)
	(supports instrument11 image1)
	(supports instrument11 infrared3)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 thermograph2)
	(supports instrument12 infrared3)
	(calibration_target instrument12 star4)
	(supports instrument13 thermograph2)
	(calibration_target instrument13 groundstation3)
	(supports instrument14 spectrograph0)
	(supports instrument14 image1)
	(supports instrument14 infrared3)
	(calibration_target instrument14 groundstation3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
)
(:goal (and
	(pointing satellite0 star5)
	(pointing satellite1 planet6)
	(have_image star5 thermograph2)
	(have_image planet6 thermograph2)
	(have_image star7 spectrograph0)
))

)
