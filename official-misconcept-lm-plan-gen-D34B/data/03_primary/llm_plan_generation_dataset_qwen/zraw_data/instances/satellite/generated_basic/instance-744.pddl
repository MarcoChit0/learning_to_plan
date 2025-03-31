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
	instrument8 - instrument
	instrument9 - instrument
	satellite2 - satellite
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	infrared0 - mode
	spectrograph1 - mode
	thermograph2 - mode
	groundstation4 - direction
	star2 - direction
	star5 - direction
	star1 - direction
	groundstation3 - direction
	star0 - direction
	planet6 - direction
	planet7 - direction
	star8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 spectrograph1)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star2)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star2)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph1)
	(supports instrument3 thermograph2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star5)
	(calibration_target instrument3 star0)
	(supports instrument4 spectrograph1)
	(supports instrument4 infrared0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph1)
	(supports instrument5 infrared0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star2)
	(supports instrument7 infrared0)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 spectrograph1)
	(supports instrument8 infrared0)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 star5)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 infrared0)
	(supports instrument9 thermograph2)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 star5)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet9)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 star5)
	(calibration_target instrument10 star1)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument11 spectrograph1)
	(supports instrument11 thermograph2)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation3)
	(calibration_target instrument11 star1)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 star0)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star8)
)
(:goal (and
	(pointing satellite0 groundstation4)
	(have_image planet6 thermograph2)
	(have_image planet7 spectrograph1)
	(have_image star8 thermograph2)
	(have_image planet9 infrared0)
))

)
