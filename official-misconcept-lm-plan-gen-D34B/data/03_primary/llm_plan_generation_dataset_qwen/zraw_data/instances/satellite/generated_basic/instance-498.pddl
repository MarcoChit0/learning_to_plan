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
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	thermograph1 - mode
	infrared2 - mode
	spectrograph3 - mode
	image0 - mode
	star4 - direction
	groundstation0 - direction
	star2 - direction
	star3 - direction
	star1 - direction
	star5 - direction
	planet6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star3)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 star2)
	(supports instrument3 infrared2)
	(supports instrument3 thermograph1)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 star1)
	(supports instrument4 image0)
	(supports instrument4 infrared2)
	(calibration_target instrument4 star3)
	(supports instrument5 thermograph1)
	(supports instrument5 infrared2)
	(calibration_target instrument5 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument6 spectrograph3)
	(supports instrument6 image0)
	(calibration_target instrument6 star3)
	(supports instrument7 infrared2)
	(supports instrument7 spectrograph3)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation0)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument8 thermograph1)
	(supports instrument8 infrared2)
	(calibration_target instrument8 star3)
	(supports instrument9 spectrograph3)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument10 image0)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 star2)
	(supports instrument11 spectrograph3)
	(supports instrument11 image0)
	(calibration_target instrument11 star2)
	(supports instrument12 spectrograph3)
	(calibration_target instrument12 star3)
	(supports instrument13 infrared2)
	(supports instrument13 thermograph1)
	(calibration_target instrument13 star1)
	(supports instrument14 spectrograph3)
	(supports instrument14 infrared2)
	(supports instrument14 image0)
	(calibration_target instrument14 star3)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 star1)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite0 star2)
	(pointing satellite2 phenomenon7)
	(have_image star5 spectrograph3)
	(have_image planet6 infrared2)
	(have_image phenomenon7 image0)
))

)
