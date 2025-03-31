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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	instrument12 - instrument
	thermograph0 - mode
	spectrograph1 - mode
	thermograph3 - mode
	infrared2 - mode
	groundstation1 - direction
	star3 - direction
	groundstation4 - direction
	star0 - direction
	star2 - direction
	planet5 - direction
	planet6 - direction
	planet7 - direction
	planet8 - direction
	planet9 - direction
	phenomenon10 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 infrared2)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star2)
	(supports instrument1 spectrograph1)
	(supports instrument1 infrared2)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star3)
	(supports instrument2 spectrograph1)
	(supports instrument2 thermograph3)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star2)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet8)
	(supports instrument4 thermograph3)
	(supports instrument4 infrared2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star2)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument5 spectrograph1)
	(supports instrument5 thermograph3)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(supports instrument6 spectrograph1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star2)
	(supports instrument7 infrared2)
	(supports instrument7 thermograph3)
	(calibration_target instrument7 star0)
	(supports instrument8 infrared2)
	(supports instrument8 thermograph3)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 star0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument10 infrared2)
	(supports instrument10 spectrograph1)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation4)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
	(supports instrument11 thermograph0)
	(supports instrument11 infrared2)
	(calibration_target instrument11 star0)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star2)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
)
(:goal (and
	(pointing satellite1 phenomenon10)
	(have_image planet5 thermograph0)
	(have_image planet6 infrared2)
	(have_image planet7 thermograph3)
	(have_image planet8 spectrograph1)
	(have_image planet9 thermograph3)
	(have_image phenomenon10 spectrograph1)
))

)
