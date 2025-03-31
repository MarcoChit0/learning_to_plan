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
	spectrograph1 - mode
	thermograph0 - mode
	infrared2 - mode
	star2 - direction
	star4 - direction
	star0 - direction
	groundstation5 - direction
	star3 - direction
	groundstation1 - direction
	star6 - direction
	planet7 - direction
	planet8 - direction
	planet9 - direction
	star10 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 spectrograph1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation5)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 infrared2)
	(supports instrument3 spectrograph1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star4)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet7)
	(supports instrument4 spectrograph1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation5)
	(calibration_target instrument4 star0)
	(supports instrument5 spectrograph1)
	(supports instrument5 infrared2)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation5)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 star3)
	(supports instrument7 spectrograph1)
	(supports instrument7 infrared2)
	(calibration_target instrument7 groundstation1)
	(calibration_target instrument7 star3)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
)
(:goal (and
	(have_image star6 spectrograph1)
	(have_image planet7 infrared2)
	(have_image planet8 infrared2)
	(have_image planet9 infrared2)
	(have_image star10 spectrograph1)
))

)
