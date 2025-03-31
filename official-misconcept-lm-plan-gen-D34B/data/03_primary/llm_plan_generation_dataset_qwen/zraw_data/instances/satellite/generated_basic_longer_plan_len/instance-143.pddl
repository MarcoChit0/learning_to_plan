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
	instrument6 - instrument
	satellite1 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	infrared1 - mode
	infrared2 - mode
	thermograph3 - mode
	spectrograph0 - mode
	groundstation2 - direction
	star0 - direction
	groundstation1 - direction
	groundstation3 - direction
	groundstation4 - direction
	planet5 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	planet9 - direction
	planet10 - direction
	star11 - direction
	star12 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 thermograph3)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared2)
	(supports instrument1 spectrograph0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 infrared2)
	(supports instrument2 infrared1)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 infrared1)
	(supports instrument3 thermograph3)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 thermograph3)
	(supports instrument4 infrared2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 spectrograph0)
	(supports instrument5 infrared2)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(on_board instrument6 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon7)
	(supports instrument7 spectrograph0)
	(supports instrument7 thermograph3)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 infrared2)
	(calibration_target instrument9 groundstation4)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet10)
)
(:goal (and
	(pointing satellite1 phenomenon7)
	(have_image planet5 thermograph3)
	(have_image star6 infrared1)
	(have_image phenomenon7 infrared2)
	(have_image phenomenon8 thermograph3)
	(have_image planet9 thermograph3)
	(have_image planet10 infrared2)
	(have_image star11 thermograph3)
	(have_image star12 spectrograph0)
))

)
