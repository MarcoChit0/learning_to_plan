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
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	infrared1 - mode
	thermograph0 - mode
	thermograph3 - mode
	infrared2 - mode
	groundstation1 - direction
	groundstation3 - direction
	star0 - direction
	groundstation2 - direction
	planet4 - direction
	planet5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 thermograph0)
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 thermograph3)
	(supports instrument2 infrared2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 thermograph0)
	(supports instrument3 infrared2)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 infrared2)
	(supports instrument5 thermograph0)
	(supports instrument5 thermograph3)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 thermograph0)
	(supports instrument6 thermograph3)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star0)
	(supports instrument7 thermograph3)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph0)
	(supports instrument8 thermograph3)
	(supports instrument8 infrared2)
	(calibration_target instrument8 star0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument9 infrared2)
	(supports instrument9 thermograph0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
)
(:goal (and
	(pointing satellite1 groundstation3)
	(have_image planet4 infrared2)
	(have_image planet5 thermograph0)
	(have_image phenomenon6 thermograph0)
))

)
