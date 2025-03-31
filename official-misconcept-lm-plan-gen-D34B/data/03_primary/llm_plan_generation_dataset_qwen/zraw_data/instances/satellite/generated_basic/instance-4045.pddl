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
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	image2 - mode
	thermograph0 - mode
	infrared1 - mode
	star0 - direction
	groundstation1 - direction
	planet2 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image2)
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph0)
	(supports instrument1 infrared1)
	(supports instrument1 image2)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph0)
	(supports instrument2 infrared1)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument3 thermograph0)
	(supports instrument3 image2)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared1)
	(supports instrument4 thermograph0)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared1)
	(supports instrument7 image2)
	(calibration_target instrument7 star0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument9 infrared1)
	(supports instrument9 image2)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star0)
	(supports instrument10 image2)
	(supports instrument10 infrared1)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 image2)
	(supports instrument11 infrared1)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 groundstation1)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite0 star0)
	(have_image planet2 image2)
))

)
