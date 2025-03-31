(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	thermograph0 - mode
	infrared1 - mode
	groundstation5 - direction
	star4 - direction
	star0 - direction
	star3 - direction
	star1 - direction
	groundstation2 - direction
	phenomenon6 - direction
	planet7 - direction
	planet8 - direction
	star9 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star4)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet8)
	(supports instrument1 thermograph0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star1)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 thermograph0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star3)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star0)
	(calibration_target instrument3 groundstation2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
	(supports instrument4 thermograph0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star4)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star4)
	(supports instrument6 thermograph0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star1)
	(supports instrument7 infrared1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star0)
	(supports instrument8 infrared1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star1)
	(calibration_target instrument8 star3)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument9 infrared1)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star1)
	(supports instrument10 thermograph0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star1)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
	(supports instrument11 infrared1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation2)
	(calibration_target instrument11 star1)
	(on_board instrument11 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet7)
)
(:goal (and
	(pointing satellite4 planet8)
	(have_image phenomenon6 thermograph0)
	(have_image planet7 infrared1)
	(have_image planet8 infrared1)
	(have_image star9 thermograph0)
))

)
