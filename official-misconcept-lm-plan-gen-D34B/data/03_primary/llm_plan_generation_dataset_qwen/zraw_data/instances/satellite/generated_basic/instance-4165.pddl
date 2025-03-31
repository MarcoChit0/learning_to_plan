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
	instrument10 - instrument
	satellite2 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	satellite3 - satellite
	instrument17 - instrument
	instrument18 - instrument
	thermograph0 - mode
	image3 - mode
	infrared1 - mode
	infrared2 - mode
	groundstation0 - direction
	groundstation1 - direction
	planet2 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 thermograph0)
	(supports instrument0 image3)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image3)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared1)
	(supports instrument3 image3)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared2)
	(supports instrument4 image3)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 thermograph0)
	(supports instrument5 infrared1)
	(supports instrument5 image3)
	(calibration_target instrument5 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument6 image3)
	(supports instrument6 infrared2)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared1)
	(supports instrument7 image3)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 thermograph0)
	(supports instrument8 image3)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 infrared2)
	(supports instrument9 image3)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 infrared2)
	(supports instrument10 image3)
	(supports instrument10 infrared1)
	(calibration_target instrument10 groundstation1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet2)
	(supports instrument11 infrared1)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 image3)
	(supports instrument12 infrared1)
	(supports instrument12 infrared2)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 image3)
	(supports instrument13 infrared1)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 infrared2)
	(supports instrument14 image3)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 thermograph0)
	(supports instrument15 infrared2)
	(calibration_target instrument15 groundstation1)
	(supports instrument16 infrared2)
	(supports instrument16 image3)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 groundstation1)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(on_board instrument15 satellite2)
	(on_board instrument16 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument17 infrared1)
	(calibration_target instrument17 groundstation1)
	(supports instrument18 infrared1)
	(calibration_target instrument18 groundstation1)
	(on_board instrument17 satellite3)
	(on_board instrument18 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite0 planet2)
	(pointing satellite2 planet2)
	(have_image planet2 infrared1)
))

)
