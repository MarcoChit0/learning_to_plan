(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	thermograph0 - mode
	image1 - mode
	infrared2 - mode
	infrared3 - mode
	star3 - direction
	star0 - direction
	star1 - direction
	groundstation2 - direction
	groundstation4 - direction
	planet5 - direction
	planet6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	star9 - direction
	star10 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument1 infrared2)
	(supports instrument1 image1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star3)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 infrared3)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon8)
	(supports instrument6 infrared3)
	(supports instrument6 infrared2)
	(calibration_target instrument6 star1)
	(supports instrument7 infrared2)
	(supports instrument7 infrared3)
	(calibration_target instrument7 star3)
	(supports instrument8 infrared2)
	(supports instrument8 thermograph0)
	(supports instrument8 infrared3)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 infrared3)
	(supports instrument9 thermograph0)
	(supports instrument9 infrared2)
	(calibration_target instrument9 groundstation4)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 infrared2)
	(supports instrument11 infrared3)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star0)
	(supports instrument12 infrared2)
	(supports instrument12 image1)
	(calibration_target instrument12 star3)
	(supports instrument13 infrared2)
	(calibration_target instrument13 star0)
	(supports instrument14 infrared2)
	(supports instrument14 infrared3)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 star1)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation4)
	(supports instrument15 image1)
	(supports instrument15 infrared2)
	(calibration_target instrument15 groundstation2)
	(supports instrument16 infrared3)
	(calibration_target instrument16 groundstation2)
	(supports instrument17 infrared2)
	(supports instrument17 image1)
	(calibration_target instrument17 groundstation4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet6)
)
(:goal (and
	(pointing satellite2 planet5)
	(have_image planet5 image1)
	(have_image planet6 infrared3)
	(have_image phenomenon7 infrared3)
	(have_image phenomenon8 image1)
	(have_image star9 infrared3)
	(have_image star10 infrared2)
))

)
