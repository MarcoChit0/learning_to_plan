(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	thermograph1 - mode
	thermograph0 - mode
	infrared2 - mode
	groundstation2 - direction
	star1 - direction
	groundstation0 - direction
	groundstation5 - direction
	star3 - direction
	groundstation4 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
	star9 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star3)
	(supports instrument1 thermograph1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation5)
	(supports instrument4 thermograph1)
	(supports instrument4 infrared2)
	(calibration_target instrument4 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star7)
	(supports instrument5 thermograph0)
	(supports instrument5 infrared2)
	(calibration_target instrument5 star3)
	(calibration_target instrument5 groundstation5)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument6 thermograph0)
	(supports instrument6 infrared2)
	(calibration_target instrument6 star3)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 infrared2)
	(supports instrument7 thermograph1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 thermograph1)
	(supports instrument8 infrared2)
	(calibration_target instrument8 star1)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 infrared2)
	(supports instrument9 thermograph1)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 infrared2)
	(calibration_target instrument10 star1)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 infrared2)
	(calibration_target instrument11 groundstation2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 star3)
	(calibration_target instrument12 star1)
	(supports instrument13 thermograph1)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 groundstation2)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 thermograph1)
	(supports instrument14 thermograph0)
	(supports instrument14 infrared2)
	(calibration_target instrument14 star3)
	(calibration_target instrument14 star1)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 groundstation5)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 thermograph0)
	(supports instrument16 thermograph1)
	(calibration_target instrument16 groundstation4)
	(calibration_target instrument16 star3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(have_image star6 thermograph0)
	(have_image star7 thermograph0)
	(have_image phenomenon8 thermograph1)
	(have_image star9 infrared2)
))

)
