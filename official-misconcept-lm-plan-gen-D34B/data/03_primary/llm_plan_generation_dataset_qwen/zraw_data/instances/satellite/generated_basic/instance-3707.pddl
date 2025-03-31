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
	infrared1 - mode
	infrared2 - mode
	infrared0 - mode
	star0 - direction
	star1 - direction
	star5 - direction
	groundstation3 - direction
	groundstation2 - direction
	groundstation4 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation3)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared1)
	(supports instrument3 infrared2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star1)
	(calibration_target instrument3 star5)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 infrared2)
	(calibration_target instrument5 star5)
	(calibration_target instrument5 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument6 infrared2)
	(calibration_target instrument6 star5)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 infrared2)
	(calibration_target instrument7 star0)
	(calibration_target instrument7 star5)
	(supports instrument8 infrared1)
	(supports instrument8 infrared0)
	(supports instrument8 infrared2)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared0)
	(supports instrument9 infrared2)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 infrared1)
	(supports instrument10 infrared0)
	(supports instrument10 infrared2)
	(calibration_target instrument10 groundstation2)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation4)
	(supports instrument12 infrared2)
	(supports instrument12 infrared1)
	(supports instrument12 infrared0)
	(calibration_target instrument12 groundstation4)
	(calibration_target instrument12 star1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star5)
	(supports instrument14 infrared1)
	(calibration_target instrument14 groundstation3)
	(supports instrument15 infrared1)
	(calibration_target instrument15 groundstation3)
	(calibration_target instrument15 groundstation2)
	(supports instrument16 infrared2)
	(supports instrument16 infrared1)
	(calibration_target instrument16 groundstation4)
	(calibration_target instrument16 groundstation2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(on_board instrument15 satellite2)
	(on_board instrument16 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(pointing satellite0 groundstation4)
	(have_image phenomenon6 infrared1)
))

)
