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
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	infrared0 - mode
	infrared2 - mode
	infrared1 - mode
	star4 - direction
	groundstation0 - direction
	groundstation5 - direction
	groundstation2 - direction
	star1 - direction
	groundstation3 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	star8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation0)
	(calibration_target instrument0 star4)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation0)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation5)
	(supports instrument4 infrared2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation2)
	(calibration_target instrument4 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon7)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation3)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 infrared2)
	(supports instrument6 infrared1)
	(calibration_target instrument6 groundstation2)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star1)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 infrared2)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 infrared0)
	(supports instrument9 infrared1)
	(supports instrument9 infrared2)
	(calibration_target instrument9 groundstation2)
	(calibration_target instrument9 groundstation5)
	(supports instrument10 infrared1)
	(supports instrument10 infrared0)
	(supports instrument10 infrared2)
	(calibration_target instrument10 groundstation3)
	(calibration_target instrument10 star1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
)
(:goal (and
	(pointing satellite0 star1)
	(have_image phenomenon6 infrared1)
	(have_image phenomenon7 infrared2)
	(have_image star8 infrared1)
	(have_image planet9 infrared1)
))

)
