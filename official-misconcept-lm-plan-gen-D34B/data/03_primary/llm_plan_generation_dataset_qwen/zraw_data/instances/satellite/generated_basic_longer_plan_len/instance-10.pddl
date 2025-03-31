(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite3 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	infrared0 - mode
	infrared2 - mode
	image1 - mode
	image3 - mode
	star0 - direction
	groundstation1 - direction
	groundstation4 - direction
	groundstation2 - direction
	groundstation3 - direction
	planet5 - direction
	phenomenon6 - direction
	star7 - direction
	planet8 - direction
	phenomenon9 - direction
	planet10 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared2)
	(supports instrument0 image3)
	(calibration_target instrument0 groundstation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 infrared0)
	(supports instrument1 image1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation2)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument2 image3)
	(supports instrument2 image1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image3)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star0)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet8)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 image1)
	(supports instrument5 infrared0)
	(supports instrument5 image3)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 infrared0)
	(supports instrument7 infrared2)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 image3)
	(calibration_target instrument8 groundstation3)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star7)
	(supports instrument9 infrared2)
	(calibration_target instrument9 groundstation3)
	(on_board instrument9 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation3)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite1 planet5)
	(have_image planet5 infrared2)
	(have_image phenomenon6 infrared2)
	(have_image star7 image1)
	(have_image planet8 infrared2)
	(have_image phenomenon9 infrared0)
	(have_image planet10 image1)
))

)
