(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	infrared2 - mode
	infrared0 - mode
	image1 - mode
	star1 - direction
	groundstation2 - direction
	star0 - direction
	phenomenon3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument2 infrared0)
	(supports instrument2 image1)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image1)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument4 infrared2)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 image1)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument7 image1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star0)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon4)
)
(:goal (and
	(pointing satellite1 groundstation2)
	(pointing satellite2 star0)
	(have_image phenomenon3 infrared0)
	(have_image phenomenon4 infrared0)
))

)
