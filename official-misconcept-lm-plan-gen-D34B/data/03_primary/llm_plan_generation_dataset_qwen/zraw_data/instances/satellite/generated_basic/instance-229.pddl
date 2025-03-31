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
	satellite3 - satellite
	instrument7 - instrument
	satellite4 - satellite
	instrument8 - instrument
	image1 - mode
	infrared0 - mode
	star4 - direction
	groundstation2 - direction
	groundstation3 - direction
	groundstation1 - direction
	groundstation0 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 image1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 image1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation3)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation1)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star8)
	(supports instrument8 image1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation0)
	(on_board instrument8 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star8)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(have_image phenomenon5 infrared0)
	(have_image phenomenon6 image1)
	(have_image planet7 infrared0)
	(have_image star8 infrared0)
))

)
