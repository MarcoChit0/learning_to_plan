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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	infrared1 - mode
	image2 - mode
	image0 - mode
	groundstation1 - direction
	groundstation0 - direction
	groundstation2 - direction
	groundstation4 - direction
	star3 - direction
	groundstation5 - direction
	star6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 image0)
	(supports instrument2 infrared1)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation0)
	(calibration_target instrument2 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation5)
	(supports instrument3 image2)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation1)
	(calibration_target instrument4 groundstation5)
	(supports instrument5 image2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation5)
	(calibration_target instrument5 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument6 infrared1)
	(supports instrument6 image2)
	(calibration_target instrument6 groundstation4)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 image2)
	(supports instrument7 infrared1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation5)
	(calibration_target instrument7 star3)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
)
(:goal (and
	(pointing satellite1 star3)
	(have_image star6 image2)
	(have_image planet7 image2)
	(have_image phenomenon8 image0)
))

)
