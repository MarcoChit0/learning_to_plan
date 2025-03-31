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
	infrared3 - mode
	image2 - mode
	infrared1 - mode
	image0 - mode
	groundstation1 - direction
	star0 - direction
	groundstation2 - direction
	groundstation3 - direction
	planet4 - direction
	star5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 image2)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument3 image0)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image2)
	(supports instrument4 infrared3)
	(calibration_target instrument4 star0)
	(supports instrument5 image0)
	(supports instrument5 image2)
	(supports instrument5 infrared3)
	(calibration_target instrument5 groundstation2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument6 image0)
	(supports instrument6 infrared3)
	(supports instrument6 image2)
	(calibration_target instrument6 groundstation3)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
)
(:goal (and
	(have_image planet4 infrared3)
	(have_image star5 infrared3)
	(have_image phenomenon6 image0)
))

)
