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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	infrared1 - mode
	image0 - mode
	star0 - direction
	star1 - direction
	star2 - direction
	phenomenon3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star2)
	(supports instrument1 image0)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared1)
	(supports instrument2 image0)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon3)
	(supports instrument3 image0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star2)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument5 infrared1)
	(supports instrument5 image0)
	(calibration_target instrument5 star2)
	(supports instrument6 infrared1)
	(supports instrument6 image0)
	(calibration_target instrument6 star1)
	(supports instrument7 infrared1)
	(supports instrument7 image0)
	(calibration_target instrument7 star2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(have_image phenomenon3 infrared1)
	(have_image planet4 infrared1)
))

)
