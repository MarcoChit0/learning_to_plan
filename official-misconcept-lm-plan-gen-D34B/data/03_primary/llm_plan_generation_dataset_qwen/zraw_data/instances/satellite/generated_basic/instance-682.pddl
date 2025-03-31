(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	image0 - mode
	infrared1 - mode
	groundstation3 - direction
	groundstation2 - direction
	star4 - direction
	star0 - direction
	star1 - direction
	star5 - direction
	phenomenon6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 image0)
	(calibration_target instrument0 star5)
	(calibration_target instrument0 star1)
	(supports instrument1 image0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star0)
	(calibration_target instrument1 star1)
	(supports instrument2 image0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation2)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument4 infrared1)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star1)
	(calibration_target instrument5 star4)
	(supports instrument6 infrared1)
	(supports instrument6 image0)
	(calibration_target instrument6 star1)
	(calibration_target instrument6 star0)
	(supports instrument7 image0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star5)
	(supports instrument8 image0)
	(calibration_target instrument8 star5)
	(calibration_target instrument8 star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
)
(:goal (and
	(have_image phenomenon6 image0)
	(have_image planet7 infrared1)
	(have_image planet8 image0)
))

)
