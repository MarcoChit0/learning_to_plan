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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	infrared2 - mode
	infrared1 - mode
	image0 - mode
	star0 - direction
	groundstation1 - direction
	groundstation3 - direction
	groundstation2 - direction
	star4 - direction
	planet5 - direction
	star6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image0)
	(supports instrument3 infrared2)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument4 infrared2)
	(supports instrument4 image0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation2)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 infrared1)
	(calibration_target instrument6 groundstation2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star8)
)
(:goal (and
	(pointing satellite0 star8)
	(have_image star4 infrared1)
	(have_image planet5 infrared1)
	(have_image star6 infrared1)
	(have_image planet7 infrared2)
	(have_image star8 infrared1)
))

)
