(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	infrared3 - mode
	infrared0 - mode
	infrared1 - mode
	infrared2 - mode
	star0 - direction
	star1 - direction
	planet2 - direction
	star3 - direction
	star4 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared3)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star1)
	(supports instrument4 infrared3)
	(supports instrument4 infrared2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared2)
	(supports instrument5 infrared1)
	(supports instrument5 infrared3)
	(calibration_target instrument5 star1)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
)
(:goal (and
	(pointing satellite1 star6)
	(have_image planet2 infrared2)
	(have_image star3 infrared2)
	(have_image star4 infrared0)
	(have_image star5 infrared3)
	(have_image star6 infrared3)
))

)
