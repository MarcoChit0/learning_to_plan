(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	infrared2 - mode
	infrared1 - mode
	infrared0 - mode
	groundstation3 - direction
	star4 - direction
	groundstation0 - direction
	star5 - direction
	groundstation1 - direction
	star2 - direction
	phenomenon6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation1)
	(calibration_target instrument0 star5)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument1 infrared0)
	(supports instrument1 infrared1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star5)
	(supports instrument4 infrared1)
	(supports instrument4 infrared0)
	(supports instrument4 infrared2)
	(calibration_target instrument4 star2)
	(calibration_target instrument4 groundstation1)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(pointing satellite2 star8)
	(have_image phenomenon6 infrared0)
	(have_image planet7 infrared0)
	(have_image star8 infrared1)
))

)
