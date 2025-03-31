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
	infrared1 - mode
	infrared0 - mode
	star2 - direction
	star3 - direction
	star0 - direction
	groundstation1 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	star7 - direction
	star8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet9)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star0)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 infrared0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation1)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite1 phenomenon5)
	(have_image phenomenon4 infrared1)
	(have_image phenomenon5 infrared0)
	(have_image phenomenon6 infrared0)
	(have_image star7 infrared1)
	(have_image star8 infrared0)
	(have_image planet9 infrared1)
))

)
