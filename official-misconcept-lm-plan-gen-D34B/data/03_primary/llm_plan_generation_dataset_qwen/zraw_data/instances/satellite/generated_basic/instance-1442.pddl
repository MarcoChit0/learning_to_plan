(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	infrared3 - mode
	spectrograph0 - mode
	image1 - mode
	image2 - mode
	star0 - direction
	planet1 - direction
	star2 - direction
	star3 - direction
)
(:init
	(supports instrument0 image2)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 image2)
	(calibration_target instrument1 star0)
	(supports instrument2 image2)
	(supports instrument2 infrared3)
	(supports instrument2 image1)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared3)
	(calibration_target instrument3 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument4 spectrograph0)
	(supports instrument4 infrared3)
	(supports instrument4 image2)
	(calibration_target instrument4 star0)
	(supports instrument5 image1)
	(calibration_target instrument5 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet1)
)
(:goal (and
	(have_image planet1 spectrograph0)
	(have_image star2 infrared3)
	(have_image star3 spectrograph0)
))

)
