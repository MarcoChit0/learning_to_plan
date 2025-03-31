(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	infrared0 - mode
	image1 - mode
	image3 - mode
	spectrograph2 - mode
	star1 - direction
	star4 - direction
	star5 - direction
	star0 - direction
	star3 - direction
	star2 - direction
	star6 - direction
)
(:init
	(supports instrument0 image3)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared0)
	(supports instrument1 spectrograph2)
	(supports instrument1 image3)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star2)
	(calibration_target instrument2 star4)
	(supports instrument3 image1)
	(supports instrument3 image3)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star3)
	(calibration_target instrument3 star5)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star3)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared0)
	(supports instrument5 image1)
	(supports instrument5 image3)
	(calibration_target instrument5 star2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(pointing satellite0 star2)
	(pointing satellite2 star5)
	(have_image star6 image3)
))

)
