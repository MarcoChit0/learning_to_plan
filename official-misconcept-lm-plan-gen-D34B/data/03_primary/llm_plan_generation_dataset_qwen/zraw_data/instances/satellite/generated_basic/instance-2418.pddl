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
	satellite3 - satellite
	instrument5 - instrument
	spectrograph2 - mode
	image1 - mode
	infrared3 - mode
	image0 - mode
	star0 - direction
	groundstation1 - direction
	star2 - direction
	groundstation3 - direction
	star4 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 image0)
	(calibration_target instrument0 star4)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument2 image1)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 image1)
	(calibration_target instrument3 star4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument4 infrared3)
	(calibration_target instrument4 star4)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star6)
	(supports instrument5 infrared3)
	(calibration_target instrument5 star4)
	(on_board instrument5 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
)
(:goal (and
	(pointing satellite0 star2)
	(pointing satellite2 groundstation1)
	(have_image star5 image1)
	(have_image star6 spectrograph2)
))

)
