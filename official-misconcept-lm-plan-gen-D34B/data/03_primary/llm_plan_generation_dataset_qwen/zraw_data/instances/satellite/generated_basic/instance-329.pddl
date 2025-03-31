(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	spectrograph2 - mode
	image1 - mode
	infrared3 - mode
	infrared0 - mode
	star4 - direction
	star1 - direction
	star2 - direction
	groundstation0 - direction
	star3 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared3)
	(calibration_target instrument1 star5)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star2)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 spectrograph2)
	(supports instrument4 image1)
	(calibration_target instrument4 star3)
	(calibration_target instrument4 star5)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument5 spectrograph2)
	(supports instrument5 infrared3)
	(calibration_target instrument5 star5)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(pointing satellite0 groundstation0)
	(pointing satellite2 star2)
	(have_image star6 image1)
))

)
