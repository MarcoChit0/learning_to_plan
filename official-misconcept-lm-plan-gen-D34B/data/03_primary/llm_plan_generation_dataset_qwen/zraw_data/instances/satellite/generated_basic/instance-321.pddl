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
	image1 - mode
	spectrograph2 - mode
	image3 - mode
	infrared0 - mode
	star3 - direction
	groundstation1 - direction
	star4 - direction
	groundstation0 - direction
	star5 - direction
	star2 - direction
	star6 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 star2)
	(supports instrument2 image1)
	(calibration_target instrument2 star2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star4)
	(calibration_target instrument3 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument4 image3)
	(supports instrument4 spectrograph2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation0)
	(calibration_target instrument4 star4)
	(supports instrument5 image1)
	(supports instrument5 image3)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star2)
	(calibration_target instrument5 star5)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(pointing satellite0 star2)
	(pointing satellite1 star6)
	(have_image star6 spectrograph2)
))

)
