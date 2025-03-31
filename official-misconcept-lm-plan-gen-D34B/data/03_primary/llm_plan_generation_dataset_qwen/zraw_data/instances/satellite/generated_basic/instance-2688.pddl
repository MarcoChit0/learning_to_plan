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
	instrument5 - instrument
	spectrograph2 - mode
	image3 - mode
	spectrograph0 - mode
	infrared1 - mode
	star0 - direction
	star1 - direction
	groundstation3 - direction
	star2 - direction
	star4 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument2 spectrograph2)
	(supports instrument2 spectrograph0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star1)
	(supports instrument4 image3)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 spectrograph0)
	(supports instrument5 spectrograph2)
	(supports instrument5 image3)
	(calibration_target instrument5 star2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(pointing satellite1 star1)
	(have_image star4 spectrograph2)
))

)
