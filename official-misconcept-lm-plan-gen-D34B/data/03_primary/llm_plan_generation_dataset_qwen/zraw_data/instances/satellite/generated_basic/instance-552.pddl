(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	spectrograph2 - mode
	image1 - mode
	spectrograph0 - mode
	groundstation1 - direction
	star0 - direction
	star2 - direction
	planet3 - direction
	phenomenon4 - direction
	planet5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 spectrograph2)
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph2)
	(supports instrument2 image1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument3 image1)
	(supports instrument3 spectrograph2)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument4 image1)
	(supports instrument4 spectrograph2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet5)
)
(:goal (and
	(pointing satellite2 star0)
	(have_image star2 spectrograph0)
	(have_image planet3 image1)
	(have_image phenomenon4 image1)
	(have_image planet5 spectrograph0)
	(have_image phenomenon6 image1)
))

)
