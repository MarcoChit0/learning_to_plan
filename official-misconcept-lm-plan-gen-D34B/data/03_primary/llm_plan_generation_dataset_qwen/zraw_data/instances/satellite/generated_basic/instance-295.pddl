(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	spectrograph1 - mode
	image0 - mode
	groundstation0 - direction
	groundstation3 - direction
	groundstation1 - direction
	star2 - direction
	planet4 - direction
	planet5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 spectrograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image0)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star2)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(have_image planet4 spectrograph1)
	(have_image planet5 image0)
	(have_image planet6 spectrograph1)
))

)
