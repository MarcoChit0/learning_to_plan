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
	image0 - mode
	spectrograph1 - mode
	star0 - direction
	star1 - direction
	star2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 spectrograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 star1)
	(supports instrument3 spectrograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(supports instrument4 spectrograph1)
	(supports instrument4 image0)
	(calibration_target instrument4 star1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(pointing satellite1 star0)
	(have_image star2 image0)
	(have_image planet3 image0)
))

)
