(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	spectrograph0 - mode
	star1 - direction
	groundstation2 - direction
	star3 - direction
	star0 - direction
	star5 - direction
	star4 - direction
	star6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star3)
	(calibration_target instrument0 star4)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star5)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon8)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star4)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(pointing satellite0 planet7)
	(pointing satellite1 groundstation2)
	(have_image star6 spectrograph0)
	(have_image planet7 spectrograph0)
	(have_image phenomenon8 spectrograph0)
))

)
