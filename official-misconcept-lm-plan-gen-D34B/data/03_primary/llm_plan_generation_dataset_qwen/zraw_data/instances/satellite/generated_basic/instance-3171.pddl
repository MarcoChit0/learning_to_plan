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
	spectrograph0 - mode
	image1 - mode
	star0 - direction
	planet1 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star0)
	(supports instrument1 image1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 image1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 star0)
	(supports instrument4 image1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star0)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon2)
)
(:goal (and
	(pointing satellite1 planet1)
	(have_image planet1 spectrograph0)
	(have_image phenomenon2 image1)
))

)
