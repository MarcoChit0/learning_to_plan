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
	spectrograph1 - mode
	image0 - mode
	star0 - direction
	groundstation1 - direction
	groundstation4 - direction
	groundstation2 - direction
	groundstation3 - direction
	star5 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 image0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument4 spectrograph1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation3)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
)
(:goal (and
	(pointing satellite2 groundstation2)
	(have_image star5 image0)
))

)
