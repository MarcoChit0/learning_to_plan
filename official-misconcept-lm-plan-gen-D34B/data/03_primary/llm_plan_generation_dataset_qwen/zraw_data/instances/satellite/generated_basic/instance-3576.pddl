(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	spectrograph0 - mode
	infrared1 - mode
	star0 - direction
	star2 - direction
	groundstation3 - direction
	groundstation1 - direction
	groundstation4 - direction
	star5 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 spectrograph0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument3 spectrograph0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation4)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
)
(:goal (and
	(have_image star5 spectrograph0)
))

)
