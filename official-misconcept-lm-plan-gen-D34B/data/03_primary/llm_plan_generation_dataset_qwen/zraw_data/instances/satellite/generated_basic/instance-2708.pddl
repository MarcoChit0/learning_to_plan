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
	image3 - mode
	infrared1 - mode
	spectrograph2 - mode
	star1 - direction
	groundstation2 - direction
	groundstation3 - direction
	groundstation0 - direction
	planet4 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 spectrograph0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument3 infrared1)
	(supports instrument3 image3)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(have_image planet4 infrared1)
))

)
