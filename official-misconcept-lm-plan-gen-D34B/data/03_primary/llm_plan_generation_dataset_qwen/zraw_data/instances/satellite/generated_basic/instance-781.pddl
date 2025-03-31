(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	spectrograph0 - mode
	infrared1 - mode
	star1 - direction
	groundstation2 - direction
	groundstation3 - direction
	groundstation4 - direction
	groundstation0 - direction
	star5 - direction
	phenomenon6 - direction
	star7 - direction
	star8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 spectrograph0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star7)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
)
(:goal (and
	(have_image star5 infrared1)
	(have_image phenomenon6 infrared1)
	(have_image star7 infrared1)
	(have_image star8 infrared1)
))

)
