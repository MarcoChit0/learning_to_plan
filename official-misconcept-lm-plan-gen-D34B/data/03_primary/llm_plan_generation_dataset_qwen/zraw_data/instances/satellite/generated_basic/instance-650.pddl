(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	image0 - mode
	spectrograph1 - mode
	groundstation0 - direction
	groundstation4 - direction
	groundstation2 - direction
	groundstation3 - direction
	star5 - direction
	star1 - direction
	star6 - direction
	phenomenon7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 image0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation2)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 image0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation4)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation3)
	(calibration_target instrument3 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument4 image0)
	(calibration_target instrument4 star1)
	(calibration_target instrument4 star5)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet8)
)
(:goal (and
	(pointing satellite0 star6)
	(have_image star6 spectrograph1)
	(have_image phenomenon7 spectrograph1)
	(have_image planet8 spectrograph1)
))

)
