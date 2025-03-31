(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	image0 - mode
	spectrograph1 - mode
	groundstation1 - direction
	groundstation2 - direction
	star3 - direction
	star5 - direction
	groundstation0 - direction
	groundstation4 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument3 image0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation2)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 spectrograph1)
	(supports instrument4 image0)
	(calibration_target instrument4 star5)
	(calibration_target instrument4 star3)
	(supports instrument5 image0)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation4)
	(calibration_target instrument5 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet8)
)
(:goal (and
	(have_image phenomenon6 image0)
	(have_image phenomenon7 spectrograph1)
	(have_image planet8 image0)
))

)
