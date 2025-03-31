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
	spectrograph0 - mode
	image1 - mode
	groundstation1 - direction
	star5 - direction
	groundstation0 - direction
	groundstation3 - direction
	groundstation4 - direction
	groundstation2 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star5)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 image1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star5)
	(supports instrument2 image1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation0)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon7)
	(supports instrument4 image1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation2)
	(calibration_target instrument4 groundstation4)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
)
(:goal (and
	(pointing satellite0 groundstation4)
	(have_image phenomenon6 image1)
	(have_image phenomenon7 spectrograph0)
	(have_image planet8 image1)
))

)
