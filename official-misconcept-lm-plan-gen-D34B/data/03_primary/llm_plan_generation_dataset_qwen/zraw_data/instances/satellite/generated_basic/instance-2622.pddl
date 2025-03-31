(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	thermograph0 - mode
	image1 - mode
	spectrograph2 - mode
	spectrograph3 - mode
	groundstation1 - direction
	groundstation0 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image1)
	(supports instrument3 spectrograph3)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 spectrograph3)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon2)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation0)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
)
(:goal (and
	(pointing satellite0 groundstation0)
	(pointing satellite1 groundstation0)
	(have_image phenomenon2 image1)
))

)
