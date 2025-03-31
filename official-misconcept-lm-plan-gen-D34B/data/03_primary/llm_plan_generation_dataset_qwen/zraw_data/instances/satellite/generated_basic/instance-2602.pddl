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
	instrument6 - instrument
	instrument7 - instrument
	thermograph3 - mode
	infrared1 - mode
	spectrograph2 - mode
	spectrograph0 - mode
	star1 - direction
	groundstation0 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph2)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 spectrograph2)
	(supports instrument4 spectrograph0)
	(supports instrument4 thermograph3)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared1)
	(supports instrument5 spectrograph0)
	(supports instrument5 thermograph3)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 spectrograph0)
	(supports instrument7 infrared1)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
)
(:goal (and
	(have_image phenomenon2 infrared1)
))

)
