(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	spectrograph0 - mode
	thermograph1 - mode
	spectrograph2 - mode
	star1 - direction
	groundstation0 - direction
	phenomenon2 - direction
	planet3 - direction
	planet4 - direction
	phenomenon5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 thermograph1)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star1)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph2)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
	(supports instrument4 spectrograph2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet4)
)
(:goal (and
	(pointing satellite1 planet4)
	(pointing satellite2 planet4)
	(have_image phenomenon2 thermograph1)
	(have_image planet3 thermograph1)
	(have_image planet4 spectrograph2)
	(have_image phenomenon5 spectrograph0)
	(have_image planet6 thermograph1)
))

)
