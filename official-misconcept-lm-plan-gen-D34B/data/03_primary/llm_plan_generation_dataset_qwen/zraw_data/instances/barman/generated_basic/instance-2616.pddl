(define (problem prob)
 (:domain barman)
 (:objects 
      shaker193 - shaker
      left right - hand
      shot407 shot122 - shot
      ingredient335 ingredient376 ingredient172 - ingredient
      cocktail180 - cocktail
      dispenser366 dispenser194 dispenser310 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker193)
  (ontable shot407)
  (ontable shot122)
  (dispenses dispenser366 ingredient335)
  (dispenses dispenser194 ingredient376)
  (dispenses dispenser310 ingredient172)
  (clean shaker193)
  (clean shot407)
  (clean shot122)
  (empty shaker193)
  (empty shot407)
  (empty shot122)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker193 l0)
  (shaker-level shaker193 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail180 ingredient335)
  (cocktail-part2 cocktail180 ingredient172)
)
 (:goal
  (and
      (contains shot407 cocktail180)
)))
