(define (problem prob)
 (:domain barman)
 (:objects 
      shaker10 - shaker
      left right - hand
      shot21 shot312 - shot
      ingredient5 ingredient330 ingredient271 ingredient150 - ingredient
      cocktail1 - cocktail
      dispenser373 dispenser315 dispenser348 dispenser159 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker10)
  (ontable shot21)
  (ontable shot312)
  (dispenses dispenser373 ingredient5)
  (dispenses dispenser315 ingredient330)
  (dispenses dispenser348 ingredient271)
  (dispenses dispenser159 ingredient150)
  (clean shaker10)
  (clean shot21)
  (clean shot312)
  (empty shaker10)
  (empty shot21)
  (empty shot312)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker10 l0)
  (shaker-level shaker10 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient5)
  (cocktail-part2 cocktail1 ingredient150)
)
 (:goal
  (and
      (contains shot21 cocktail1)
)))
