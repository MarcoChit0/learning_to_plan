(define (problem prob)
 (:domain barman)
 (:objects 
      shaker272 - shaker
      left right - hand
      shot435 shot356 - shot
      ingredient267 ingredient73 ingredient121 ingredient476 - ingredient
      cocktail268 - cocktail
      dispenser220 dispenser271 dispenser201 dispenser161 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker272)
  (ontable shot435)
  (ontable shot356)
  (dispenses dispenser220 ingredient267)
  (dispenses dispenser271 ingredient73)
  (dispenses dispenser201 ingredient121)
  (dispenses dispenser161 ingredient476)
  (clean shaker272)
  (clean shot435)
  (clean shot356)
  (empty shaker272)
  (empty shot435)
  (empty shot356)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker272 l0)
  (shaker-level shaker272 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail268 ingredient121)
  (cocktail-part2 cocktail268 ingredient476)
)
 (:goal
  (and
      (contains shot435 cocktail268)
)))
