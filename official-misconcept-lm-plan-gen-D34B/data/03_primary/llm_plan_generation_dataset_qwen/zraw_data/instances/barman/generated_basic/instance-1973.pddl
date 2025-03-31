(define (problem prob)
 (:domain barman)
 (:objects 
      shaker99 - shaker
      left right - hand
      shot342 shot182 - shot
      ingredient81 ingredient184 ingredient313 - ingredient
      cocktail216 - cocktail
      dispenser458 dispenser349 dispenser102 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker99)
  (ontable shot342)
  (ontable shot182)
  (dispenses dispenser458 ingredient81)
  (dispenses dispenser349 ingredient184)
  (dispenses dispenser102 ingredient313)
  (clean shaker99)
  (clean shot342)
  (clean shot182)
  (empty shaker99)
  (empty shot342)
  (empty shot182)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker99 l0)
  (shaker-level shaker99 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail216 ingredient313)
  (cocktail-part2 cocktail216 ingredient81)
)
 (:goal
  (and
      (contains shot342 cocktail216)
)))
