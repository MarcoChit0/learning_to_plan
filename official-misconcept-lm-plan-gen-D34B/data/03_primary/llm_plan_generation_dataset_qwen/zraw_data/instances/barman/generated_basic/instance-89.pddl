(define (problem prob)
 (:domain barman)
 (:objects 
      shaker435 - shaker
      left right - hand
      shot71 - shot
      ingredient308 ingredient296 ingredient22 ingredient422 - ingredient
      cocktail1 - cocktail
      dispenser457 dispenser168 dispenser361 dispenser398 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker435)
  (ontable shot71)
  (dispenses dispenser457 ingredient308)
  (dispenses dispenser168 ingredient296)
  (dispenses dispenser361 ingredient22)
  (dispenses dispenser398 ingredient422)
  (clean shaker435)
  (clean shot71)
  (empty shaker435)
  (empty shot71)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker435 l0)
  (shaker-level shaker435 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient422)
  (cocktail-part2 cocktail1 ingredient308)
)
 (:goal
  (and
      (contains shot71 cocktail1)
)))
