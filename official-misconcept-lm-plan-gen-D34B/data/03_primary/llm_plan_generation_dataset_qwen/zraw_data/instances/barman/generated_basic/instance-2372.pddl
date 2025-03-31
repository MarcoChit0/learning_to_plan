(define (problem prob)
 (:domain barman)
 (:objects 
      shaker477 - shaker
      left right - hand
      shot286 - shot
      ingredient119 ingredient354 ingredient260 ingredient342 - ingredient
      cocktail167 - cocktail
      dispenser66 dispenser141 dispenser172 dispenser188 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker477)
  (ontable shot286)
  (dispenses dispenser66 ingredient119)
  (dispenses dispenser141 ingredient354)
  (dispenses dispenser172 ingredient260)
  (dispenses dispenser188 ingredient342)
  (clean shaker477)
  (clean shot286)
  (empty shaker477)
  (empty shot286)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker477 l0)
  (shaker-level shaker477 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail167 ingredient260)
  (cocktail-part2 cocktail167 ingredient119)
)
 (:goal
  (and
      (contains shot286 cocktail167)
)))
