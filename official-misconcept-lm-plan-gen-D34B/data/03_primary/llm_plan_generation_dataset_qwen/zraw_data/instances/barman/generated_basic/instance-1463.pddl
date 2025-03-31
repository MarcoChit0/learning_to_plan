(define (problem prob)
 (:domain barman)
 (:objects 
      shaker107 - shaker
      left right - hand
      shot31 shot111 shot36 - shot
      ingredient180 ingredient342 - ingredient
      cocktail423 - cocktail
      dispenser187 dispenser419 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker107)
  (ontable shot31)
  (ontable shot111)
  (ontable shot36)
  (dispenses dispenser187 ingredient180)
  (dispenses dispenser419 ingredient342)
  (clean shaker107)
  (clean shot31)
  (clean shot111)
  (clean shot36)
  (empty shaker107)
  (empty shot31)
  (empty shot111)
  (empty shot36)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker107 l0)
  (shaker-level shaker107 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail423 ingredient342)
  (cocktail-part2 cocktail423 ingredient180)
)
 (:goal
  (and
      (contains shot31 cocktail423)
      (contains shot111 cocktail423)
)))
