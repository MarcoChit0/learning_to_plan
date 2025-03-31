(define (problem prob)
 (:domain barman)
 (:objects 
      shaker39 - shaker
      left right - hand
      shot437 - shot
      ingredient471 ingredient228 ingredient331 ingredient259 - ingredient
      cocktail334 - cocktail
      dispenser127 dispenser487 dispenser420 dispenser289 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker39)
  (ontable shot437)
  (dispenses dispenser127 ingredient471)
  (dispenses dispenser487 ingredient228)
  (dispenses dispenser420 ingredient331)
  (dispenses dispenser289 ingredient259)
  (clean shaker39)
  (clean shot437)
  (empty shaker39)
  (empty shot437)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker39 l0)
  (shaker-level shaker39 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail334 ingredient228)
  (cocktail-part2 cocktail334 ingredient331)
)
 (:goal
  (and
      (contains shot437 cocktail334)
)))
