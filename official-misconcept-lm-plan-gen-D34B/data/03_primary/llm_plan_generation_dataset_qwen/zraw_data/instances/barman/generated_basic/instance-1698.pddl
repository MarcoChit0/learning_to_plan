(define (problem prob)
 (:domain barman)
 (:objects 
      shaker191 - shaker
      left right - hand
      shot384 - shot
      ingredient446 ingredient462 ingredient406 ingredient343 - ingredient
      cocktail384 - cocktail
      dispenser222 dispenser292 dispenser392 dispenser391 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker191)
  (ontable shot384)
  (dispenses dispenser222 ingredient446)
  (dispenses dispenser292 ingredient462)
  (dispenses dispenser392 ingredient406)
  (dispenses dispenser391 ingredient343)
  (clean shaker191)
  (clean shot384)
  (empty shaker191)
  (empty shot384)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker191 l0)
  (shaker-level shaker191 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail384 ingredient406)
  (cocktail-part2 cocktail384 ingredient462)
)
 (:goal
  (and
      (contains shot384 cocktail384)
)))
