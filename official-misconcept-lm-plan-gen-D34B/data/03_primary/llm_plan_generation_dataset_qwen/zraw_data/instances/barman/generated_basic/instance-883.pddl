(define (problem prob)
 (:domain barman)
 (:objects 
      shaker469 - shaker
      left right - hand
      shot384 - shot
      ingredient388 ingredient493 - ingredient
      cocktail1 - cocktail
      dispenser346 dispenser249 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker469)
  (ontable shot384)
  (dispenses dispenser346 ingredient388)
  (dispenses dispenser249 ingredient493)
  (clean shaker469)
  (clean shot384)
  (empty shaker469)
  (empty shot384)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker469 l0)
  (shaker-level shaker469 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient388)
  (cocktail-part2 cocktail1 ingredient493)
)
 (:goal
  (and
      (contains shot384 cocktail1)
)))
