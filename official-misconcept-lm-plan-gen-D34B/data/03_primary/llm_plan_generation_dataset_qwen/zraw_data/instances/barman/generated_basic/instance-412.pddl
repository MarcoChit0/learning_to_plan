(define (problem prob)
 (:domain barman)
 (:objects 
      shaker16 - shaker
      left right - hand
      shot262 shot443 - shot
      ingredient383 ingredient305 - ingredient
      cocktail1 - cocktail
      dispenser4 dispenser44 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker16)
  (ontable shot262)
  (ontable shot443)
  (dispenses dispenser4 ingredient383)
  (dispenses dispenser44 ingredient305)
  (clean shaker16)
  (clean shot262)
  (clean shot443)
  (empty shaker16)
  (empty shot262)
  (empty shot443)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker16 l0)
  (shaker-level shaker16 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient305)
  (cocktail-part2 cocktail1 ingredient383)
)
 (:goal
  (and
      (contains shot262 cocktail1)
)))
