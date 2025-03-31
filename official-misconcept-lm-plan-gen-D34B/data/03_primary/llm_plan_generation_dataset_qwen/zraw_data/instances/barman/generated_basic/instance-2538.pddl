(define (problem prob)
 (:domain barman)
 (:objects 
      shaker3 - shaker
      left right - hand
      shot467 shot312 - shot
      ingredient41 ingredient229 - ingredient
      cocktail451 - cocktail
      dispenser83 dispenser98 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker3)
  (ontable shot467)
  (ontable shot312)
  (dispenses dispenser83 ingredient41)
  (dispenses dispenser98 ingredient229)
  (clean shaker3)
  (clean shot467)
  (clean shot312)
  (empty shaker3)
  (empty shot467)
  (empty shot312)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker3 l0)
  (shaker-level shaker3 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail451 ingredient41)
  (cocktail-part2 cocktail451 ingredient229)
)
 (:goal
  (and
      (contains shot467 cocktail451)
)))
