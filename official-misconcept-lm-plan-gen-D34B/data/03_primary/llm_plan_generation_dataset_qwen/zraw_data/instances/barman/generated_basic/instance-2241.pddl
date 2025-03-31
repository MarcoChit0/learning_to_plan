(define (problem prob)
 (:domain barman)
 (:objects 
      shaker72 - shaker
      left right - hand
      shot298 shot459 shot348 - shot
      ingredient58 ingredient267 ingredient334 ingredient238 - ingredient
      cocktail262 - cocktail
      dispenser260 dispenser324 dispenser76 dispenser454 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker72)
  (ontable shot298)
  (ontable shot459)
  (ontable shot348)
  (dispenses dispenser260 ingredient58)
  (dispenses dispenser324 ingredient267)
  (dispenses dispenser76 ingredient334)
  (dispenses dispenser454 ingredient238)
  (clean shaker72)
  (clean shot298)
  (clean shot459)
  (clean shot348)
  (empty shaker72)
  (empty shot298)
  (empty shot459)
  (empty shot348)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker72 l0)
  (shaker-level shaker72 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail262 ingredient238)
  (cocktail-part2 cocktail262 ingredient58)
)
 (:goal
  (and
      (contains shot298 cocktail262)
      (contains shot459 ingredient267)
)))
