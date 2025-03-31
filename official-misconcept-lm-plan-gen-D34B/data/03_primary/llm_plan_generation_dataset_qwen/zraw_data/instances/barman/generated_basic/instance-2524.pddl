(define (problem prob)
 (:domain barman)
 (:objects 
      shaker492 - shaker
      left right - hand
      shot401 shot12 shot362 - shot
      ingredient418 ingredient115 - ingredient
      cocktail8 - cocktail
      dispenser374 dispenser51 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker492)
  (ontable shot401)
  (ontable shot12)
  (ontable shot362)
  (dispenses dispenser374 ingredient418)
  (dispenses dispenser51 ingredient115)
  (clean shaker492)
  (clean shot401)
  (clean shot12)
  (clean shot362)
  (empty shaker492)
  (empty shot401)
  (empty shot12)
  (empty shot362)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker492 l0)
  (shaker-level shaker492 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail8 ingredient115)
  (cocktail-part2 cocktail8 ingredient418)
)
 (:goal
  (and
      (contains shot401 cocktail8)
      (contains shot12 ingredient115)
)))
