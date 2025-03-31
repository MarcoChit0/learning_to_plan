(define (problem prob)
 (:domain barman)
 (:objects 
      shaker118 - shaker
      left right - hand
      shot77 shot320 - shot
      ingredient443 ingredient9 ingredient312 - ingredient
      cocktail391 - cocktail
      dispenser393 dispenser131 dispenser281 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker118)
  (ontable shot77)
  (ontable shot320)
  (dispenses dispenser393 ingredient443)
  (dispenses dispenser131 ingredient9)
  (dispenses dispenser281 ingredient312)
  (clean shaker118)
  (clean shot77)
  (clean shot320)
  (empty shaker118)
  (empty shot77)
  (empty shot320)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker118 l0)
  (shaker-level shaker118 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail391 ingredient9)
  (cocktail-part2 cocktail391 ingredient443)
)
 (:goal
  (and
      (contains shot77 cocktail391)
)))
