(define (problem prob)
 (:domain barman)
 (:objects 
      shaker246 - shaker
      left right - hand
      shot394 shot185 - shot
      ingredient499 ingredient57 - ingredient
      cocktail255 - cocktail
      dispenser463 dispenser1 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker246)
  (ontable shot394)
  (ontable shot185)
  (dispenses dispenser463 ingredient499)
  (dispenses dispenser1 ingredient57)
  (clean shaker246)
  (clean shot394)
  (clean shot185)
  (empty shaker246)
  (empty shot394)
  (empty shot185)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker246 l0)
  (shaker-level shaker246 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail255 ingredient57)
  (cocktail-part2 cocktail255 ingredient499)
)
 (:goal
  (and
      (contains shot394 cocktail255)
)))
