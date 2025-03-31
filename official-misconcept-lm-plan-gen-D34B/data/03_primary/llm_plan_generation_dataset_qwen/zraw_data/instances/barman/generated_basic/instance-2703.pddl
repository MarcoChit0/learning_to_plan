(define (problem prob)
 (:domain barman)
 (:objects 
      shaker56 - shaker
      left right - hand
      shot259 shot219 shot473 - shot
      ingredient451 ingredient102 - ingredient
      cocktail137 - cocktail
      dispenser152 dispenser427 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker56)
  (ontable shot259)
  (ontable shot219)
  (ontable shot473)
  (dispenses dispenser152 ingredient451)
  (dispenses dispenser427 ingredient102)
  (clean shaker56)
  (clean shot259)
  (clean shot219)
  (clean shot473)
  (empty shaker56)
  (empty shot259)
  (empty shot219)
  (empty shot473)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker56 l0)
  (shaker-level shaker56 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail137 ingredient451)
  (cocktail-part2 cocktail137 ingredient102)
)
 (:goal
  (and
      (contains shot259 cocktail137)
      (contains shot219 ingredient451)
)))
