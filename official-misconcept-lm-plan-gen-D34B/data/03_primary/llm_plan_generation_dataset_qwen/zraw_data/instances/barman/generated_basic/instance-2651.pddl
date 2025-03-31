(define (problem prob)
 (:domain barman)
 (:objects 
      shaker479 - shaker
      left right - hand
      shot225 shot458 shot467 shot43 - shot
      ingredient416 ingredient449 - ingredient
      cocktail127 - cocktail
      dispenser486 dispenser245 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker479)
  (ontable shot225)
  (ontable shot458)
  (ontable shot467)
  (ontable shot43)
  (dispenses dispenser486 ingredient416)
  (dispenses dispenser245 ingredient449)
  (clean shaker479)
  (clean shot225)
  (clean shot458)
  (clean shot467)
  (clean shot43)
  (empty shaker479)
  (empty shot225)
  (empty shot458)
  (empty shot467)
  (empty shot43)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker479 l0)
  (shaker-level shaker479 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail127 ingredient416)
  (cocktail-part2 cocktail127 ingredient449)
)
 (:goal
  (and
      (contains shot225 cocktail127)
      (contains shot458 ingredient449)
      (contains shot467 cocktail127)
)))
