(define (problem prob)
 (:domain barman)
 (:objects 
      shaker132 - shaker
      left right - hand
      shot64 shot458 - shot
      ingredient422 ingredient169 ingredient204 - ingredient
      cocktail268 - cocktail
      dispenser468 dispenser89 dispenser264 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker132)
  (ontable shot64)
  (ontable shot458)
  (dispenses dispenser468 ingredient422)
  (dispenses dispenser89 ingredient169)
  (dispenses dispenser264 ingredient204)
  (clean shaker132)
  (clean shot64)
  (clean shot458)
  (empty shaker132)
  (empty shot64)
  (empty shot458)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker132 l0)
  (shaker-level shaker132 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail268 ingredient204)
  (cocktail-part2 cocktail268 ingredient169)
)
 (:goal
  (and
      (contains shot64 cocktail268)
)))
