(define (problem prob)
 (:domain barman)
 (:objects 
      shaker262 - shaker
      left right - hand
      shot269 shot263 - shot
      ingredient218 ingredient320 ingredient461 - ingredient
      cocktail201 - cocktail
      dispenser422 dispenser351 dispenser212 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker262)
  (ontable shot269)
  (ontable shot263)
  (dispenses dispenser422 ingredient218)
  (dispenses dispenser351 ingredient320)
  (dispenses dispenser212 ingredient461)
  (clean shaker262)
  (clean shot269)
  (clean shot263)
  (empty shaker262)
  (empty shot269)
  (empty shot263)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker262 l0)
  (shaker-level shaker262 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail201 ingredient218)
  (cocktail-part2 cocktail201 ingredient461)
)
 (:goal
  (and
      (contains shot269 cocktail201)
)))
