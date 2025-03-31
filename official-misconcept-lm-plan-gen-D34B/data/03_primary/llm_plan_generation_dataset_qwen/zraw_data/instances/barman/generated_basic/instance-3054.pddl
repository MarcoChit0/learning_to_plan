(define (problem prob)
 (:domain barman)
 (:objects 
      shaker308 - shaker
      left right - hand
      shot56 shot22 - shot
      ingredient449 ingredient208 ingredient496 - ingredient
      cocktail368 - cocktail
      dispenser171 dispenser462 dispenser84 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker308)
  (ontable shot56)
  (ontable shot22)
  (dispenses dispenser171 ingredient449)
  (dispenses dispenser462 ingredient208)
  (dispenses dispenser84 ingredient496)
  (clean shaker308)
  (clean shot56)
  (clean shot22)
  (empty shaker308)
  (empty shot56)
  (empty shot22)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker308 l0)
  (shaker-level shaker308 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail368 ingredient208)
  (cocktail-part2 cocktail368 ingredient449)
)
 (:goal
  (and
      (contains shot56 cocktail368)
)))
