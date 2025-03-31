(define (problem prob)
 (:domain barman)
 (:objects 
      shaker483 - shaker
      left right - hand
      shot260 shot202 shot3 - shot
      ingredient208 ingredient223 - ingredient
      cocktail466 - cocktail
      dispenser116 dispenser448 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker483)
  (ontable shot260)
  (ontable shot202)
  (ontable shot3)
  (dispenses dispenser116 ingredient208)
  (dispenses dispenser448 ingredient223)
  (clean shaker483)
  (clean shot260)
  (clean shot202)
  (clean shot3)
  (empty shaker483)
  (empty shot260)
  (empty shot202)
  (empty shot3)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker483 l0)
  (shaker-level shaker483 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail466 ingredient208)
  (cocktail-part2 cocktail466 ingredient223)
)
 (:goal
  (and
      (contains shot260 cocktail466)
      (contains shot202 ingredient223)
)))
