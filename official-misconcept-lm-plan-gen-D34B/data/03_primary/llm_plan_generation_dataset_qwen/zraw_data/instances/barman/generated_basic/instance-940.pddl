(define (problem prob)
 (:domain barman)
 (:objects 
      shaker21 - shaker
      left right - hand
      shot246 shot271 shot449 - shot
      ingredient150 ingredient261 ingredient198 ingredient162 - ingredient
      cocktail1 - cocktail
      dispenser42 dispenser480 dispenser32 dispenser110 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker21)
  (ontable shot246)
  (ontable shot271)
  (ontable shot449)
  (dispenses dispenser42 ingredient150)
  (dispenses dispenser480 ingredient261)
  (dispenses dispenser32 ingredient198)
  (dispenses dispenser110 ingredient162)
  (clean shaker21)
  (clean shot246)
  (clean shot271)
  (clean shot449)
  (empty shaker21)
  (empty shot246)
  (empty shot271)
  (empty shot449)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker21 l0)
  (shaker-level shaker21 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient162)
  (cocktail-part2 cocktail1 ingredient150)
)
 (:goal
  (and
      (contains shot246 cocktail1)
      (contains shot271 ingredient162)
)))
