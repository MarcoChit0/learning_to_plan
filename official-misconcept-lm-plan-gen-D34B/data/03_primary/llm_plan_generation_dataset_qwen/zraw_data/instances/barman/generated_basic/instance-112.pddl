(define (problem prob)
 (:domain barman)
 (:objects 
      shaker34 - shaker
      left right - hand
      shot449 shot430 shot447 - shot
      ingredient346 ingredient116 - ingredient
      cocktail1 - cocktail
      dispenser414 dispenser167 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker34)
  (ontable shot449)
  (ontable shot430)
  (ontable shot447)
  (dispenses dispenser414 ingredient346)
  (dispenses dispenser167 ingredient116)
  (clean shaker34)
  (clean shot449)
  (clean shot430)
  (clean shot447)
  (empty shaker34)
  (empty shot449)
  (empty shot430)
  (empty shot447)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker34 l0)
  (shaker-level shaker34 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient116)
  (cocktail-part2 cocktail1 ingredient346)
)
 (:goal
  (and
      (contains shot449 cocktail1)
      (contains shot430 cocktail1)
)))
