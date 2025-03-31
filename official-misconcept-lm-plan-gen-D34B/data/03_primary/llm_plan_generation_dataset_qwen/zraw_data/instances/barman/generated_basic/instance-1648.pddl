(define (problem prob)
 (:domain barman)
 (:objects 
      shaker126 - shaker
      left right - hand
      shot294 shot449 shot104 - shot
      ingredient398 ingredient459 ingredient360 ingredient249 - ingredient
      cocktail29 - cocktail
      dispenser123 dispenser487 dispenser312 dispenser404 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker126)
  (ontable shot294)
  (ontable shot449)
  (ontable shot104)
  (dispenses dispenser123 ingredient398)
  (dispenses dispenser487 ingredient459)
  (dispenses dispenser312 ingredient360)
  (dispenses dispenser404 ingredient249)
  (clean shaker126)
  (clean shot294)
  (clean shot449)
  (clean shot104)
  (empty shaker126)
  (empty shot294)
  (empty shot449)
  (empty shot104)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker126 l0)
  (shaker-level shaker126 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail29 ingredient360)
  (cocktail-part2 cocktail29 ingredient249)
)
 (:goal
  (and
      (contains shot294 cocktail29)
      (contains shot449 cocktail29)
)))
