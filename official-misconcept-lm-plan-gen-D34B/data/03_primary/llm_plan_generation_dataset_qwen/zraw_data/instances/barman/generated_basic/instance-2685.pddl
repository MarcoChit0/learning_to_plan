(define (problem prob)
 (:domain barman)
 (:objects 
      shaker494 - shaker
      left right - hand
      shot449 shot344 - shot
      ingredient14 ingredient97 ingredient58 - ingredient
      cocktail91 - cocktail
      dispenser109 dispenser202 dispenser344 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker494)
  (ontable shot449)
  (ontable shot344)
  (dispenses dispenser109 ingredient14)
  (dispenses dispenser202 ingredient97)
  (dispenses dispenser344 ingredient58)
  (clean shaker494)
  (clean shot449)
  (clean shot344)
  (empty shaker494)
  (empty shot449)
  (empty shot344)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker494 l0)
  (shaker-level shaker494 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail91 ingredient58)
  (cocktail-part2 cocktail91 ingredient14)
)
 (:goal
  (and
      (contains shot449 cocktail91)
)))
