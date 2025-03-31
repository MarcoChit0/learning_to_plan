(define (problem prob)
 (:domain barman)
 (:objects 
      shaker220 - shaker
      left right - hand
      shot152 shot285 - shot
      ingredient463 ingredient451 ingredient469 ingredient208 - ingredient
      cocktail470 - cocktail
      dispenser239 dispenser283 dispenser344 dispenser498 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker220)
  (ontable shot152)
  (ontable shot285)
  (dispenses dispenser239 ingredient463)
  (dispenses dispenser283 ingredient451)
  (dispenses dispenser344 ingredient469)
  (dispenses dispenser498 ingredient208)
  (clean shaker220)
  (clean shot152)
  (clean shot285)
  (empty shaker220)
  (empty shot152)
  (empty shot285)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker220 l0)
  (shaker-level shaker220 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail470 ingredient469)
  (cocktail-part2 cocktail470 ingredient208)
)
 (:goal
  (and
      (contains shot152 cocktail470)
)))
