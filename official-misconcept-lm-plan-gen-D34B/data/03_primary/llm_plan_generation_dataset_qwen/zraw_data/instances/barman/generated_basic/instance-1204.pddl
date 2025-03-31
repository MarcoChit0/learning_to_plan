(define (problem prob)
 (:domain barman)
 (:objects 
      shaker352 - shaker
      left right - hand
      shot431 shot172 - shot
      ingredient31 ingredient477 - ingredient
      cocktail326 - cocktail
      dispenser122 dispenser254 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker352)
  (ontable shot431)
  (ontable shot172)
  (dispenses dispenser122 ingredient31)
  (dispenses dispenser254 ingredient477)
  (clean shaker352)
  (clean shot431)
  (clean shot172)
  (empty shaker352)
  (empty shot431)
  (empty shot172)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker352 l0)
  (shaker-level shaker352 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail326 ingredient31)
  (cocktail-part2 cocktail326 ingredient477)
)
 (:goal
  (and
      (contains shot431 cocktail326)
)))
