(define (problem prob)
 (:domain barman)
 (:objects 
      shaker183 - shaker
      left right - hand
      shot472 shot471 - shot
      ingredient39 ingredient393 ingredient95 - ingredient
      cocktail255 - cocktail
      dispenser488 dispenser57 dispenser160 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker183)
  (ontable shot472)
  (ontable shot471)
  (dispenses dispenser488 ingredient39)
  (dispenses dispenser57 ingredient393)
  (dispenses dispenser160 ingredient95)
  (clean shaker183)
  (clean shot472)
  (clean shot471)
  (empty shaker183)
  (empty shot472)
  (empty shot471)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker183 l0)
  (shaker-level shaker183 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail255 ingredient39)
  (cocktail-part2 cocktail255 ingredient393)
)
 (:goal
  (and
      (contains shot472 cocktail255)
)))
