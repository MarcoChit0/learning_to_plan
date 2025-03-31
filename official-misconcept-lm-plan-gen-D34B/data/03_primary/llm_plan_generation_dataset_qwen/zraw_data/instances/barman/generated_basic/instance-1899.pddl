(define (problem prob)
 (:domain barman)
 (:objects 
      shaker316 - shaker
      left right - hand
      shot468 shot397 shot178 shot405 - shot
      ingredient267 ingredient54 - ingredient
      cocktail406 - cocktail
      dispenser32 dispenser270 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker316)
  (ontable shot468)
  (ontable shot397)
  (ontable shot178)
  (ontable shot405)
  (dispenses dispenser32 ingredient267)
  (dispenses dispenser270 ingredient54)
  (clean shaker316)
  (clean shot468)
  (clean shot397)
  (clean shot178)
  (clean shot405)
  (empty shaker316)
  (empty shot468)
  (empty shot397)
  (empty shot178)
  (empty shot405)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker316 l0)
  (shaker-level shaker316 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail406 ingredient54)
  (cocktail-part2 cocktail406 ingredient267)
)
 (:goal
  (and
      (contains shot468 cocktail406)
      (contains shot397 cocktail406)
      (contains shot178 ingredient267)
)))
