(define (problem prob)
 (:domain barman)
 (:objects 
      shaker356 - shaker
      left right - hand
      shot49 shot16 - shot
      ingredient367 ingredient477 ingredient184 - ingredient
      cocktail1 - cocktail
      dispenser13 dispenser231 dispenser394 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker356)
  (ontable shot49)
  (ontable shot16)
  (dispenses dispenser13 ingredient367)
  (dispenses dispenser231 ingredient477)
  (dispenses dispenser394 ingredient184)
  (clean shaker356)
  (clean shot49)
  (clean shot16)
  (empty shaker356)
  (empty shot49)
  (empty shot16)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker356 l0)
  (shaker-level shaker356 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient184)
  (cocktail-part2 cocktail1 ingredient477)
)
 (:goal
  (and
      (contains shot49 cocktail1)
)))
