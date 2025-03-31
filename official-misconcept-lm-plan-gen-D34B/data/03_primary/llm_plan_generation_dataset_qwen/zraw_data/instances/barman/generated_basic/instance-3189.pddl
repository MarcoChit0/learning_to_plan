(define (problem prob)
 (:domain barman)
 (:objects 
      shaker408 - shaker
      left right - hand
      shot53 shot164 - shot
      ingredient98 ingredient133 ingredient202 - ingredient
      cocktail394 - cocktail
      dispenser446 dispenser339 dispenser273 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker408)
  (ontable shot53)
  (ontable shot164)
  (dispenses dispenser446 ingredient98)
  (dispenses dispenser339 ingredient133)
  (dispenses dispenser273 ingredient202)
  (clean shaker408)
  (clean shot53)
  (clean shot164)
  (empty shaker408)
  (empty shot53)
  (empty shot164)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker408 l0)
  (shaker-level shaker408 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail394 ingredient98)
  (cocktail-part2 cocktail394 ingredient133)
)
 (:goal
  (and
      (contains shot53 cocktail394)
)))
