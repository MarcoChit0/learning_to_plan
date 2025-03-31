(define (problem prob)
 (:domain barman)
 (:objects 
      shaker337 - shaker
      left right - hand
      shot16 - shot
      ingredient37 ingredient192 ingredient397 - ingredient
      cocktail202 - cocktail
      dispenser409 dispenser2 dispenser176 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker337)
  (ontable shot16)
  (dispenses dispenser409 ingredient37)
  (dispenses dispenser2 ingredient192)
  (dispenses dispenser176 ingredient397)
  (clean shaker337)
  (clean shot16)
  (empty shaker337)
  (empty shot16)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker337 l0)
  (shaker-level shaker337 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail202 ingredient37)
  (cocktail-part2 cocktail202 ingredient192)
)
 (:goal
  (and
      (contains shot16 cocktail202)
)))
