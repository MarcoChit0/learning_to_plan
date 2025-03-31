(define (problem prob)
 (:domain barman)
 (:objects 
      shaker408 - shaker
      left right - hand
      shot495 shot167 shot152 - shot
      ingredient202 ingredient32 ingredient130 ingredient483 - ingredient
      cocktail1 - cocktail
      dispenser351 dispenser435 dispenser108 dispenser445 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker408)
  (ontable shot495)
  (ontable shot167)
  (ontable shot152)
  (dispenses dispenser351 ingredient202)
  (dispenses dispenser435 ingredient32)
  (dispenses dispenser108 ingredient130)
  (dispenses dispenser445 ingredient483)
  (clean shaker408)
  (clean shot495)
  (clean shot167)
  (clean shot152)
  (empty shaker408)
  (empty shot495)
  (empty shot167)
  (empty shot152)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker408 l0)
  (shaker-level shaker408 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient32)
  (cocktail-part2 cocktail1 ingredient202)
)
 (:goal
  (and
      (contains shot495 cocktail1)
      (contains shot167 cocktail1)
)))
