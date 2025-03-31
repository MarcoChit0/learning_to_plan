(define (problem prob)
 (:domain barman)
 (:objects 
      shaker460 - shaker
      left right - hand
      shot258 shot337 - shot
      ingredient489 ingredient63 ingredient300 - ingredient
      cocktail495 - cocktail
      dispenser183 dispenser118 dispenser270 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker460)
  (ontable shot258)
  (ontable shot337)
  (dispenses dispenser183 ingredient489)
  (dispenses dispenser118 ingredient63)
  (dispenses dispenser270 ingredient300)
  (clean shaker460)
  (clean shot258)
  (clean shot337)
  (empty shaker460)
  (empty shot258)
  (empty shot337)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker460 l0)
  (shaker-level shaker460 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail495 ingredient300)
  (cocktail-part2 cocktail495 ingredient63)
)
 (:goal
  (and
      (contains shot258 cocktail495)
)))
