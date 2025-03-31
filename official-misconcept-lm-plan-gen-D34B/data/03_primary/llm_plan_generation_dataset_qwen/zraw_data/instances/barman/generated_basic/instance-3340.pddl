(define (problem prob)
 (:domain barman)
 (:objects 
      shaker185 - shaker
      left right - hand
      shot237 shot337 - shot
      ingredient201 ingredient182 - ingredient
      cocktail156 - cocktail
      dispenser422 dispenser492 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker185)
  (ontable shot237)
  (ontable shot337)
  (dispenses dispenser422 ingredient201)
  (dispenses dispenser492 ingredient182)
  (clean shaker185)
  (clean shot237)
  (clean shot337)
  (empty shaker185)
  (empty shot237)
  (empty shot337)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker185 l0)
  (shaker-level shaker185 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail156 ingredient182)
  (cocktail-part2 cocktail156 ingredient201)
)
 (:goal
  (and
      (contains shot237 cocktail156)
)))
