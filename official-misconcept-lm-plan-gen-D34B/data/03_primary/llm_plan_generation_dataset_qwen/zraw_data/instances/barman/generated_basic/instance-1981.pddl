(define (problem prob)
 (:domain barman)
 (:objects 
      shaker9 - shaker
      left right - hand
      shot104 shot339 - shot
      ingredient155 ingredient209 ingredient169 - ingredient
      cocktail435 - cocktail
      dispenser261 dispenser395 dispenser336 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker9)
  (ontable shot104)
  (ontable shot339)
  (dispenses dispenser261 ingredient155)
  (dispenses dispenser395 ingredient209)
  (dispenses dispenser336 ingredient169)
  (clean shaker9)
  (clean shot104)
  (clean shot339)
  (empty shaker9)
  (empty shot104)
  (empty shot339)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker9 l0)
  (shaker-level shaker9 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail435 ingredient209)
  (cocktail-part2 cocktail435 ingredient155)
)
 (:goal
  (and
      (contains shot104 cocktail435)
)))
