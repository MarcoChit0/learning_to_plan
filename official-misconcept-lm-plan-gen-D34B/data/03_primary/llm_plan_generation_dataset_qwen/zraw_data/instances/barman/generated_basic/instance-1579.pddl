(define (problem prob)
 (:domain barman)
 (:objects 
      shaker86 - shaker
      left right - hand
      shot494 shot323 shot291 shot337 - shot
      ingredient413 ingredient442 - ingredient
      cocktail295 - cocktail
      dispenser325 dispenser191 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker86)
  (ontable shot494)
  (ontable shot323)
  (ontable shot291)
  (ontable shot337)
  (dispenses dispenser325 ingredient413)
  (dispenses dispenser191 ingredient442)
  (clean shaker86)
  (clean shot494)
  (clean shot323)
  (clean shot291)
  (clean shot337)
  (empty shaker86)
  (empty shot494)
  (empty shot323)
  (empty shot291)
  (empty shot337)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker86 l0)
  (shaker-level shaker86 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail295 ingredient413)
  (cocktail-part2 cocktail295 ingredient442)
)
 (:goal
  (and
      (contains shot494 cocktail295)
      (contains shot323 ingredient442)
      (contains shot291 cocktail295)
)))
