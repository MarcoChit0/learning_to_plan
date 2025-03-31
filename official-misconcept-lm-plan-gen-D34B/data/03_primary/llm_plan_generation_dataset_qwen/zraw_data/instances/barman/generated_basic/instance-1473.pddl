(define (problem prob)
 (:domain barman)
 (:objects 
      shaker421 - shaker
      left right - hand
      shot496 shot350 shot258 - shot
      ingredient27 ingredient60 - ingredient
      cocktail223 - cocktail
      dispenser368 dispenser83 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker421)
  (ontable shot496)
  (ontable shot350)
  (ontable shot258)
  (dispenses dispenser368 ingredient27)
  (dispenses dispenser83 ingredient60)
  (clean shaker421)
  (clean shot496)
  (clean shot350)
  (clean shot258)
  (empty shaker421)
  (empty shot496)
  (empty shot350)
  (empty shot258)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker421 l0)
  (shaker-level shaker421 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail223 ingredient27)
  (cocktail-part2 cocktail223 ingredient60)
)
 (:goal
  (and
      (contains shot496 cocktail223)
      (contains shot350 cocktail223)
)))
