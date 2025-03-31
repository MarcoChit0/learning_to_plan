(define (problem prob)
 (:domain barman)
 (:objects 
      shaker86 - shaker
      left right - hand
      shot74 shot69 - shot
      ingredient194 ingredient90 ingredient492 ingredient158 - ingredient
      cocktail196 - cocktail
      dispenser328 dispenser101 dispenser473 dispenser306 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker86)
  (ontable shot74)
  (ontable shot69)
  (dispenses dispenser328 ingredient194)
  (dispenses dispenser101 ingredient90)
  (dispenses dispenser473 ingredient492)
  (dispenses dispenser306 ingredient158)
  (clean shaker86)
  (clean shot74)
  (clean shot69)
  (empty shaker86)
  (empty shot74)
  (empty shot69)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker86 l0)
  (shaker-level shaker86 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail196 ingredient158)
  (cocktail-part2 cocktail196 ingredient194)
)
 (:goal
  (and
      (contains shot74 cocktail196)
)))
