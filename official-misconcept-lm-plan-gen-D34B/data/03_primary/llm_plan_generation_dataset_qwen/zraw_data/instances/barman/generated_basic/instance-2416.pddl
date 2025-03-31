(define (problem prob)
 (:domain barman)
 (:objects 
      shaker210 - shaker
      left right - hand
      shot299 - shot
      ingredient4 ingredient198 - ingredient
      cocktail449 - cocktail
      dispenser91 dispenser176 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker210)
  (ontable shot299)
  (dispenses dispenser91 ingredient4)
  (dispenses dispenser176 ingredient198)
  (clean shaker210)
  (clean shot299)
  (empty shaker210)
  (empty shot299)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker210 l0)
  (shaker-level shaker210 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail449 ingredient4)
  (cocktail-part2 cocktail449 ingredient198)
)
 (:goal
  (and
      (contains shot299 cocktail449)
)))
