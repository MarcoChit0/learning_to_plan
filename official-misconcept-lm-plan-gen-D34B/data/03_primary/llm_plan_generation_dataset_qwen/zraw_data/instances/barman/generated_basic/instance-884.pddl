(define (problem prob)
 (:domain barman)
 (:objects 
      shaker207 - shaker
      left right - hand
      shot372 - shot
      ingredient185 ingredient499 - ingredient
      cocktail1 - cocktail
      dispenser451 dispenser161 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker207)
  (ontable shot372)
  (dispenses dispenser451 ingredient185)
  (dispenses dispenser161 ingredient499)
  (clean shaker207)
  (clean shot372)
  (empty shaker207)
  (empty shot372)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker207 l0)
  (shaker-level shaker207 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient499)
  (cocktail-part2 cocktail1 ingredient185)
)
 (:goal
  (and
      (contains shot372 cocktail1)
)))
