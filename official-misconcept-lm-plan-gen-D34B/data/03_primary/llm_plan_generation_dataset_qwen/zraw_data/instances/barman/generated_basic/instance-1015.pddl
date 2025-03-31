(define (problem prob)
 (:domain barman)
 (:objects 
      shaker125 - shaker
      left right - hand
      shot342 - shot
      ingredient228 ingredient57 ingredient273 - ingredient
      cocktail1 - cocktail
      dispenser206 dispenser146 dispenser113 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker125)
  (ontable shot342)
  (dispenses dispenser206 ingredient228)
  (dispenses dispenser146 ingredient57)
  (dispenses dispenser113 ingredient273)
  (clean shaker125)
  (clean shot342)
  (empty shaker125)
  (empty shot342)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker125 l0)
  (shaker-level shaker125 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient57)
  (cocktail-part2 cocktail1 ingredient228)
)
 (:goal
  (and
      (contains shot342 cocktail1)
)))
