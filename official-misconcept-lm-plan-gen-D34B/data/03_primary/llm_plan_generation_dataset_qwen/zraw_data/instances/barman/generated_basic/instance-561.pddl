(define (problem prob)
 (:domain barman)
 (:objects 
      shaker207 - shaker
      left right - hand
      shot147 shot152 - shot
      ingredient7 ingredient429 ingredient302 - ingredient
      cocktail1 - cocktail
      dispenser48 dispenser158 dispenser113 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker207)
  (ontable shot147)
  (ontable shot152)
  (dispenses dispenser48 ingredient7)
  (dispenses dispenser158 ingredient429)
  (dispenses dispenser113 ingredient302)
  (clean shaker207)
  (clean shot147)
  (clean shot152)
  (empty shaker207)
  (empty shot147)
  (empty shot152)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker207 l0)
  (shaker-level shaker207 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient302)
  (cocktail-part2 cocktail1 ingredient7)
)
 (:goal
  (and
      (contains shot147 cocktail1)
)))
