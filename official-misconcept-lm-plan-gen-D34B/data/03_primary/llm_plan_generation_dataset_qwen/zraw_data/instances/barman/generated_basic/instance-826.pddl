(define (problem prob)
 (:domain barman)
 (:objects 
      shaker413 - shaker
      left right - hand
      shot129 shot166 shot93 - shot
      ingredient415 ingredient184 ingredient310 ingredient7 - ingredient
      cocktail1 - cocktail
      dispenser300 dispenser91 dispenser217 dispenser412 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker413)
  (ontable shot129)
  (ontable shot166)
  (ontable shot93)
  (dispenses dispenser300 ingredient415)
  (dispenses dispenser91 ingredient184)
  (dispenses dispenser217 ingredient310)
  (dispenses dispenser412 ingredient7)
  (clean shaker413)
  (clean shot129)
  (clean shot166)
  (clean shot93)
  (empty shaker413)
  (empty shot129)
  (empty shot166)
  (empty shot93)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker413 l0)
  (shaker-level shaker413 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient310)
  (cocktail-part2 cocktail1 ingredient7)
)
 (:goal
  (and
      (contains shot129 cocktail1)
      (contains shot166 cocktail1)
)))
