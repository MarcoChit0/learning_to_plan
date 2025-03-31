(define (problem prob)
 (:domain barman)
 (:objects 
      shaker124 - shaker
      left right - hand
      shot158 - shot
      ingredient109 ingredient226 ingredient7 - ingredient
      cocktail1 - cocktail
      dispenser265 dispenser12 dispenser237 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker124)
  (ontable shot158)
  (dispenses dispenser265 ingredient109)
  (dispenses dispenser12 ingredient226)
  (dispenses dispenser237 ingredient7)
  (clean shaker124)
  (clean shot158)
  (empty shaker124)
  (empty shot158)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker124 l0)
  (shaker-level shaker124 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient109)
  (cocktail-part2 cocktail1 ingredient226)
)
 (:goal
  (and
      (contains shot158 cocktail1)
)))
