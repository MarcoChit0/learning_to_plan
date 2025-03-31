(define (problem prob)
 (:domain barman)
 (:objects 
      shaker129 - shaker
      left right - hand
      shot391 shot116 - shot
      ingredient251 ingredient228 - ingredient
      cocktail419 - cocktail
      dispenser443 dispenser321 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker129)
  (ontable shot391)
  (ontable shot116)
  (dispenses dispenser443 ingredient251)
  (dispenses dispenser321 ingredient228)
  (clean shaker129)
  (clean shot391)
  (clean shot116)
  (empty shaker129)
  (empty shot391)
  (empty shot116)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker129 l0)
  (shaker-level shaker129 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail419 ingredient251)
  (cocktail-part2 cocktail419 ingredient228)
)
 (:goal
  (and
      (contains shot391 cocktail419)
)))
