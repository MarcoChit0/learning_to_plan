(define (problem prob)
 (:domain barman)
 (:objects 
      shaker265 - shaker
      left right - hand
      shot424 - shot
      ingredient114 ingredient134 ingredient21 - ingredient
      cocktail1 - cocktail
      dispenser275 dispenser496 dispenser324 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker265)
  (ontable shot424)
  (dispenses dispenser275 ingredient114)
  (dispenses dispenser496 ingredient134)
  (dispenses dispenser324 ingredient21)
  (clean shaker265)
  (clean shot424)
  (empty shaker265)
  (empty shot424)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker265 l0)
  (shaker-level shaker265 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient114)
  (cocktail-part2 cocktail1 ingredient21)
)
 (:goal
  (and
      (contains shot424 cocktail1)
)))
