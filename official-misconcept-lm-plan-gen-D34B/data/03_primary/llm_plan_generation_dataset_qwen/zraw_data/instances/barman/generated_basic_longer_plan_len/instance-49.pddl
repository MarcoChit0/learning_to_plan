(define (problem prob)
 (:domain barman)
 (:objects 
      shaker122 - shaker
      left right - hand
      shot420 shot91 shot194 - shot
      ingredient436 ingredient418 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser188 dispenser344 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker122)
  (ontable shot420)
  (ontable shot91)
  (ontable shot194)
  (dispenses dispenser188 ingredient436)
  (dispenses dispenser344 ingredient418)
  (clean shaker122)
  (clean shot420)
  (clean shot91)
  (clean shot194)
  (empty shaker122)
  (empty shot420)
  (empty shot91)
  (empty shot194)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker122 l0)
  (shaker-level shaker122 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient418)
  (cocktail-part2 cocktail1 ingredient436)
  (cocktail-part1 cocktail2 ingredient436)
  (cocktail-part2 cocktail2 ingredient418)
)
 (:goal
  (and
      (contains shot420 cocktail2)
      (contains shot91 cocktail1)
)))
