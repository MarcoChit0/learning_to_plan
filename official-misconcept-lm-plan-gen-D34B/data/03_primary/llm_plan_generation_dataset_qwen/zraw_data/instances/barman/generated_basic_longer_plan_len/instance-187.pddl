(define (problem prob)
 (:domain barman)
 (:objects 
      shaker185 - shaker
      left right - hand
      shot55 shot65 shot1 - shot
      ingredient466 ingredient409 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser447 dispenser299 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker185)
  (ontable shot55)
  (ontable shot65)
  (ontable shot1)
  (dispenses dispenser447 ingredient466)
  (dispenses dispenser299 ingredient409)
  (clean shaker185)
  (clean shot55)
  (clean shot65)
  (clean shot1)
  (empty shaker185)
  (empty shot55)
  (empty shot65)
  (empty shot1)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker185 l0)
  (shaker-level shaker185 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient409)
  (cocktail-part2 cocktail1 ingredient466)
  (cocktail-part1 cocktail2 ingredient409)
  (cocktail-part2 cocktail2 ingredient466)
)
 (:goal
  (and
      (contains shot55 cocktail2)
      (contains shot65 cocktail1)
)))
