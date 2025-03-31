(define (problem prob)
 (:domain barman)
 (:objects 
      shaker479 - shaker
      left right - hand
      shot321 shot21 shot135 - shot
      ingredient492 ingredient361 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser475 dispenser200 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker479)
  (ontable shot321)
  (ontable shot21)
  (ontable shot135)
  (dispenses dispenser475 ingredient492)
  (dispenses dispenser200 ingredient361)
  (clean shaker479)
  (clean shot321)
  (clean shot21)
  (clean shot135)
  (empty shaker479)
  (empty shot321)
  (empty shot21)
  (empty shot135)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker479 l0)
  (shaker-level shaker479 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient492)
  (cocktail-part2 cocktail1 ingredient361)
  (cocktail-part1 cocktail2 ingredient492)
  (cocktail-part2 cocktail2 ingredient361)
)
 (:goal
  (and
      (contains shot321 cocktail1)
      (contains shot21 cocktail2)
)))
