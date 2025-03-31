(define (problem prob)
 (:domain barman)
 (:objects 
      shaker326 - shaker
      left right - hand
      shot480 - shot
      ingredient12 ingredient471 ingredient208 ingredient361 - ingredient
      cocktail72 - cocktail
      dispenser87 dispenser474 dispenser447 dispenser373 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker326)
  (ontable shot480)
  (dispenses dispenser87 ingredient12)
  (dispenses dispenser474 ingredient471)
  (dispenses dispenser447 ingredient208)
  (dispenses dispenser373 ingredient361)
  (clean shaker326)
  (clean shot480)
  (empty shaker326)
  (empty shot480)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker326 l0)
  (shaker-level shaker326 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail72 ingredient471)
  (cocktail-part2 cocktail72 ingredient208)
)
 (:goal
  (and
      (contains shot480 cocktail72)
)))
