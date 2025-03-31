(define (problem prob)
 (:domain barman)
 (:objects 
      shaker428 - shaker
      left right - hand
      shot425 - shot
      ingredient431 ingredient52 - ingredient
      cocktail1 - cocktail
      dispenser163 dispenser193 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker428)
  (ontable shot425)
  (dispenses dispenser163 ingredient431)
  (dispenses dispenser193 ingredient52)
  (clean shaker428)
  (clean shot425)
  (empty shaker428)
  (empty shot425)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker428 l0)
  (shaker-level shaker428 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient431)
  (cocktail-part2 cocktail1 ingredient52)
)
 (:goal
  (and
      (contains shot425 cocktail1)
)))
