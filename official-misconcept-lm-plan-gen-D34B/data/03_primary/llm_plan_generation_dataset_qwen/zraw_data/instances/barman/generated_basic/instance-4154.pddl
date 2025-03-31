(define (problem prob)
 (:domain barman)
 (:objects 
      shaker466 - shaker
      left right - hand
      shot66 - shot
      ingredient108 ingredient332 - ingredient
      cocktail212 - cocktail
      dispenser30 dispenser111 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker466)
  (ontable shot66)
  (dispenses dispenser30 ingredient108)
  (dispenses dispenser111 ingredient332)
  (clean shaker466)
  (clean shot66)
  (empty shaker466)
  (empty shot66)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker466 l0)
  (shaker-level shaker466 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail212 ingredient108)
  (cocktail-part2 cocktail212 ingredient332)
)
 (:goal
  (and
      (contains shot66 cocktail212)
)))
