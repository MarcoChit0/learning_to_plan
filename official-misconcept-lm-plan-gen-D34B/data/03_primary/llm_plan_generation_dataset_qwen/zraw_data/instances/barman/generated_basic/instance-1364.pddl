(define (problem prob)
 (:domain barman)
 (:objects 
      shaker330 - shaker
      left right - hand
      shot66 - shot
      ingredient491 ingredient98 ingredient157 - ingredient
      cocktail140 - cocktail
      dispenser259 dispenser52 dispenser393 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker330)
  (ontable shot66)
  (dispenses dispenser259 ingredient491)
  (dispenses dispenser52 ingredient98)
  (dispenses dispenser393 ingredient157)
  (clean shaker330)
  (clean shot66)
  (empty shaker330)
  (empty shot66)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker330 l0)
  (shaker-level shaker330 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail140 ingredient157)
  (cocktail-part2 cocktail140 ingredient98)
)
 (:goal
  (and
      (contains shot66 cocktail140)
)))
