(define (problem prob)
 (:domain barman)
 (:objects 
      shaker157 - shaker
      left right - hand
      shot108 - shot
      ingredient446 ingredient369 ingredient454 - ingredient
      cocktail8 - cocktail
      dispenser302 dispenser191 dispenser499 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker157)
  (ontable shot108)
  (dispenses dispenser302 ingredient446)
  (dispenses dispenser191 ingredient369)
  (dispenses dispenser499 ingredient454)
  (clean shaker157)
  (clean shot108)
  (empty shaker157)
  (empty shot108)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker157 l0)
  (shaker-level shaker157 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail8 ingredient446)
  (cocktail-part2 cocktail8 ingredient454)
)
 (:goal
  (and
      (contains shot108 cocktail8)
)))
