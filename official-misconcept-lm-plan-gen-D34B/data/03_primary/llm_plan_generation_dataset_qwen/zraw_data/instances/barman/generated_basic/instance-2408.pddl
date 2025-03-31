(define (problem prob)
 (:domain barman)
 (:objects 
      shaker435 - shaker
      left right - hand
      shot52 - shot
      ingredient300 ingredient119 - ingredient
      cocktail468 - cocktail
      dispenser434 dispenser314 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker435)
  (ontable shot52)
  (dispenses dispenser434 ingredient300)
  (dispenses dispenser314 ingredient119)
  (clean shaker435)
  (clean shot52)
  (empty shaker435)
  (empty shot52)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker435 l0)
  (shaker-level shaker435 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail468 ingredient119)
  (cocktail-part2 cocktail468 ingredient300)
)
 (:goal
  (and
      (contains shot52 cocktail468)
)))
