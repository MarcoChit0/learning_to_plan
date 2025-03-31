(define (problem prob)
 (:domain barman)
 (:objects 
      shaker400 - shaker
      left right - hand
      shot446 shot37 shot343 - shot
      ingredient153 ingredient302 ingredient205 ingredient126 - ingredient
      cocktail1 - cocktail
      dispenser69 dispenser260 dispenser148 dispenser419 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker400)
  (ontable shot446)
  (ontable shot37)
  (ontable shot343)
  (dispenses dispenser69 ingredient153)
  (dispenses dispenser260 ingredient302)
  (dispenses dispenser148 ingredient205)
  (dispenses dispenser419 ingredient126)
  (clean shaker400)
  (clean shot446)
  (clean shot37)
  (clean shot343)
  (empty shaker400)
  (empty shot446)
  (empty shot37)
  (empty shot343)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker400 l0)
  (shaker-level shaker400 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient126)
  (cocktail-part2 cocktail1 ingredient153)
)
 (:goal
  (and
      (contains shot446 cocktail1)
      (contains shot37 ingredient205)
)))
