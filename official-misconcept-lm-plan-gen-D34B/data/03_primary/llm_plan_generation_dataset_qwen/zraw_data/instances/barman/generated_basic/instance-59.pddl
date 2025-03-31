(define (problem prob)
 (:domain barman)
 (:objects 
      shaker330 - shaker
      left right - hand
      shot241 shot452 - shot
      ingredient416 ingredient158 ingredient283 ingredient205 - ingredient
      cocktail1 - cocktail
      dispenser65 dispenser68 dispenser125 dispenser462 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker330)
  (ontable shot241)
  (ontable shot452)
  (dispenses dispenser65 ingredient416)
  (dispenses dispenser68 ingredient158)
  (dispenses dispenser125 ingredient283)
  (dispenses dispenser462 ingredient205)
  (clean shaker330)
  (clean shot241)
  (clean shot452)
  (empty shaker330)
  (empty shot241)
  (empty shot452)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker330 l0)
  (shaker-level shaker330 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient416)
  (cocktail-part2 cocktail1 ingredient283)
)
 (:goal
  (and
      (contains shot241 cocktail1)
)))
