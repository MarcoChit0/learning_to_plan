(define (problem prob)
 (:domain barman)
 (:objects 
      shaker492 - shaker
      left right - hand
      shot174 - shot
      ingredient225 ingredient283 ingredient478 ingredient428 - ingredient
      cocktail370 - cocktail
      dispenser420 dispenser406 dispenser209 dispenser126 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker492)
  (ontable shot174)
  (dispenses dispenser420 ingredient225)
  (dispenses dispenser406 ingredient283)
  (dispenses dispenser209 ingredient478)
  (dispenses dispenser126 ingredient428)
  (clean shaker492)
  (clean shot174)
  (empty shaker492)
  (empty shot174)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker492 l0)
  (shaker-level shaker492 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail370 ingredient428)
  (cocktail-part2 cocktail370 ingredient225)
)
 (:goal
  (and
      (contains shot174 cocktail370)
)))
