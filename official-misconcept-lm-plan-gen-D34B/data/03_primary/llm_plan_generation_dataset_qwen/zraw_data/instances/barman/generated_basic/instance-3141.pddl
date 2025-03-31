(define (problem prob)
 (:domain barman)
 (:objects 
      shaker403 - shaker
      left right - hand
      shot355 shot167 shot366 - shot
      ingredient380 ingredient195 - ingredient
      cocktail205 - cocktail
      dispenser77 dispenser320 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker403)
  (ontable shot355)
  (ontable shot167)
  (ontable shot366)
  (dispenses dispenser77 ingredient380)
  (dispenses dispenser320 ingredient195)
  (clean shaker403)
  (clean shot355)
  (clean shot167)
  (clean shot366)
  (empty shaker403)
  (empty shot355)
  (empty shot167)
  (empty shot366)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker403 l0)
  (shaker-level shaker403 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail205 ingredient380)
  (cocktail-part2 cocktail205 ingredient195)
)
 (:goal
  (and
      (contains shot355 cocktail205)
      (contains shot167 cocktail205)
)))
