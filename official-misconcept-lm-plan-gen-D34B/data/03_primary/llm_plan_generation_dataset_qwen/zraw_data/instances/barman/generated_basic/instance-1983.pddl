(define (problem prob)
 (:domain barman)
 (:objects 
      shaker305 - shaker
      left right - hand
      shot468 shot143 - shot
      ingredient489 ingredient492 ingredient384 - ingredient
      cocktail412 - cocktail
      dispenser333 dispenser278 dispenser191 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker305)
  (ontable shot468)
  (ontable shot143)
  (dispenses dispenser333 ingredient489)
  (dispenses dispenser278 ingredient492)
  (dispenses dispenser191 ingredient384)
  (clean shaker305)
  (clean shot468)
  (clean shot143)
  (empty shaker305)
  (empty shot468)
  (empty shot143)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker305 l0)
  (shaker-level shaker305 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail412 ingredient489)
  (cocktail-part2 cocktail412 ingredient384)
)
 (:goal
  (and
      (contains shot468 cocktail412)
)))
