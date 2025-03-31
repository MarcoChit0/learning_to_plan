(define (problem prob)
 (:domain barman)
 (:objects 
      shaker101 - shaker
      left right - hand
      shot422 shot167 shot75 - shot
      ingredient87 ingredient207 ingredient219 - ingredient
      cocktail34 - cocktail
      dispenser161 dispenser483 dispenser489 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker101)
  (ontable shot422)
  (ontable shot167)
  (ontable shot75)
  (dispenses dispenser161 ingredient87)
  (dispenses dispenser483 ingredient207)
  (dispenses dispenser489 ingredient219)
  (clean shaker101)
  (clean shot422)
  (clean shot167)
  (clean shot75)
  (empty shaker101)
  (empty shot422)
  (empty shot167)
  (empty shot75)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker101 l0)
  (shaker-level shaker101 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail34 ingredient87)
  (cocktail-part2 cocktail34 ingredient207)
)
 (:goal
  (and
      (contains shot422 cocktail34)
      (contains shot167 ingredient219)
)))
