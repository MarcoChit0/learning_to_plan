(define (problem prob)
 (:domain barman)
 (:objects 
      shaker251 - shaker
      left right - hand
      shot249 - shot
      ingredient253 ingredient324 ingredient45 ingredient320 - ingredient
      cocktail432 - cocktail
      dispenser197 dispenser321 dispenser381 dispenser192 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker251)
  (ontable shot249)
  (dispenses dispenser197 ingredient253)
  (dispenses dispenser321 ingredient324)
  (dispenses dispenser381 ingredient45)
  (dispenses dispenser192 ingredient320)
  (clean shaker251)
  (clean shot249)
  (empty shaker251)
  (empty shot249)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker251 l0)
  (shaker-level shaker251 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail432 ingredient45)
  (cocktail-part2 cocktail432 ingredient253)
)
 (:goal
  (and
      (contains shot249 cocktail432)
)))
