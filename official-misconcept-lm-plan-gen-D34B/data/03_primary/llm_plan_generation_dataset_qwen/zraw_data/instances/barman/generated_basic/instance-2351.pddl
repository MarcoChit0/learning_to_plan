(define (problem prob)
 (:domain barman)
 (:objects 
      shaker15 - shaker
      left right - hand
      shot207 shot22 - shot
      ingredient284 ingredient156 ingredient125 ingredient336 - ingredient
      cocktail44 - cocktail
      dispenser70 dispenser383 dispenser0 dispenser142 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker15)
  (ontable shot207)
  (ontable shot22)
  (dispenses dispenser70 ingredient284)
  (dispenses dispenser383 ingredient156)
  (dispenses dispenser0 ingredient125)
  (dispenses dispenser142 ingredient336)
  (clean shaker15)
  (clean shot207)
  (clean shot22)
  (empty shaker15)
  (empty shot207)
  (empty shot22)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker15 l0)
  (shaker-level shaker15 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail44 ingredient156)
  (cocktail-part2 cocktail44 ingredient284)
)
 (:goal
  (and
      (contains shot207 cocktail44)
)))
