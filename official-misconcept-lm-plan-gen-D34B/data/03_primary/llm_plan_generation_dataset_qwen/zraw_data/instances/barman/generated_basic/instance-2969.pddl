(define (problem prob)
 (:domain barman)
 (:objects 
      shaker278 - shaker
      left right - hand
      shot243 - shot
      ingredient170 ingredient492 - ingredient
      cocktail140 - cocktail
      dispenser442 dispenser190 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker278)
  (ontable shot243)
  (dispenses dispenser442 ingredient170)
  (dispenses dispenser190 ingredient492)
  (clean shaker278)
  (clean shot243)
  (empty shaker278)
  (empty shot243)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker278 l0)
  (shaker-level shaker278 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail140 ingredient170)
  (cocktail-part2 cocktail140 ingredient492)
)
 (:goal
  (and
      (contains shot243 cocktail140)
)))
