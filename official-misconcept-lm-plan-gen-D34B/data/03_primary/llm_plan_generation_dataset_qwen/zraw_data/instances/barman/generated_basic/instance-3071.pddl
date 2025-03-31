(define (problem prob)
 (:domain barman)
 (:objects 
      shaker59 - shaker
      left right - hand
      shot230 shot143 - shot
      ingredient302 ingredient252 - ingredient
      cocktail299 - cocktail
      dispenser365 dispenser43 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker59)
  (ontable shot230)
  (ontable shot143)
  (dispenses dispenser365 ingredient302)
  (dispenses dispenser43 ingredient252)
  (clean shaker59)
  (clean shot230)
  (clean shot143)
  (empty shaker59)
  (empty shot230)
  (empty shot143)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker59 l0)
  (shaker-level shaker59 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail299 ingredient252)
  (cocktail-part2 cocktail299 ingredient302)
)
 (:goal
  (and
      (contains shot230 cocktail299)
)))
