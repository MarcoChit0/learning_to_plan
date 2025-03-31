(define (problem prob)
 (:domain barman)
 (:objects 
      shaker236 - shaker
      left right - hand
      shot299 shot49 - shot
      ingredient417 ingredient432 ingredient13 ingredient4 - ingredient
      cocktail305 - cocktail
      dispenser461 dispenser271 dispenser238 dispenser187 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker236)
  (ontable shot299)
  (ontable shot49)
  (dispenses dispenser461 ingredient417)
  (dispenses dispenser271 ingredient432)
  (dispenses dispenser238 ingredient13)
  (dispenses dispenser187 ingredient4)
  (clean shaker236)
  (clean shot299)
  (clean shot49)
  (empty shaker236)
  (empty shot299)
  (empty shot49)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker236 l0)
  (shaker-level shaker236 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail305 ingredient432)
  (cocktail-part2 cocktail305 ingredient4)
)
 (:goal
  (and
      (contains shot299 cocktail305)
)))
