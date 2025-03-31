(define (problem prob)
 (:domain barman)
 (:objects 
      shaker149 - shaker
      left right - hand
      shot322 shot333 - shot
      ingredient183 ingredient305 - ingredient
      cocktail397 - cocktail
      dispenser104 dispenser381 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker149)
  (ontable shot322)
  (ontable shot333)
  (dispenses dispenser104 ingredient183)
  (dispenses dispenser381 ingredient305)
  (clean shaker149)
  (clean shot322)
  (clean shot333)
  (empty shaker149)
  (empty shot322)
  (empty shot333)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker149 l0)
  (shaker-level shaker149 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail397 ingredient183)
  (cocktail-part2 cocktail397 ingredient305)
)
 (:goal
  (and
      (contains shot322 cocktail397)
)))
