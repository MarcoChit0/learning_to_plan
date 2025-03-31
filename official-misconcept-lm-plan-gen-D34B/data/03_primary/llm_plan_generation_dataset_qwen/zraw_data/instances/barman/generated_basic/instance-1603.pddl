(define (problem prob)
 (:domain barman)
 (:objects 
      shaker286 - shaker
      left right - hand
      shot283 shot112 - shot
      ingredient271 ingredient88 ingredient107 - ingredient
      cocktail493 - cocktail
      dispenser291 dispenser168 dispenser214 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker286)
  (ontable shot283)
  (ontable shot112)
  (dispenses dispenser291 ingredient271)
  (dispenses dispenser168 ingredient88)
  (dispenses dispenser214 ingredient107)
  (clean shaker286)
  (clean shot283)
  (clean shot112)
  (empty shaker286)
  (empty shot283)
  (empty shot112)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker286 l0)
  (shaker-level shaker286 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail493 ingredient88)
  (cocktail-part2 cocktail493 ingredient107)
)
 (:goal
  (and
      (contains shot283 cocktail493)
)))
