(define (problem prob)
 (:domain barman)
 (:objects 
      shaker59 - shaker
      left right - hand
      shot392 - shot
      ingredient305 ingredient248 ingredient199 ingredient403 - ingredient
      cocktail332 - cocktail
      dispenser405 dispenser170 dispenser103 dispenser202 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker59)
  (ontable shot392)
  (dispenses dispenser405 ingredient305)
  (dispenses dispenser170 ingredient248)
  (dispenses dispenser103 ingredient199)
  (dispenses dispenser202 ingredient403)
  (clean shaker59)
  (clean shot392)
  (empty shaker59)
  (empty shot392)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker59 l0)
  (shaker-level shaker59 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail332 ingredient248)
  (cocktail-part2 cocktail332 ingredient403)
)
 (:goal
  (and
      (contains shot392 cocktail332)
)))
