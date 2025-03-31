(define (problem prob)
 (:domain barman)
 (:objects 
      shaker36 - shaker
      left right - hand
      shot239 shot489 - shot
      ingredient172 ingredient401 - ingredient
      cocktail142 - cocktail
      dispenser29 dispenser206 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker36)
  (ontable shot239)
  (ontable shot489)
  (dispenses dispenser29 ingredient172)
  (dispenses dispenser206 ingredient401)
  (clean shaker36)
  (clean shot239)
  (clean shot489)
  (empty shaker36)
  (empty shot239)
  (empty shot489)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker36 l0)
  (shaker-level shaker36 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail142 ingredient401)
  (cocktail-part2 cocktail142 ingredient172)
)
 (:goal
  (and
      (contains shot239 cocktail142)
)))
