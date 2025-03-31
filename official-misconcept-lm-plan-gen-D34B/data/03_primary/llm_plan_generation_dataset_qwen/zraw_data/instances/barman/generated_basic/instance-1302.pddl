(define (problem prob)
 (:domain barman)
 (:objects 
      shaker10 - shaker
      left right - hand
      shot20 shot352 shot105 - shot
      ingredient446 ingredient173 - ingredient
      cocktail169 - cocktail
      dispenser14 dispenser198 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker10)
  (ontable shot20)
  (ontable shot352)
  (ontable shot105)
  (dispenses dispenser14 ingredient446)
  (dispenses dispenser198 ingredient173)
  (clean shaker10)
  (clean shot20)
  (clean shot352)
  (clean shot105)
  (empty shaker10)
  (empty shot20)
  (empty shot352)
  (empty shot105)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker10 l0)
  (shaker-level shaker10 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail169 ingredient446)
  (cocktail-part2 cocktail169 ingredient173)
)
 (:goal
  (and
      (contains shot20 cocktail169)
      (contains shot352 cocktail169)
)))
