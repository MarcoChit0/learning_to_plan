(define (problem prob)
 (:domain barman)
 (:objects 
      shaker116 - shaker
      left right - hand
      shot193 shot98 shot275 - shot
      ingredient391 ingredient370 ingredient441 ingredient79 - ingredient
      cocktail1 - cocktail
      dispenser83 dispenser212 dispenser445 dispenser104 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker116)
  (ontable shot193)
  (ontable shot98)
  (ontable shot275)
  (dispenses dispenser83 ingredient391)
  (dispenses dispenser212 ingredient370)
  (dispenses dispenser445 ingredient441)
  (dispenses dispenser104 ingredient79)
  (clean shaker116)
  (clean shot193)
  (clean shot98)
  (clean shot275)
  (empty shaker116)
  (empty shot193)
  (empty shot98)
  (empty shot275)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker116 l0)
  (shaker-level shaker116 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient441)
  (cocktail-part2 cocktail1 ingredient79)
)
 (:goal
  (and
      (contains shot193 cocktail1)
      (contains shot98 ingredient391)
)))
