(define (problem prob)
 (:domain barman)
 (:objects 
      shaker273 - shaker
      left right - hand
      shot29 shot10 - shot
      ingredient221 ingredient289 ingredient352 - ingredient
      cocktail1 - cocktail
      dispenser448 dispenser317 dispenser194 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker273)
  (ontable shot29)
  (ontable shot10)
  (dispenses dispenser448 ingredient221)
  (dispenses dispenser317 ingredient289)
  (dispenses dispenser194 ingredient352)
  (clean shaker273)
  (clean shot29)
  (clean shot10)
  (empty shaker273)
  (empty shot29)
  (empty shot10)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker273 l0)
  (shaker-level shaker273 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient289)
  (cocktail-part2 cocktail1 ingredient352)
)
 (:goal
  (and
      (contains shot29 cocktail1)
)))
