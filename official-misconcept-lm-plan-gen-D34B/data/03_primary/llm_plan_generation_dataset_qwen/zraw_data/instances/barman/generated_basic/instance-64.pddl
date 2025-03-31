(define (problem prob)
 (:domain barman)
 (:objects 
      shaker146 - shaker
      left right - hand
      shot317 shot161 - shot
      ingredient489 ingredient371 ingredient427 - ingredient
      cocktail1 - cocktail
      dispenser305 dispenser33 dispenser119 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker146)
  (ontable shot317)
  (ontable shot161)
  (dispenses dispenser305 ingredient489)
  (dispenses dispenser33 ingredient371)
  (dispenses dispenser119 ingredient427)
  (clean shaker146)
  (clean shot317)
  (clean shot161)
  (empty shaker146)
  (empty shot317)
  (empty shot161)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker146 l0)
  (shaker-level shaker146 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient427)
  (cocktail-part2 cocktail1 ingredient371)
)
 (:goal
  (and
      (contains shot317 cocktail1)
)))
