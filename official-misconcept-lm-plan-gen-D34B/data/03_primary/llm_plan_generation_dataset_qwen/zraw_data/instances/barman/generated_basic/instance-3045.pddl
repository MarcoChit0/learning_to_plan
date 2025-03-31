(define (problem prob)
 (:domain barman)
 (:objects 
      shaker41 - shaker
      left right - hand
      shot444 shot317 - shot
      ingredient489 ingredient371 ingredient291 - ingredient
      cocktail367 - cocktail
      dispenser264 dispenser273 dispenser271 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker41)
  (ontable shot444)
  (ontable shot317)
  (dispenses dispenser264 ingredient489)
  (dispenses dispenser273 ingredient371)
  (dispenses dispenser271 ingredient291)
  (clean shaker41)
  (clean shot444)
  (clean shot317)
  (empty shaker41)
  (empty shot444)
  (empty shot317)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker41 l0)
  (shaker-level shaker41 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail367 ingredient291)
  (cocktail-part2 cocktail367 ingredient371)
)
 (:goal
  (and
      (contains shot444 cocktail367)
)))
