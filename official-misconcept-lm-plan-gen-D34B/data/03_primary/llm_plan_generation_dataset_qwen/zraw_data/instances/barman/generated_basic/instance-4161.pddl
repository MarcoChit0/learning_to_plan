(define (problem prob)
 (:domain barman)
 (:objects 
      shaker463 - shaker
      left right - hand
      shot371 shot317 - shot
      ingredient220 ingredient288 - ingredient
      cocktail480 - cocktail
      dispenser449 dispenser226 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker463)
  (ontable shot371)
  (ontable shot317)
  (dispenses dispenser449 ingredient220)
  (dispenses dispenser226 ingredient288)
  (clean shaker463)
  (clean shot371)
  (clean shot317)
  (empty shaker463)
  (empty shot371)
  (empty shot317)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker463 l0)
  (shaker-level shaker463 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail480 ingredient288)
  (cocktail-part2 cocktail480 ingredient220)
)
 (:goal
  (and
      (contains shot371 cocktail480)
)))
