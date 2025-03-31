(define (problem prob)
 (:domain barman)
 (:objects 
      shaker307 - shaker
      left right - hand
      shot434 - shot
      ingredient399 ingredient77 - ingredient
      cocktail175 - cocktail
      dispenser443 dispenser19 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker307)
  (ontable shot434)
  (dispenses dispenser443 ingredient399)
  (dispenses dispenser19 ingredient77)
  (clean shaker307)
  (clean shot434)
  (empty shaker307)
  (empty shot434)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker307 l0)
  (shaker-level shaker307 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail175 ingredient77)
  (cocktail-part2 cocktail175 ingredient399)
)
 (:goal
  (and
      (contains shot434 cocktail175)
)))
