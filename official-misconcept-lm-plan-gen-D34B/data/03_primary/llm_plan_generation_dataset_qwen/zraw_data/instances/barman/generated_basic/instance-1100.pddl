(define (problem prob)
 (:domain barman)
 (:objects 
      shaker488 - shaker
      left right - hand
      shot326 - shot
      ingredient299 ingredient463 ingredient298 - ingredient
      cocktail1 - cocktail
      dispenser241 dispenser79 dispenser322 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker488)
  (ontable shot326)
  (dispenses dispenser241 ingredient299)
  (dispenses dispenser79 ingredient463)
  (dispenses dispenser322 ingredient298)
  (clean shaker488)
  (clean shot326)
  (empty shaker488)
  (empty shot326)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker488 l0)
  (shaker-level shaker488 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient298)
  (cocktail-part2 cocktail1 ingredient299)
)
 (:goal
  (and
      (contains shot326 cocktail1)
)))
