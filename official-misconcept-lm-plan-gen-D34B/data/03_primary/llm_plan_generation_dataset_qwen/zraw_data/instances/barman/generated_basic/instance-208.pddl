(define (problem prob)
 (:domain barman)
 (:objects 
      shaker313 - shaker
      left right - hand
      shot191 - shot
      ingredient59 ingredient472 ingredient127 - ingredient
      cocktail1 - cocktail
      dispenser180 dispenser37 dispenser240 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker313)
  (ontable shot191)
  (dispenses dispenser180 ingredient59)
  (dispenses dispenser37 ingredient472)
  (dispenses dispenser240 ingredient127)
  (clean shaker313)
  (clean shot191)
  (empty shaker313)
  (empty shot191)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker313 l0)
  (shaker-level shaker313 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient127)
  (cocktail-part2 cocktail1 ingredient59)
)
 (:goal
  (and
      (contains shot191 cocktail1)
)))
