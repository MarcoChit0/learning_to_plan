(define (problem prob)
 (:domain barman)
 (:objects 
      shaker170 - shaker
      left right - hand
      shot77 shot349 - shot
      ingredient79 ingredient346 - ingredient
      cocktail474 - cocktail
      dispenser12 dispenser116 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker170)
  (ontable shot77)
  (ontable shot349)
  (dispenses dispenser12 ingredient79)
  (dispenses dispenser116 ingredient346)
  (clean shaker170)
  (clean shot77)
  (clean shot349)
  (empty shaker170)
  (empty shot77)
  (empty shot349)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker170 l0)
  (shaker-level shaker170 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail474 ingredient79)
  (cocktail-part2 cocktail474 ingredient346)
)
 (:goal
  (and
      (contains shot77 cocktail474)
)))
