(define (problem prob)
 (:domain barman)
 (:objects 
      shaker425 - shaker
      left right - hand
      shot486 shot489 - shot
      ingredient471 ingredient307 ingredient341 - ingredient
      cocktail162 - cocktail
      dispenser132 dispenser313 dispenser11 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker425)
  (ontable shot486)
  (ontable shot489)
  (dispenses dispenser132 ingredient471)
  (dispenses dispenser313 ingredient307)
  (dispenses dispenser11 ingredient341)
  (clean shaker425)
  (clean shot486)
  (clean shot489)
  (empty shaker425)
  (empty shot486)
  (empty shot489)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker425 l0)
  (shaker-level shaker425 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail162 ingredient341)
  (cocktail-part2 cocktail162 ingredient471)
)
 (:goal
  (and
      (contains shot486 cocktail162)
)))
