(define (problem prob)
 (:domain barman)
 (:objects 
      shaker270 - shaker
      left right - hand
      shot486 shot221 shot224 - shot
      ingredient47 ingredient38 ingredient477 ingredient399 - ingredient
      cocktail88 - cocktail
      dispenser381 dispenser480 dispenser354 dispenser116 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker270)
  (ontable shot486)
  (ontable shot221)
  (ontable shot224)
  (dispenses dispenser381 ingredient47)
  (dispenses dispenser480 ingredient38)
  (dispenses dispenser354 ingredient477)
  (dispenses dispenser116 ingredient399)
  (clean shaker270)
  (clean shot486)
  (clean shot221)
  (clean shot224)
  (empty shaker270)
  (empty shot486)
  (empty shot221)
  (empty shot224)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker270 l0)
  (shaker-level shaker270 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail88 ingredient47)
  (cocktail-part2 cocktail88 ingredient399)
)
 (:goal
  (and
      (contains shot486 cocktail88)
      (contains shot221 ingredient477)
)))
