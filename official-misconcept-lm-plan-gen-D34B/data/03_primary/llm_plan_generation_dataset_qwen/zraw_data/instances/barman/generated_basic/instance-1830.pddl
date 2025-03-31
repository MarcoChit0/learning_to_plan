(define (problem prob)
 (:domain barman)
 (:objects 
      shaker152 - shaker
      left right - hand
      shot393 shot85 shot491 - shot
      ingredient283 ingredient194 ingredient488 ingredient358 - ingredient
      cocktail85 - cocktail
      dispenser155 dispenser275 dispenser185 dispenser384 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker152)
  (ontable shot393)
  (ontable shot85)
  (ontable shot491)
  (dispenses dispenser155 ingredient283)
  (dispenses dispenser275 ingredient194)
  (dispenses dispenser185 ingredient488)
  (dispenses dispenser384 ingredient358)
  (clean shaker152)
  (clean shot393)
  (clean shot85)
  (clean shot491)
  (empty shaker152)
  (empty shot393)
  (empty shot85)
  (empty shot491)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker152 l0)
  (shaker-level shaker152 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail85 ingredient194)
  (cocktail-part2 cocktail85 ingredient283)
)
 (:goal
  (and
      (contains shot393 cocktail85)
      (contains shot85 ingredient194)
)))
