(define (problem prob)
 (:domain barman)
 (:objects 
      shaker247 - shaker
      left right - hand
      shot85 shot387 shot105 - shot
      ingredient127 ingredient155 - ingredient
      cocktail323 - cocktail
      dispenser332 dispenser64 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker247)
  (ontable shot85)
  (ontable shot387)
  (ontable shot105)
  (dispenses dispenser332 ingredient127)
  (dispenses dispenser64 ingredient155)
  (clean shaker247)
  (clean shot85)
  (clean shot387)
  (clean shot105)
  (empty shaker247)
  (empty shot85)
  (empty shot387)
  (empty shot105)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker247 l0)
  (shaker-level shaker247 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail323 ingredient155)
  (cocktail-part2 cocktail323 ingredient127)
)
 (:goal
  (and
      (contains shot85 cocktail323)
      (contains shot387 cocktail323)
)))
