(define (problem prob)
 (:domain barman)
 (:objects 
      shaker133 - shaker
      left right - hand
      shot285 shot5 shot429 - shot
      ingredient307 ingredient369 ingredient6 ingredient322 - ingredient
      cocktail114 - cocktail
      dispenser374 dispenser48 dispenser53 dispenser154 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker133)
  (ontable shot285)
  (ontable shot5)
  (ontable shot429)
  (dispenses dispenser374 ingredient307)
  (dispenses dispenser48 ingredient369)
  (dispenses dispenser53 ingredient6)
  (dispenses dispenser154 ingredient322)
  (clean shaker133)
  (clean shot285)
  (clean shot5)
  (clean shot429)
  (empty shaker133)
  (empty shot285)
  (empty shot5)
  (empty shot429)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker133 l0)
  (shaker-level shaker133 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail114 ingredient322)
  (cocktail-part2 cocktail114 ingredient369)
)
 (:goal
  (and
      (contains shot285 cocktail114)
      (contains shot5 cocktail114)
)))
