(define (problem prob)
 (:domain barman)
 (:objects 
      shaker89 - shaker
      left right - hand
      shot436 shot306 shot461 - shot
      ingredient221 ingredient124 ingredient175 - ingredient
      cocktail49 - cocktail
      dispenser405 dispenser463 dispenser35 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker89)
  (ontable shot436)
  (ontable shot306)
  (ontable shot461)
  (dispenses dispenser405 ingredient221)
  (dispenses dispenser463 ingredient124)
  (dispenses dispenser35 ingredient175)
  (clean shaker89)
  (clean shot436)
  (clean shot306)
  (clean shot461)
  (empty shaker89)
  (empty shot436)
  (empty shot306)
  (empty shot461)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker89 l0)
  (shaker-level shaker89 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail49 ingredient175)
  (cocktail-part2 cocktail49 ingredient221)
)
 (:goal
  (and
      (contains shot436 cocktail49)
      (contains shot306 cocktail49)
)))
