(define (problem prob)
 (:domain barman)
 (:objects 
      shaker111 - shaker
      left right - hand
      shot78 shot82 - shot
      ingredient387 ingredient343 ingredient443 ingredient466 - ingredient
      cocktail277 - cocktail
      dispenser392 dispenser74 dispenser197 dispenser43 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker111)
  (ontable shot78)
  (ontable shot82)
  (dispenses dispenser392 ingredient387)
  (dispenses dispenser74 ingredient343)
  (dispenses dispenser197 ingredient443)
  (dispenses dispenser43 ingredient466)
  (clean shaker111)
  (clean shot78)
  (clean shot82)
  (empty shaker111)
  (empty shot78)
  (empty shot82)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker111 l0)
  (shaker-level shaker111 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail277 ingredient343)
  (cocktail-part2 cocktail277 ingredient466)
)
 (:goal
  (and
      (contains shot78 cocktail277)
)))
