(define (problem prob)
 (:domain barman)
 (:objects 
      shaker277 - shaker
      left right - hand
      shot263 shot429 - shot
      ingredient464 ingredient401 ingredient444 - ingredient
      cocktail397 - cocktail
      dispenser88 dispenser383 dispenser257 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker277)
  (ontable shot263)
  (ontable shot429)
  (dispenses dispenser88 ingredient464)
  (dispenses dispenser383 ingredient401)
  (dispenses dispenser257 ingredient444)
  (clean shaker277)
  (clean shot263)
  (clean shot429)
  (empty shaker277)
  (empty shot263)
  (empty shot429)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker277 l0)
  (shaker-level shaker277 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail397 ingredient444)
  (cocktail-part2 cocktail397 ingredient401)
)
 (:goal
  (and
      (contains shot263 cocktail397)
)))
