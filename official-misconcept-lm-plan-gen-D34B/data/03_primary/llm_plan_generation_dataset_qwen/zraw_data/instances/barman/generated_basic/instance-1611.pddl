(define (problem prob)
 (:domain barman)
 (:objects 
      shaker392 - shaker
      left right - hand
      shot28 shot0 - shot
      ingredient418 ingredient78 ingredient339 - ingredient
      cocktail128 - cocktail
      dispenser365 dispenser163 dispenser419 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker392)
  (ontable shot28)
  (ontable shot0)
  (dispenses dispenser365 ingredient418)
  (dispenses dispenser163 ingredient78)
  (dispenses dispenser419 ingredient339)
  (clean shaker392)
  (clean shot28)
  (clean shot0)
  (empty shaker392)
  (empty shot28)
  (empty shot0)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker392 l0)
  (shaker-level shaker392 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail128 ingredient418)
  (cocktail-part2 cocktail128 ingredient78)
)
 (:goal
  (and
      (contains shot28 cocktail128)
)))
