(define (problem prob)
 (:domain barman)
 (:objects 
      shaker167 - shaker
      left right - hand
      shot296 shot251 shot341 - shot
      ingredient264 ingredient57 ingredient139 - ingredient
      cocktail124 - cocktail
      dispenser70 dispenser49 dispenser244 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker167)
  (ontable shot296)
  (ontable shot251)
  (ontable shot341)
  (dispenses dispenser70 ingredient264)
  (dispenses dispenser49 ingredient57)
  (dispenses dispenser244 ingredient139)
  (clean shaker167)
  (clean shot296)
  (clean shot251)
  (clean shot341)
  (empty shaker167)
  (empty shot296)
  (empty shot251)
  (empty shot341)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker167 l0)
  (shaker-level shaker167 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail124 ingredient139)
  (cocktail-part2 cocktail124 ingredient264)
)
 (:goal
  (and
      (contains shot296 cocktail124)
      (contains shot251 cocktail124)
)))
