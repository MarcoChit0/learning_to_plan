(define (problem prob)
 (:domain barman)
 (:objects 
      shaker395 - shaker
      left right - hand
      shot93 shot293 shot118 shot440 - shot
      ingredient136 ingredient4 ingredient251 - ingredient
      cocktail467 - cocktail
      dispenser364 dispenser323 dispenser44 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker395)
  (ontable shot93)
  (ontable shot293)
  (ontable shot118)
  (ontable shot440)
  (dispenses dispenser364 ingredient136)
  (dispenses dispenser323 ingredient4)
  (dispenses dispenser44 ingredient251)
  (clean shaker395)
  (clean shot93)
  (clean shot293)
  (clean shot118)
  (clean shot440)
  (empty shaker395)
  (empty shot93)
  (empty shot293)
  (empty shot118)
  (empty shot440)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker395 l0)
  (shaker-level shaker395 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail467 ingredient251)
  (cocktail-part2 cocktail467 ingredient4)
)
 (:goal
  (and
      (contains shot93 cocktail467)
      (contains shot293 cocktail467)
      (contains shot118 ingredient251)
)))
