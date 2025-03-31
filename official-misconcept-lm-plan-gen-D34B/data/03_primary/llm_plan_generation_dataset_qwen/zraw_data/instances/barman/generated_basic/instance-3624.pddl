(define (problem prob)
 (:domain barman)
 (:objects 
      shaker134 - shaker
      left right - hand
      shot440 shot27 shot176 - shot
      ingredient337 ingredient188 ingredient447 - ingredient
      cocktail327 - cocktail
      dispenser311 dispenser192 dispenser405 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker134)
  (ontable shot440)
  (ontable shot27)
  (ontable shot176)
  (dispenses dispenser311 ingredient337)
  (dispenses dispenser192 ingredient188)
  (dispenses dispenser405 ingredient447)
  (clean shaker134)
  (clean shot440)
  (clean shot27)
  (clean shot176)
  (empty shaker134)
  (empty shot440)
  (empty shot27)
  (empty shot176)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker134 l0)
  (shaker-level shaker134 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail327 ingredient337)
  (cocktail-part2 cocktail327 ingredient188)
)
 (:goal
  (and
      (contains shot440 cocktail327)
      (contains shot27 cocktail327)
)))
