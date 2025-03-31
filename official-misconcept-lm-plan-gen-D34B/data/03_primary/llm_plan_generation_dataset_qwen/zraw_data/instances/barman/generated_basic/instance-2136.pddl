(define (problem prob)
 (:domain barman)
 (:objects 
      shaker15 - shaker
      left right - hand
      shot492 shot190 shot485 - shot
      ingredient440 ingredient225 - ingredient
      cocktail460 - cocktail
      dispenser494 dispenser114 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker15)
  (ontable shot492)
  (ontable shot190)
  (ontable shot485)
  (dispenses dispenser494 ingredient440)
  (dispenses dispenser114 ingredient225)
  (clean shaker15)
  (clean shot492)
  (clean shot190)
  (clean shot485)
  (empty shaker15)
  (empty shot492)
  (empty shot190)
  (empty shot485)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker15 l0)
  (shaker-level shaker15 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail460 ingredient440)
  (cocktail-part2 cocktail460 ingredient225)
)
 (:goal
  (and
      (contains shot492 cocktail460)
      (contains shot190 ingredient440)
)))
