(define (problem prob)
 (:domain barman)
 (:objects 
      shaker179 - shaker
      left right - hand
      shot28 shot334 shot392 - shot
      ingredient331 ingredient235 ingredient297 - ingredient
      cocktail231 - cocktail
      dispenser383 dispenser366 dispenser149 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker179)
  (ontable shot28)
  (ontable shot334)
  (ontable shot392)
  (dispenses dispenser383 ingredient331)
  (dispenses dispenser366 ingredient235)
  (dispenses dispenser149 ingredient297)
  (clean shaker179)
  (clean shot28)
  (clean shot334)
  (clean shot392)
  (empty shaker179)
  (empty shot28)
  (empty shot334)
  (empty shot392)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker179 l0)
  (shaker-level shaker179 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail231 ingredient331)
  (cocktail-part2 cocktail231 ingredient235)
)
 (:goal
  (and
      (contains shot28 cocktail231)
      (contains shot334 cocktail231)
)))
