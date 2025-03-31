(define (problem prob)
 (:domain barman)
 (:objects 
      shaker334 - shaker
      left right - hand
      shot393 shot289 shot341 - shot
      ingredient8 ingredient346 ingredient203 - ingredient
      cocktail217 - cocktail
      dispenser296 dispenser377 dispenser290 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker334)
  (ontable shot393)
  (ontable shot289)
  (ontable shot341)
  (dispenses dispenser296 ingredient8)
  (dispenses dispenser377 ingredient346)
  (dispenses dispenser290 ingredient203)
  (clean shaker334)
  (clean shot393)
  (clean shot289)
  (clean shot341)
  (empty shaker334)
  (empty shot393)
  (empty shot289)
  (empty shot341)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker334 l0)
  (shaker-level shaker334 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail217 ingredient203)
  (cocktail-part2 cocktail217 ingredient346)
)
 (:goal
  (and
      (contains shot393 cocktail217)
      (contains shot289 cocktail217)
)))
