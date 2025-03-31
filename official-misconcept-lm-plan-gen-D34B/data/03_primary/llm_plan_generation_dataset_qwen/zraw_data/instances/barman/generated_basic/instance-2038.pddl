(define (problem prob)
 (:domain barman)
 (:objects 
      shaker176 - shaker
      left right - hand
      shot368 shot394 - shot
      ingredient125 ingredient136 - ingredient
      cocktail418 - cocktail
      dispenser138 dispenser157 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker176)
  (ontable shot368)
  (ontable shot394)
  (dispenses dispenser138 ingredient125)
  (dispenses dispenser157 ingredient136)
  (clean shaker176)
  (clean shot368)
  (clean shot394)
  (empty shaker176)
  (empty shot368)
  (empty shot394)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker176 l0)
  (shaker-level shaker176 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail418 ingredient125)
  (cocktail-part2 cocktail418 ingredient136)
)
 (:goal
  (and
      (contains shot368 cocktail418)
)))
