(define (problem prob)
 (:domain barman)
 (:objects 
      shaker42 - shaker
      left right - hand
      shot340 shot372 shot298 shot483 - shot
      ingredient263 ingredient227 ingredient75 - ingredient
      cocktail128 - cocktail
      dispenser307 dispenser138 dispenser184 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker42)
  (ontable shot340)
  (ontable shot372)
  (ontable shot298)
  (ontable shot483)
  (dispenses dispenser307 ingredient263)
  (dispenses dispenser138 ingredient227)
  (dispenses dispenser184 ingredient75)
  (clean shaker42)
  (clean shot340)
  (clean shot372)
  (clean shot298)
  (clean shot483)
  (empty shaker42)
  (empty shot340)
  (empty shot372)
  (empty shot298)
  (empty shot483)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker42 l0)
  (shaker-level shaker42 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail128 ingredient227)
  (cocktail-part2 cocktail128 ingredient75)
)
 (:goal
  (and
      (contains shot340 cocktail128)
      (contains shot372 ingredient75)
      (contains shot298 cocktail128)
)))
