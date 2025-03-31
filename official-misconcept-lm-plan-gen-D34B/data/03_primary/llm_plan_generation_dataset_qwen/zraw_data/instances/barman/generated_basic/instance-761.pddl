(define (problem prob)
 (:domain barman)
 (:objects 
      shaker319 - shaker
      left right - hand
      shot265 shot253 shot155 - shot
      ingredient420 ingredient48 ingredient463 - ingredient
      cocktail1 - cocktail
      dispenser227 dispenser296 dispenser0 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker319)
  (ontable shot265)
  (ontable shot253)
  (ontable shot155)
  (dispenses dispenser227 ingredient420)
  (dispenses dispenser296 ingredient48)
  (dispenses dispenser0 ingredient463)
  (clean shaker319)
  (clean shot265)
  (clean shot253)
  (clean shot155)
  (empty shaker319)
  (empty shot265)
  (empty shot253)
  (empty shot155)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker319 l0)
  (shaker-level shaker319 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient48)
  (cocktail-part2 cocktail1 ingredient420)
)
 (:goal
  (and
      (contains shot265 cocktail1)
      (contains shot253 cocktail1)
)))
