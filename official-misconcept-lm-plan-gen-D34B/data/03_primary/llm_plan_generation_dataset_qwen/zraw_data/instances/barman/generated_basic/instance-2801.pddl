(define (problem prob)
 (:domain barman)
 (:objects 
      shaker221 - shaker
      left right - hand
      shot114 shot294 - shot
      ingredient341 ingredient324 - ingredient
      cocktail180 - cocktail
      dispenser138 dispenser73 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker221)
  (ontable shot114)
  (ontable shot294)
  (dispenses dispenser138 ingredient341)
  (dispenses dispenser73 ingredient324)
  (clean shaker221)
  (clean shot114)
  (clean shot294)
  (empty shaker221)
  (empty shot114)
  (empty shot294)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker221 l0)
  (shaker-level shaker221 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail180 ingredient324)
  (cocktail-part2 cocktail180 ingredient341)
)
 (:goal
  (and
      (contains shot114 cocktail180)
)))
