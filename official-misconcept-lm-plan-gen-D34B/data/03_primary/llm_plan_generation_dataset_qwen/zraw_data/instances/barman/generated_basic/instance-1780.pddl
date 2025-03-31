(define (problem prob)
 (:domain barman)
 (:objects 
      shaker159 - shaker
      left right - hand
      shot322 shot236 - shot
      ingredient436 ingredient264 ingredient365 - ingredient
      cocktail133 - cocktail
      dispenser129 dispenser370 dispenser319 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker159)
  (ontable shot322)
  (ontable shot236)
  (dispenses dispenser129 ingredient436)
  (dispenses dispenser370 ingredient264)
  (dispenses dispenser319 ingredient365)
  (clean shaker159)
  (clean shot322)
  (clean shot236)
  (empty shaker159)
  (empty shot322)
  (empty shot236)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker159 l0)
  (shaker-level shaker159 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail133 ingredient436)
  (cocktail-part2 cocktail133 ingredient365)
)
 (:goal
  (and
      (contains shot322 cocktail133)
)))
