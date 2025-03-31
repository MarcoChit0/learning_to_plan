(define (problem prob)
 (:domain barman)
 (:objects 
      shaker488 - shaker
      left right - hand
      shot317 shot180 shot243 - shot
      ingredient138 ingredient288 ingredient336 ingredient134 - ingredient
      cocktail1 - cocktail
      dispenser428 dispenser210 dispenser344 dispenser213 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker488)
  (ontable shot317)
  (ontable shot180)
  (ontable shot243)
  (dispenses dispenser428 ingredient138)
  (dispenses dispenser210 ingredient288)
  (dispenses dispenser344 ingredient336)
  (dispenses dispenser213 ingredient134)
  (clean shaker488)
  (clean shot317)
  (clean shot180)
  (clean shot243)
  (empty shaker488)
  (empty shot317)
  (empty shot180)
  (empty shot243)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker488 l0)
  (shaker-level shaker488 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient288)
  (cocktail-part2 cocktail1 ingredient134)
)
 (:goal
  (and
      (contains shot317 cocktail1)
      (contains shot180 cocktail1)
)))
