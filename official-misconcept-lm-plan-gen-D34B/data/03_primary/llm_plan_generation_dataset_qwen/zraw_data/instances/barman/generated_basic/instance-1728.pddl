(define (problem prob)
 (:domain barman)
 (:objects 
      shaker191 - shaker
      left right - hand
      shot236 shot67 - shot
      ingredient374 ingredient220 ingredient124 ingredient138 - ingredient
      cocktail419 - cocktail
      dispenser188 dispenser436 dispenser158 dispenser323 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker191)
  (ontable shot236)
  (ontable shot67)
  (dispenses dispenser188 ingredient374)
  (dispenses dispenser436 ingredient220)
  (dispenses dispenser158 ingredient124)
  (dispenses dispenser323 ingredient138)
  (clean shaker191)
  (clean shot236)
  (clean shot67)
  (empty shaker191)
  (empty shot236)
  (empty shot67)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker191 l0)
  (shaker-level shaker191 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail419 ingredient138)
  (cocktail-part2 cocktail419 ingredient220)
)
 (:goal
  (and
      (contains shot236 cocktail419)
)))
