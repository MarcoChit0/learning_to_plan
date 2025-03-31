(define (problem prob)
 (:domain barman)
 (:objects 
      shaker464 - shaker
      left right - hand
      shot18 - shot
      ingredient39 ingredient293 ingredient20 ingredient16 - ingredient
      cocktail321 - cocktail
      dispenser311 dispenser207 dispenser238 dispenser143 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker464)
  (ontable shot18)
  (dispenses dispenser311 ingredient39)
  (dispenses dispenser207 ingredient293)
  (dispenses dispenser238 ingredient20)
  (dispenses dispenser143 ingredient16)
  (clean shaker464)
  (clean shot18)
  (empty shaker464)
  (empty shot18)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker464 l0)
  (shaker-level shaker464 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail321 ingredient39)
  (cocktail-part2 cocktail321 ingredient293)
)
 (:goal
  (and
      (contains shot18 cocktail321)
)))
