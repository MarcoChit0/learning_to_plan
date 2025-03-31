(define (problem prob)
 (:domain barman)
 (:objects 
      shaker311 - shaker
      left right - hand
      shot29 shot143 shot126 - shot
      ingredient365 ingredient16 ingredient310 ingredient96 - ingredient
      cocktail338 - cocktail
      dispenser233 dispenser325 dispenser238 dispenser118 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker311)
  (ontable shot29)
  (ontable shot143)
  (ontable shot126)
  (dispenses dispenser233 ingredient365)
  (dispenses dispenser325 ingredient16)
  (dispenses dispenser238 ingredient310)
  (dispenses dispenser118 ingredient96)
  (clean shaker311)
  (clean shot29)
  (clean shot143)
  (clean shot126)
  (empty shaker311)
  (empty shot29)
  (empty shot143)
  (empty shot126)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker311 l0)
  (shaker-level shaker311 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail338 ingredient96)
  (cocktail-part2 cocktail338 ingredient310)
)
 (:goal
  (and
      (contains shot29 cocktail338)
      (contains shot143 ingredient310)
)))
