(define (problem prob)
 (:domain barman)
 (:objects 
      shaker447 - shaker
      left right - hand
      shot370 shot107 shot230 - shot
      ingredient449 ingredient401 ingredient85 ingredient156 - ingredient
      cocktail152 - cocktail
      dispenser473 dispenser261 dispenser441 dispenser148 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker447)
  (ontable shot370)
  (ontable shot107)
  (ontable shot230)
  (dispenses dispenser473 ingredient449)
  (dispenses dispenser261 ingredient401)
  (dispenses dispenser441 ingredient85)
  (dispenses dispenser148 ingredient156)
  (clean shaker447)
  (clean shot370)
  (clean shot107)
  (clean shot230)
  (empty shaker447)
  (empty shot370)
  (empty shot107)
  (empty shot230)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker447 l0)
  (shaker-level shaker447 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail152 ingredient156)
  (cocktail-part2 cocktail152 ingredient401)
)
 (:goal
  (and
      (contains shot370 cocktail152)
      (contains shot107 cocktail152)
)))
